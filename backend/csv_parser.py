"""
CSV parsing and validation for batch video analysis.

Handles parsing CSV files with video data and extracting politician names.
"""

import csv
import re
import io
import logging
from typing import List, Tuple, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_valid_youtube_url(url: str) -> bool:
    """
    Validate if a URL is a valid YouTube URL.

    Args:
        url: URL to validate

    Returns:
        True if valid YouTube URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc in [
            'youtube.com', 'www.youtube.com',
            'youtu.be', 'www.youtu.be',
            'm.youtube.com'
        ] and bool(parsed.path or parsed.query)
    except Exception:
        return False


def extract_politician_name(title: str) -> str:
    """
    Extract politician name from video title.

    The CSV format has titles like:
    - "נאום ראש הממשלה בנימין נתניהו..." (contains Netanyahu)
    - "נאום רה\"מ נפתלי בנט..." (contains Bennett)

    This function attempts to extract the politician name.
    For Hebrew/Arabic text, we look for common patterns.

    Args:
        title: Video title from CSV

    Returns:
        Extracted politician name or "Unknown"
    """
    if not title:
        return "Unknown"

    # Common name patterns (Hebrew and English)
    # Format: (pattern, extracted_name)
    patterns = [
        (r'נתניהו|netanyahu', 'Benjamin Netanyahu'),
        (r'בנט|bennett', 'Naftali Bennett'),
        (r'לפיד|lapid', 'Yair Lapid'),
        (r'גנץ|gantz', 'Benny Gantz'),
        (r'ליברמן|lieberman', 'Avigdor Lieberman'),
    ]

    title_lower = title.lower()

    for pattern, name in patterns:
        if re.search(pattern, title_lower, re.IGNORECASE):
            return name

    # If no known politician found, try to extract from "רה\"מ" or "ראש הממשלה" patterns
    # These mean "Prime Minister" in Hebrew
    pm_match = re.search(r'(?:רה["\']מ|ראש הממשלה)\s+([א-ת\s]+?)(?:\s+ב(?:פני|עצרת|קונגרס)|$)', title)
    if pm_match:
        name = pm_match.group(1).strip()
        if name:
            return name

    # Default to "Unknown" if we can't extract
    logger.warning(f"Could not extract politician name from title: {title}")
    return "Unknown"


def parse_csv(source: str, is_file_path: bool = True) -> List[Tuple[str, str, str]]:
    """
    Parse CSV data and extract video information.

    Expected CSV format:
        date,name,url

    Args:
        source: Either file path or CSV string data
        is_file_path: If True, source is a file path; if False, source is CSV string

    Returns:
        List of tuples: (date, politician_name, url)

    Raises:
        ValueError: If CSV format is invalid
        FileNotFoundError: If file path doesn't exist (when is_file_path=True)
    """
    videos = []
    seen_urls = set()
    row_number = 0

    try:
        if is_file_path:
            with open(source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        else:
            reader = csv.DictReader(io.StringIO(source))
            rows = list(reader)

        for row in rows:
            row_number += 1

            # Validate required fields
            if 'date' not in row or 'name' not in row or 'url' not in row:
                logger.warning(
                    f"Row {row_number}: Missing required fields (date, name, url). "
                    f"Available fields: {list(row.keys())}"
                )
                continue

            date = row['date'].strip()
            title = row['name'].strip()
            url = row['url'].strip()

            # Validate date format (YYYY-MM-DD)
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
                logger.warning(
                    f"Row {row_number}: Invalid date format '{date}'. "
                    f"Expected YYYY-MM-DD"
                )
                # Continue anyway, but log the warning

            # Validate URL
            if not url:
                logger.warning(f"Row {row_number}: Empty URL, skipping")
                continue

            if not is_valid_youtube_url(url):
                logger.warning(
                    f"Row {row_number}: Invalid YouTube URL '{url}', skipping"
                )
                continue

            # Check for duplicates
            if url in seen_urls:
                logger.warning(
                    f"Row {row_number}: Duplicate URL '{url}', skipping"
                )
                continue

            # Extract politician name from title
            politician_name = extract_politician_name(title)

            videos.append((date, politician_name, url))
            seen_urls.add(url)

            logger.debug(
                f"Row {row_number}: Parsed video - "
                f"Date: {date}, Politician: {politician_name}, URL: {url[:50]}..."
            )

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {source}")
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {e}")

    logger.info(f"Parsed {len(videos)} valid videos from CSV")

    if len(videos) == 0:
        logger.warning("No valid videos found in CSV")

    return videos


def validate_csv_format(source: str, is_file_path: bool = True) -> Tuple[bool, str]:
    """
    Validate CSV format without fully parsing.

    Args:
        source: Either file path or CSV string data
        is_file_path: If True, source is a file path; if False, source is CSV string

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if is_file_path:
            with open(source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
        else:
            reader = csv.DictReader(io.StringIO(source))
            headers = reader.fieldnames

        if not headers:
            return False, "CSV file is empty or has no headers"

        required_fields = {'date', 'name', 'url'}
        missing_fields = required_fields - set(headers)

        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

        return True, "CSV format is valid"

    except FileNotFoundError:
        return False, f"File not found: {source}"
    except Exception as e:
        return False, f"Error reading CSV: {e}"
