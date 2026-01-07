"""
SQLite database for batch video analysis results.

This module provides database operations for storing and retrieving
video analysis results with caching to avoid duplicate processing.

Schema: Hybrid approach with indexed metadata + JSON blob for full results
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Default database path (relative to backend directory)
DEFAULT_DB_PATH = "batch_results.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get a connection to the SQLite database.

    Args:
        db_path: Path to the database file

    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize the database schema.

    Creates the videos table with hybrid schema:
    - Indexed metadata columns for fast queries
    - JSON blob for complete analysis results

    Args:
        db_path: Path to the database file
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Create videos table with hybrid schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            url TEXT PRIMARY KEY,
            date TEXT,
            politician_name TEXT,
            title TEXT,
            transcription TEXT,
            language TEXT,
            analysis_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT
        )
    """)

    # Create indices for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_politician_name
        ON videos(politician_name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_date
        ON videos(date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status
        ON videos(status)
    """)

    conn.commit()
    conn.close()

    logger.info(f"Database initialized at {db_path}")


def check_video_exists(url: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Check if a video has already been processed.

    Args:
        url: Video URL to check
        db_path: Path to the database file

    Returns:
        True if video exists and was successfully processed, False otherwise
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT status FROM videos WHERE url = ?",
        (url,)
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        return result["status"] == "complete"
    return False


def save_video_analysis(
    url: str,
    politician_name: str,
    data: Dict[str, Any],
    date: Optional[str] = None,
    title: Optional[str] = None,
    status: str = "complete",
    error_message: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Save or update video analysis results.

    Args:
        url: Video URL (primary key)
        politician_name: Name of politician in the video
        data: Complete analysis results dictionary
        date: Video date (from CSV)
        title: Video title
        status: Processing status (pending, processing, complete, error)
        error_message: Error message if status is error
        db_path: Path to the database file
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()

    # Extract key fields from data
    transcription = data.get("transcription", "")
    if isinstance(transcription, dict):
        # Handle translated transcription format
        transcription = transcription.get("translated_text") or transcription.get("text", "")

    language = data.get("detected_language", "") or data.get("language", "")

    # Serialize full analysis to JSON
    analysis_json = json.dumps(data)

    # Check if record exists
    cursor.execute("SELECT url FROM videos WHERE url = ?", (url,))
    exists = cursor.fetchone() is not None

    if exists:
        # Update existing record
        cursor.execute("""
            UPDATE videos
            SET politician_name = ?,
                title = ?,
                transcription = ?,
                language = ?,
                analysis_json = ?,
                updated_at = ?,
                status = ?,
                error_message = ?,
                date = COALESCE(?, date)
            WHERE url = ?
        """, (
            politician_name,
            title,
            transcription,
            language,
            analysis_json,
            now,
            status,
            error_message,
            date,
            url
        ))
        logger.info(f"Updated video analysis for {url}")
    else:
        # Insert new record
        cursor.execute("""
            INSERT INTO videos (
                url, date, politician_name, title, transcription,
                language, analysis_json, created_at, updated_at,
                status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            url,
            date,
            politician_name,
            title,
            transcription,
            language,
            analysis_json,
            now,
            now,
            status,
            error_message
        ))
        logger.info(f"Saved new video analysis for {url}")

    conn.commit()
    conn.close()


def get_all_results(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all video analysis results.

    Args:
        db_path: Path to the database file

    Returns:
        List of video analysis dictionaries
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT url, date, politician_name, title, transcription,
               language, analysis_json, created_at, updated_at,
               status, error_message
        FROM videos
        ORDER BY date DESC
    """)

    results = []
    for row in cursor.fetchall():
        result = dict(row)
        # Parse JSON analysis data
        if result["analysis_json"]:
            result["analysis"] = json.loads(result["analysis_json"])
        results.append(result)

    conn.close()
    return results


def get_videos_by_politician(
    politician_name: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Get all videos for a specific politician.

    Args:
        politician_name: Name of the politician
        db_path: Path to the database file

    Returns:
        List of video analysis dictionaries
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT url, date, politician_name, title, transcription,
               language, analysis_json, created_at, updated_at,
               status, error_message
        FROM videos
        WHERE politician_name = ?
        ORDER BY date DESC
    """, (politician_name,))

    results = []
    for row in cursor.fetchall():
        result = dict(row)
        if result["analysis_json"]:
            result["analysis"] = json.loads(result["analysis_json"])
        results.append(result)

    conn.close()
    return results


def get_politician_summary(
    politician_name: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Get summary statistics for a politician.

    Args:
        politician_name: Name of the politician
        db_path: Path to the database file

    Returns:
        Summary dictionary with video count, date range, etc.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_videos,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
        FROM videos
        WHERE politician_name = ?
    """, (politician_name,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return {}


def get_date_range_results(
    start_date: str,
    end_date: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Get videos within a date range.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        db_path: Path to the database file

    Returns:
        List of video analysis dictionaries
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT url, date, politician_name, title, transcription,
               language, analysis_json, created_at, updated_at,
               status, error_message
        FROM videos
        WHERE date BETWEEN ? AND ?
        ORDER BY date DESC
    """, (start_date, end_date))

    results = []
    for row in cursor.fetchall():
        result = dict(row)
        if result["analysis_json"]:
            result["analysis"] = json.loads(result["analysis_json"])
        results.append(result)

    conn.close()
    return results


def get_sentiment_statistics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Get overall sentiment statistics across all videos.

    Args:
        db_path: Path to the database file

    Returns:
        Statistics dictionary with sentiment distribution
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT politician_name, analysis_json, status
        FROM videos
        WHERE status = 'complete'
    """)

    stats = {
        "total_videos": 0,
        "by_politician": {},
        "overall_sentiment": {"positive": 0, "neutral": 0, "negative": 0}
    }

    for row in cursor.fetchall():
        stats["total_videos"] += 1
        politician = row["politician_name"]

        if politician not in stats["by_politician"]:
            stats["by_politician"][politician] = {
                "count": 0,
                "sentiment": {"positive": 0, "neutral": 0, "negative": 0}
            }

        stats["by_politician"][politician]["count"] += 1

        # Parse analysis JSON and extract sentiment
        if row["analysis_json"]:
            try:
                analysis = json.loads(row["analysis_json"])
                overall_sentiment = analysis.get("overall_sentiment", "").lower()

                if overall_sentiment in ["positive", "neutral", "negative"]:
                    stats["overall_sentiment"][overall_sentiment] += 1
                    stats["by_politician"][politician]["sentiment"][overall_sentiment] += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing analysis JSON: {e}")

    conn.close()
    return stats


def generate_text_report(db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Generate a text summary report of all analyses.

    Args:
        db_path: Path to the database file

    Returns:
        Markdown-formatted report string
    """
    stats = get_sentiment_statistics(db_path)

    report = "# Batch Video Analysis Report\n\n"
    report += f"**Total Videos Analyzed**: {stats['total_videos']}\n\n"

    report += "## Overall Sentiment Distribution\n\n"
    total = stats['total_videos']
    if total > 0:
        for sentiment, count in stats['overall_sentiment'].items():
            percentage = (count / total) * 100
            report += f"- **{sentiment.capitalize()}**: {count} ({percentage:.1f}%)\n"

    report += "\n## By Politician\n\n"
    for politician, data in sorted(stats['by_politician'].items()):
        report += f"### {politician}\n\n"
        report += f"**Videos**: {data['count']}\n\n"
        report += "**Sentiment**:\n"
        for sentiment, count in data['sentiment'].items():
            if data['count'] > 0:
                percentage = (count / data['count']) * 100
                report += f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)\n"
        report += "\n"

    return report
