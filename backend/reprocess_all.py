#!/usr/bin/env python3
"""
Reprocessing script to populate video metadata (title and duration) for all videos.

This script reads videos from CSV files and reprocesses them through the batch
processing pipeline to ensure all videos have complete metadata including title
and duration fields.

Usage:
    python reprocess_all.py --csv path/to/videos.csv           # Process single CSV
    python reprocess_all.py --csv-dir path/to/csv/dir          # Process all CSVs in directory
    python reprocess_all.py --resume state.json                # Resume from saved state
    python reprocess_all.py --dry-run --csv videos.csv         # Show what would be processed
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'reprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Import batch processor
try:
    from batch_processor import process_single_video, init_database, DEFAULT_DB_PATH
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from batch_processor import process_single_video, init_database, DEFAULT_DB_PATH


class ReprocessState:
    """Manages reprocessing state for resume capability."""

    def __init__(self, state_file: str = "reprocess_state.json"):
        self.state_file = state_file
        self.processed_urls = set()
        self.last_processed_index = -1
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        self.errors = []

    def load(self) -> bool:
        """Load state from file."""
        if not os.path.exists(self.state_file):
            return False

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.processed_urls = set(data.get('processed_urls', []))
                self.last_processed_index = data.get('last_processed_index', -1)
                self.total_processed = data.get('total_processed', 0)
                self.total_successful = data.get('total_successful', 0)
                self.total_failed = data.get('total_failed', 0)
                self.errors = data.get('errors', [])
                logger.info(f"Loaded state from {self.state_file}: {len(self.processed_urls)} videos already processed")
                return True
        except Exception as e:
            logger.error(f"Failed to load state file: {e}")
            return False

    def save(self):
        """Save state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'processed_urls': list(self.processed_urls),
                    'last_processed_index': self.last_processed_index,
                    'total_processed': self.total_processed,
                    'total_successful': self.total_successful,
                    'total_failed': self.total_failed,
                    'errors': self.errors,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def mark_processed(self, url: str, index: int, success: bool, error_msg: Optional[str] = None):
        """Mark a video as processed."""
        self.processed_urls.add(url)
        self.last_processed_index = index
        self.total_processed += 1

        if success:
            self.total_successful += 1
        else:
            self.total_failed += 1
            if error_msg:
                self.errors.append({
                    'url': url,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })

    def is_processed(self, url: str) -> bool:
        """Check if URL was already processed."""
        return url in self.processed_urls


def read_csv_file(csv_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Read videos from CSV file.

    Expected CSV format:
        date,name,person,url

    Returns:
        List of tuples (date, name, person, url)
    """
    videos = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate headers
            required_fields = {'date', 'person', 'url'}
            if not required_fields.issubset(reader.fieldnames or []):
                logger.error(f"CSV file missing required fields. Expected: {required_fields}, Found: {reader.fieldnames}")
                return []

            for row in reader:
                date = row['date'].strip()
                name = row.get('name', '').strip()
                person = row['person'].strip()
                url = row['url'].strip()

                if not url or not person:
                    logger.warning(f"Skipping row with missing data: {row}")
                    continue

                videos.append((date, name, person, url))

        logger.info(f"Read {len(videos)} videos from {csv_path}")
        return videos

    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return []


def find_csv_files(directory: str) -> List[str]:
    """Find all CSV files in a directory."""
    csv_files = []
    for file in Path(directory).glob('*.csv'):
        if file.is_file():
            csv_files.append(str(file))
    return sorted(csv_files)


def process_videos_with_retry(
    videos: List[Tuple[str, str, str, str]],
    state: ReprocessState,
    db_path: str,
    dry_run: bool = False,
    max_retries: int = 3,
    base_delay: float = 2.0,
    batch_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Process videos with exponential backoff retry logic.

    Args:
        videos: List of (date, name, person, url) tuples
        state: ReprocessState object for tracking progress
        db_path: Path to database
        dry_run: If True, don't actually process videos
        max_retries: Maximum number of retries per video
        base_delay: Base delay in seconds between videos
        batch_delay: Delay in seconds between batches of 10 videos

    Returns:
        Dict with processing statistics
    """
    total = len(videos)
    processed = 0
    successful = 0
    failed = 0
    skipped = 0

    logger.info(f"Starting processing of {total} videos (dry_run={dry_run})")

    for index, (date, name, person, url) in enumerate(videos):
        # Check if already processed
        if state.is_processed(url):
            logger.info(f"[{index + 1}/{total}] Skipping already processed: {url}")
            skipped += 1
            continue

        if dry_run:
            logger.info(f"[{index + 1}/{total}] [DRY RUN] Would process: {person} - {url}")
            processed += 1
            continue

        logger.info(f"[{index + 1}/{total}] Processing: {person} - {name or 'No title'}")
        logger.info(f"  URL: {url}")

        # Retry logic with exponential backoff
        retry_count = 0
        success = False
        error_msg = None

        while retry_count < max_retries and not success:
            if retry_count > 0:
                delay = base_delay * (2 ** (retry_count - 1))
                logger.info(f"  Retry {retry_count}/{max_retries} after {delay}s delay...")
                time.sleep(delay)

            try:
                success, error_msg = process_single_video(
                    url=url,
                    person_name=person,
                    date=date,
                    db_path=db_path
                )

                if success:
                    logger.info(f"  ✓ Successfully processed")
                    successful += 1
                else:
                    logger.error(f"  ✗ Processing failed: {error_msg}")
                    retry_count += 1

            except Exception as e:
                error_msg = str(e)
                logger.error(f"  ✗ Exception during processing: {error_msg}")
                retry_count += 1

        # Mark as processed regardless of success
        state.mark_processed(url, index, success, error_msg)
        processed += 1

        if not success:
            failed += 1
            logger.error(f"  ✗ Failed after {retry_count} retries")

        # Save state after each video
        state.save()

        # Add delay between videos to avoid rate limits
        if processed < total:
            time.sleep(base_delay)

        # Longer delay every 10 videos
        if processed % 10 == 0:
            logger.info(f"Progress: {processed}/{total} processed ({successful} successful, {failed} failed)")
            time.sleep(batch_delay)

    return {
        'total': total,
        'processed': processed,
        'successful': successful,
        'failed': failed,
        'skipped': skipped
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess videos to populate metadata (title and duration)"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--csv',
        help='Path to CSV file with video list'
    )
    input_group.add_argument(
        '--csv-dir',
        help='Path to directory containing CSV files'
    )

    # Processing options
    parser.add_argument(
        '--db-path',
        default=DEFAULT_DB_PATH,
        help=f'Path to SQLite database (default: {DEFAULT_DB_PATH})'
    )
    parser.add_argument(
        '--state-file',
        default='reprocess_state.json',
        help='Path to state file for resume capability (default: reprocess_state.json)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last saved state'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retries per video (default: 3)'
    )
    parser.add_argument(
        '--base-delay',
        type=float,
        default=2.0,
        help='Base delay in seconds between videos (default: 2.0)'
    )
    parser.add_argument(
        '--batch-delay',
        type=float,
        default=5.0,
        help='Extra delay in seconds every 10 videos (default: 5.0)'
    )

    args = parser.parse_args()

    # Initialize state
    state = ReprocessState(args.state_file)

    if args.resume:
        if not state.load():
            logger.warning("No state file found, starting fresh")
    else:
        logger.info("Starting fresh (use --resume to continue from previous run)")

    # Initialize database
    if not args.dry_run:
        logger.info(f"Initializing database: {args.db_path}")
        init_database(args.db_path)

    # Collect videos from CSV file(s)
    all_videos = []

    if args.csv:
        all_videos = read_csv_file(args.csv)
    elif args.csv_dir:
        csv_files = find_csv_files(args.csv_dir)
        logger.info(f"Found {len(csv_files)} CSV files in {args.csv_dir}")

        for csv_file in csv_files:
            videos = read_csv_file(csv_file)
            all_videos.extend(videos)
            logger.info(f"  {csv_file}: {len(videos)} videos")

    if not all_videos:
        logger.error("No videos found to process")
        sys.exit(1)

    # Remove duplicates (keep first occurrence)
    seen_urls = set()
    unique_videos = []
    for video in all_videos:
        url = video[3]
        if url not in seen_urls:
            seen_urls.add(url)
            unique_videos.append(video)

    if len(all_videos) != len(unique_videos):
        logger.info(f"Removed {len(all_videos) - len(unique_videos)} duplicate videos")

    all_videos = unique_videos

    # Process videos
    start_time = datetime.now()
    logger.info(f"Starting reprocessing at {start_time}")

    results = process_videos_with_retry(
        videos=all_videos,
        state=state,
        db_path=args.db_path,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        batch_delay=args.batch_delay
    )

    end_time = datetime.now()
    duration = end_time - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("REPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total videos:     {results['total']}")
    print(f"Processed:        {results['processed']}")
    print(f"Successful:       {results['successful']}")
    print(f"Failed:           {results['failed']}")
    print(f"Skipped:          {results['skipped']}")
    print(f"Duration:         {duration}")
    print("=" * 80)

    if state.errors:
        print(f"\nErrors ({len(state.errors)}):")
        for error in state.errors[-10:]:  # Show last 10 errors
            print(f"  - {error['url']}: {error['error']}")
        if len(state.errors) > 10:
            print(f"  ... and {len(state.errors) - 10} more errors")

    # Save final state
    state.save()
    logger.info(f"Final state saved to {args.state_file}")

    if results['failed'] > 0:
        logger.warning(f"Completed with {results['failed']} failures")
        sys.exit(1)
    else:
        logger.info("Reprocessing completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
