#!/usr/bin/env python3
"""
Batch Video Analysis CLI

Standalone command-line tool for batch processing YouTube videos.
Analyzes multiple videos from a CSV file and stores results in SQLite.

Usage:
    python batch_analyze.py --input videos.csv
    python batch_analyze.py --input videos.csv --export-json results.json
    python batch_analyze.py --input videos.csv --sync-firebase --user-id USER_ID
    python batch_analyze.py --report
"""

import argparse
import sys
import os
import logging
import json
from pathlib import Path
from typing import Optional

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from csv_parser import parse_csv, validate_csv_format
from batch_processor import process_video_batch, sync_to_firebase
from batch_db import (
    DEFAULT_DB_PATH,
    get_all_results,
    get_videos_by_politician,
    generate_text_report,
    init_database
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_analyze.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "batch_config.json") -> dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading config file: {e}")
        return {}


def export_to_json(output_path: str, politician: Optional[str] = None, db_path: str = DEFAULT_DB_PATH):
    """
    Export results to JSON format.

    Args:
        output_path: Path to output JSON file
        politician: Optional filter by politician name
        db_path: Path to SQLite database
    """
    logger.info(f"Exporting results to JSON: {output_path}")

    if politician:
        results = get_videos_by_politician(politician, db_path)
    else:
        results = get_all_results(db_path)

    # Group by politician
    grouped = {}
    for result in results:
        pol = result.get("politician_name", "Unknown")
        if pol not in grouped:
            grouped[pol] = []

        grouped[pol].append({
            "url": result["url"],
            "date": result["date"],
            "title": result.get("title"),
            "language": result.get("language"),
            "status": result["status"],
            "overall_sentiment": result.get("analysis", {}).get("overall_sentiment") if result.get("analysis") else None,
            "created_at": result["created_at"]
        })

    output = {
        "total_videos": len(results),
        "politicians": grouped,
        "generated_at": results[0]["created_at"] if results else None
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(results)} results to {output_path}")


def export_to_csv(output_path: str, politician: Optional[str] = None, db_path: str = DEFAULT_DB_PATH):
    """
    Export results to enhanced CSV format.

    Args:
        output_path: Path to output CSV file
        politician: Optional filter by politician name
        db_path: Path to SQLite database
    """
    import csv

    logger.info(f"Exporting results to CSV: {output_path}")

    if politician:
        results = get_videos_by_politician(politician, db_path)
    else:
        results = get_all_results(db_path)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'date', 'politician', 'url', 'language',
            'overall_sentiment', 'status', 'created_at'
        ])

        # Write data
        for result in results:
            analysis = result.get("analysis", {})
            writer.writerow([
                result.get("date", ""),
                result.get("politician_name", "Unknown"),
                result["url"],
                result.get("language", ""),
                analysis.get("overall_sentiment", "") if analysis else "",
                result["status"],
                result["created_at"]
            ])

    logger.info(f"Exported {len(results)} results to {output_path}")


def progress_callback(current: int, total: int, message: str):
    """Progress callback for batch processing."""
    percent = (current / total) * 100 if total > 0 else 0
    print(f"\r[{current}/{total}] ({percent:.1f}%) {message}", end='', flush=True)
    if current == total:
        print()  # New line at end


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Batch analyze YouTube videos for sentiment and emotions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file (date,name,url format)'
    )

    parser.add_argument(
        '--export-json',
        type=str,
        metavar='PATH',
        help='Export results to JSON file'
    )

    parser.add_argument(
        '--export-csv',
        type=str,
        metavar='PATH',
        help='Export results to CSV file'
    )

    parser.add_argument(
        '--sync-firebase',
        action='store_true',
        help='Sync results to Firebase after processing'
    )

    parser.add_argument(
        '--user-id',
        type=str,
        help='Firebase user ID (required with --sync-firebase)'
    )

    parser.add_argument(
        '--politician',
        type=str,
        help='Filter by politician name (for processing or export)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview videos without processing'
    )

    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate and print analysis report'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='batch_config.json',
        help='Path to configuration file (default: batch_config.json)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=DEFAULT_DB_PATH,
        help=f'Path to SQLite database (default: {DEFAULT_DB_PATH})'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip videos that are already in the database (default: True)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config file
    config = load_config(args.config)

    # Merge CLI args with config (CLI takes precedence)
    input_file = args.input or config.get('input_file')
    user_id = args.user_id or config.get('user_id')
    db_path = args.db_path or config.get('db_path', DEFAULT_DB_PATH)

    # Handle report mode
    if args.report:
        logger.info("Generating analysis report...")
        report = generate_text_report(db_path)
        print("\n" + report)
        return 0

    # Handle export-only mode
    if args.export_json or args.export_csv:
        if args.export_json:
            export_to_json(args.export_json, args.politician, db_path)
        if args.export_csv:
            export_to_csv(args.export_csv, args.politician, db_path)
        return 0

    # Processing mode requires input file
    if not input_file:
        parser.error("--input is required for processing mode (or use --report)")

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 1

    # Validate CSV format
    logger.info(f"Validating CSV file: {input_file}")
    is_valid, error_msg = validate_csv_format(input_file, is_file_path=True)
    if not is_valid:
        logger.error(f"Invalid CSV format: {error_msg}")
        return 1

    # Parse CSV
    logger.info("Parsing CSV...")
    try:
        videos = parse_csv(input_file, is_file_path=True)
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        return 1

    logger.info(f"Found {len(videos)} videos to process")

    # Filter by politician if specified
    if args.politician:
        videos = [(d, p, u) for d, p, u in videos if p == args.politician]
        logger.info(f"Filtered to {len(videos)} videos for politician: {args.politician}")

    if not videos:
        logger.warning("No videos to process after filtering")
        return 0

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN - No processing will occur")
        print("\nVideos to be processed:")
        for idx, (date, politician, url) in enumerate(videos, 1):
            print(f"  {idx}. [{date}] {politician} - {url[:60]}...")
        return 0

    # Initialize database
    init_database(db_path)

    # Process batch
    logger.info("Starting batch processing...")
    print()  # Blank line before progress

    result = process_video_batch(
        videos=videos,
        progress_callback=progress_callback,
        db_path=db_path,
        skip_existing=args.skip_existing
    )

    # Print results
    print("\n" + "="*60)
    print(result)
    print("="*60)

    # Export if requested
    if args.export_json:
        export_to_json(args.export_json, args.politician, db_path)

    if args.export_csv:
        export_to_csv(args.export_csv, args.politician, db_path)

    # Sync to Firebase if requested
    if args.sync_firebase:
        if not user_id:
            logger.error("--user-id is required when using --sync-firebase")
            return 1

        logger.info("Syncing results to Firebase...")
        sync_stats = sync_to_firebase(user_id, db_path=db_path)
        print(f"\nFirebase Sync: {sync_stats['synced']}/{sync_stats['total']} videos synced")

        if sync_stats['failed'] > 0:
            print(f"  {sync_stats['failed']} videos failed to sync")
            for error in sync_stats['errors']:
                print(f"    - {error['url']}: {error['error']}")

    # Return error code if any videos failed
    if result.failed > 0:
        logger.warning(f"{result.failed} videos failed to process")
        return 1

    logger.info("Batch processing complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
