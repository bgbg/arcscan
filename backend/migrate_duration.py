#!/usr/bin/env python3
"""
Migration script to add duration column to video_metadata table.
Optionally clears all existing data for reprocessing.

Usage:
    python migrate_duration.py                    # Add column only
    python migrate_duration.py --clear-data       # Add column and clear all data
    python migrate_duration.py --db-path <path>   # Specify database path
"""

import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime


def add_duration_column(db_path: str, dry_run: bool = False) -> bool:
    """Add duration column to video_metadata table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(video_metadata)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'duration' in columns:
            print("✓ Duration column already exists in video_metadata table")
            return True

        if dry_run:
            print("[DRY RUN] Would add duration INTEGER column to video_metadata")
            return True

        # Add duration column
        cursor.execute("ALTER TABLE video_metadata ADD COLUMN duration INTEGER")
        conn.commit()
        print("✓ Added duration INTEGER column to video_metadata table")
        return True

    except sqlite3.Error as e:
        print(f"✗ Error adding duration column: {e}", file=sys.stderr)
        return False
    finally:
        conn.close()


def clear_all_data(db_path: str, dry_run: bool = False) -> bool:
    """Clear all data from video-related tables (cascades to child tables)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get counts before deletion
        cursor.execute("SELECT COUNT(*) FROM video_metadata")
        video_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transcriptions")
        transcription_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sentences")
        sentence_count = cursor.fetchone()[0]

        print(f"\nCurrent database contents:")
        print(f"  - Videos: {video_count}")
        print(f"  - Transcriptions: {transcription_count}")
        print(f"  - Sentences: {sentence_count}")

        if dry_run:
            print("\n[DRY RUN] Would delete all records from video_metadata (cascades to child tables)")
            return True

        # Confirm deletion
        print("\n⚠️  WARNING: This will delete ALL video data from the database!")
        response = input("Type 'DELETE ALL' to confirm: ")

        if response != "DELETE ALL":
            print("✗ Deletion cancelled")
            return False

        # Delete all videos (cascades to child tables due to ON DELETE CASCADE)
        cursor.execute("DELETE FROM video_metadata")
        deleted_count = cursor.rowcount
        conn.commit()

        print(f"\n✓ Deleted {deleted_count} videos from database (child records cascaded)")

        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM video_metadata")
        remaining = cursor.fetchone()[0]

        if remaining > 0:
            print(f"⚠️  Warning: {remaining} videos still remain in database", file=sys.stderr)
            return False

        print("✓ Database cleared successfully")
        return True

    except sqlite3.Error as e:
        print(f"✗ Error clearing data: {e}", file=sys.stderr)
        conn.rollback()
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Add duration column to video_metadata and optionally clear data"
    )
    parser.add_argument(
        "--db-path",
        default="/Users/boris/devel/jce/arcscan/backend/batch_results.db",
        help="Path to SQLite database (default: /Users/boris/devel/jce/arcscan/backend/batch_results.db)"
    )
    parser.add_argument(
        "--clear-data",
        action="store_true",
        help="Clear all existing video data after adding column"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)

    if not db_path.exists():
        print(f"✗ Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {db_path}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Step 1: Add duration column
    print("Step 1: Adding duration column...")
    if not add_duration_column(str(db_path), args.dry_run):
        sys.exit(1)

    # Step 2: Clear data if requested
    if args.clear_data:
        print("\nStep 2: Clearing all video data...")
        if not clear_all_data(str(db_path), args.dry_run):
            sys.exit(1)

    print("\n✓ Migration completed successfully")

    if args.clear_data and not args.dry_run:
        print("\nNext steps:")
        print("  1. Run reprocessing script to populate database with new data")
        print("  2. Verify all videos have duration and title populated")


if __name__ == "__main__":
    main()
