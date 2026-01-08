#!/usr/bin/env python3
"""
Database schema migration script: V1 (JSON blob) → V2 (normalized tables)

This script migrates the existing SQLite database from the monolithic
analysis_json schema to the new normalized schema with separate tables
for transcriptions, sentences, sentiments, and emotions.

Usage:
    python migrate_schema.py [--db-path path/to/database.db] [--dry-run] [--rollback]

Options:
    --db-path: Path to database file (default: backend/batch_results.db)
    --dry-run: Validate migration without making changes
    --rollback: Restore from backup table
"""

import sqlite3
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(Path(__file__).parent / "batch_results.db")
SCHEMA_FILE = str(Path(__file__).parent / "schema_v2.sql")


class MigrationReport:
    """Container for migration statistics and validation results."""

    def __init__(self):
        self.total_videos = 0
        self.migrated = 0
        self.failed = 0
        self.errors = []

        # Per-table counts
        self.transcriptions_created = 0
        self.translations_created = 0
        self.sentences_created = 0
        self.sentiments_created = 0
        self.emotions_created = 0

        self.start_time = None
        self.end_time = None

    def start(self):
        """Mark migration start time."""
        self.start_time = datetime.utcnow()

    def end(self):
        """Mark migration end time."""
        self.end_time = datetime.utcnow()

    def duration_seconds(self) -> float:
        """Calculate migration duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_videos": self.total_videos,
            "migrated": self.migrated,
            "failed": self.failed,
            "errors": self.errors,
            "tables": {
                "transcriptions": self.transcriptions_created,
                "translations": self.translations_created,
                "sentences": self.sentences_created,
                "sentiments": self.sentiments_created,
                "emotions": self.emotions_created,
            },
            "duration_seconds": self.duration_seconds()
        }

    def __str__(self) -> str:
        """Generate human-readable report."""
        report = "\n" + "="*70 + "\n"
        report += "MIGRATION REPORT\n"
        report += "="*70 + "\n"
        report += f"Total videos:        {self.total_videos}\n"
        report += f"Successfully migrated: {self.migrated}\n"
        report += f"Failed:              {self.failed}\n"
        report += f"\nTable row counts:\n"
        report += f"  Transcriptions:    {self.transcriptions_created}\n"
        report += f"  Translations:      {self.translations_created}\n"
        report += f"  Sentences:         {self.sentences_created}\n"
        report += f"  Sentiments:        {self.sentiments_created}\n"
        report += f"  Emotions:          {self.emotions_created}\n"
        report += f"\nDuration:            {self.duration_seconds():.2f} seconds\n"

        if self.errors:
            report += f"\nERRORS ({len(self.errors)}):\n"
            for error in self.errors[:10]:  # Show first 10 errors
                report += f"  - {error['url']}: {error['error']}\n"
            if len(self.errors) > 10:
                report += f"  ... and {len(self.errors) - 10} more errors\n"

        report += "="*70 + "\n"
        return report


def create_backup(conn: sqlite3.Connection) -> None:
    """
    Create backup of videos table before migration.

    Args:
        conn: Database connection
    """
    logger.info("Creating backup table...")
    cursor = conn.cursor()

    # Drop backup if exists (from previous migration attempt)
    cursor.execute("DROP TABLE IF EXISTS videos_backup")

    # Create backup
    cursor.execute("CREATE TABLE videos_backup AS SELECT * FROM videos")

    backup_count = cursor.execute("SELECT COUNT(*) FROM videos_backup").fetchone()[0]
    logger.info(f"Backed up {backup_count} videos to 'videos_backup' table")

    conn.commit()


def load_new_schema(conn: sqlite3.Connection) -> None:
    """
    Load and execute new schema SQL.

    Args:
        conn: Database connection
    """
    logger.info(f"Loading new schema from {SCHEMA_FILE}...")

    with open(SCHEMA_FILE, 'r') as f:
        schema_sql = f.read()

    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    conn.commit()

    logger.info("New schema created successfully")


def migrate_video(
    video: sqlite3.Row,
    conn: sqlite3.Connection,
    report: MigrationReport
) -> bool:
    """
    Migrate a single video from old to new schema.

    Args:
        video: Row from old videos table
        conn: Database connection
        report: Migration report to update

    Returns:
        True if migration successful, False otherwise
    """
    cursor = conn.cursor()
    url = video['url']

    try:
        # Parse analysis JSON
        analysis = {}
        if video['analysis_json']:
            try:
                analysis = json.loads(video['analysis_json'])
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON for {url}: {e}")
                analysis = {}

        now = datetime.utcnow().isoformat()

        # 1. Insert into video_metadata (keep analysis_json for backward compatibility)
        cursor.execute("""
            INSERT INTO video_metadata (
                url, date, person_name, title, created_at, updated_at,
                status, error_message, analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            url,
            video['date'],
            video['person_name'],
            video['title'],
            video['created_at'],
            video['updated_at'],
            video['status'],
            video['error_message'],
            video['analysis_json']  # Keep for transition period
        ))

        # Skip further processing if video errored or has no analysis
        if video['status'] != 'complete' or not analysis:
            report.migrated += 1
            return True

        # 2. Insert transcription
        transcription_text = analysis.get("transcription") or analysis.get("text", "")
        detected_lang = analysis.get("detected_language", "")
        output_lang = analysis.get("output_language", "")
        source = analysis.get("source", "whisper")  # Default to whisper
        subtitle_lang = analysis.get("subtitle_language")
        decision_path = analysis.get("decision_path", "")

        cursor.execute("""
            INSERT INTO transcriptions (
                video_url, text, language, source, detected_language,
                subtitle_language, decision_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            url,
            transcription_text,
            output_lang,
            source,
            detected_lang,
            subtitle_lang,
            decision_path,
            now
        ))

        transcription_id = cursor.lastrowid
        report.transcriptions_created += 1

        # 3. Insert translation if exists
        if "translated_text" in analysis and "original_text" in analysis:
            cursor.execute("""
                INSERT INTO translations (
                    transcription_id, original_text, translated_text, model, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                transcription_id,
                analysis.get("original_text", ""),
                analysis.get("translated_text", ""),
                analysis.get("translation_model", "gpt-3.5-turbo"),
                now
            ))
            report.translations_created += 1

        # 4. Insert sentences with sentiment
        sentences = analysis.get("sentences", [])
        if sentences:
            for idx, sentence in enumerate(sentences):
                cursor.execute("""
                    INSERT INTO sentences (
                        video_url, sentence_index, text, start_time, end_time,
                        sentiment, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    url,
                    idx,
                    sentence.get("text", ""),
                    sentence.get("start", 0.0),
                    sentence.get("end", 0.0),
                    sentence.get("sentiment", "neutral"),
                    now
                ))

                sentence_id = cursor.lastrowid
                report.sentences_created += 1

                # 5. Insert sentence emotions if available
                emotions = sentence.get("emotions", {})
                if emotions:
                    for emotion_name, score in emotions.items():
                        cursor.execute("""
                            INSERT INTO sentence_emotions (
                                sentence_id, emotion_name, score, created_at
                            ) VALUES (?, ?, ?, ?)
                        """, (sentence_id, emotion_name, score, now))
                        report.emotions_created += 1

        # 6. Insert video sentiment summary
        summary = analysis.get("summary", {})
        overall_sentiment = analysis.get("overall_sentiment", "neutral")

        positive_count = summary.get("positive", 0)
        neutral_count = summary.get("neutral", 0)
        negative_count = summary.get("negative", 0)
        total_count = positive_count + neutral_count + negative_count

        if total_count > 0:
            pos_pct = (positive_count / total_count) * 100
            neu_pct = (neutral_count / total_count) * 100
            neg_pct = (negative_count / total_count) * 100
        else:
            pos_pct = neu_pct = neg_pct = 0.0

        cursor.execute("""
            INSERT INTO video_sentiments (
                video_url, overall_sentiment,
                positive_count, positive_pct,
                neutral_count, neutral_pct,
                negative_count, negative_pct,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            url, overall_sentiment,
            positive_count, pos_pct,
            neutral_count, neu_pct,
            negative_count, neg_pct,
            now
        ))
        report.sentiments_created += 1

        # 7. Insert video emotion summary if available
        # Note: This would be populated if advanced emotion analysis was run
        # For now, we'll create a placeholder entry with default values
        # Future enhancement: Parse emotion data from analysis if available
        cursor.execute("""
            INSERT INTO video_emotion_summary (
                video_url, created_at
            ) VALUES (?, ?)
        """, (url, now))

        report.migrated += 1
        return True

    except Exception as e:
        logger.error(f"Failed to migrate {url}: {e}")
        report.failed += 1
        report.errors.append({
            "url": url,
            "error": str(e)
        })
        return False


def run_migration(
    db_path: str = DEFAULT_DB_PATH,
    dry_run: bool = False
) -> MigrationReport:
    """
    Execute the migration from old to new schema.

    Args:
        db_path: Path to database file
        dry_run: If True, validate without making changes

    Returns:
        MigrationReport with statistics
    """
    logger.info(f"Starting migration for database: {db_path}")
    logger.info(f"Dry run mode: {dry_run}")

    report = MigrationReport()
    report.start()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Check if old videos table exists
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='videos'
        """)
        if not cursor.fetchone():
            logger.error("Old 'videos' table not found. Nothing to migrate.")
            return report

        # Get video count
        cursor.execute("SELECT COUNT(*) FROM videos")
        report.total_videos = cursor.fetchone()[0]
        logger.info(f"Found {report.total_videos} videos to migrate")

        if dry_run:
            logger.info("Dry run - validating data only, no changes will be made")

            # Validate JSON parsing
            cursor.execute("SELECT url, analysis_json FROM videos")
            invalid_json_count = 0
            for row in cursor.fetchall():
                if row['analysis_json']:
                    try:
                        json.loads(row['analysis_json'])
                    except json.JSONDecodeError:
                        invalid_json_count += 1
                        logger.warning(f"Invalid JSON in {row['url']}")

            logger.info(f"Validation complete: {invalid_json_count} videos have invalid JSON")
            report.end()
            return report

        # Real migration
        logger.info("Creating backup...")
        create_backup(conn)

        logger.info("Creating new schema...")
        load_new_schema(conn)

        logger.info("Migrating video data...")
        cursor.execute("""
            SELECT * FROM videos
            ORDER BY created_at ASC
        """)

        videos = cursor.fetchall()
        for idx, video in enumerate(videos, 1):
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{report.total_videos}")

            migrate_video(video, conn, report)

        # Commit all changes
        conn.commit()
        logger.info("Migration committed successfully")

        # Validate row counts
        logger.info("Validating migration...")
        cursor.execute("SELECT COUNT(*) FROM video_metadata")
        metadata_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transcriptions")
        transcription_count = cursor.fetchone()[0]

        logger.info(f"Validation: {metadata_count} videos in new schema")
        logger.info(f"Validation: {transcription_count} transcriptions created")

        if metadata_count != report.total_videos:
            logger.warning(
                f"Row count mismatch! Expected {report.total_videos}, "
                f"got {metadata_count}"
            )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if not dry_run:
            logger.info("Rolling back transaction...")
            conn.rollback()
        raise
    finally:
        conn.close()
        report.end()

    return report


def rollback_migration(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Restore database from backup table.

    Args:
        db_path: Path to database file
    """
    logger.info(f"Rolling back migration for: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if backup exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='videos_backup'
        """)
        if not cursor.fetchone():
            logger.error("No backup table found. Cannot rollback.")
            return

        # Drop views first (they may depend on tables we're about to drop)
        logger.info("Dropping views...")
        cursor.execute("DROP VIEW IF EXISTS videos_legacy")

        # Drop new tables
        logger.info("Dropping new schema tables...")
        new_tables = [
            'sentence_emotions',
            'video_emotion_summary',
            'video_sentiments',
            'sentences',
            'translations',
            'transcriptions',
            'video_metadata'
        ]
        for table in new_tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        # Drop legacy videos table if it exists (in case migration was partially complete)
        cursor.execute("DROP TABLE IF EXISTS videos")

        # Restore videos table from backup
        logger.info("Restoring videos table from backup...")
        cursor.execute("ALTER TABLE videos_backup RENAME TO videos")

        # Recreate indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_name ON videos(person_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_date ON videos(date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON videos(status)
        """)

        conn.commit()
        logger.info("Rollback completed successfully")

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate video analysis database from V1 to V2 schema"
    )
    parser.add_argument(
        '--db-path',
        default=DEFAULT_DB_PATH,
        help=f'Path to database file (default: {DEFAULT_DB_PATH})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate migration without making changes'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Restore from backup table'
    )

    args = parser.parse_args()

    if args.rollback:
        rollback_migration(args.db_path)
        return

    report = run_migration(args.db_path, dry_run=args.dry_run)
    print(report)

    if report.failed > 0:
        logger.warning(f"{report.failed} videos failed to migrate")
        exit(1)
    else:
        logger.info("Migration completed successfully!")
        exit(0)


if __name__ == "__main__":
    main()
