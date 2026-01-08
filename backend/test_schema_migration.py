"""
Tests for schema V2 migration and new query functions.

Run with: pytest test_schema_migration.py -v
"""

import pytest
import sqlite3
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Import functions to test
from batch_db import (
    init_database,
    save_video_analysis,
    get_all_results,
    get_videos_by_person,
    get_person_summary,
    get_sentiment_statistics,
    get_video_timeline,
    get_person_sentiment_trends,
    get_emotion_breakdown,
    get_db_connection
)
from migrate_schema import run_migration, rollback_migration


# Sample test data
SAMPLE_ANALYSIS_DATA = {
    "transcription": "This is a sample transcription.",
    "text": "This is a sample transcription.",
    "sentences": [
        {
            "text": "This is a sample sentence.",
            "start": 0.0,
            "end": 2.5,
            "sentiment": "positive",
            "emotions": {
                "joy": 0.8,
                "neutral": 0.2
            }
        },
        {
            "text": "This is another sentence.",
            "start": 2.5,
            "end": 5.0,
            "sentiment": "neutral",
            "emotions": {
                "neutral": 0.9,
                "joy": 0.1
            }
        },
        {
            "text": "And a negative one.",
            "start": 5.0,
            "end": 7.0,
            "sentiment": "negative",
            "emotions": {
                "sadness": 0.6,
                "anger": 0.3,
                "neutral": 0.1
            }
        }
    ],
    "summary": {
        "positive": 1,
        "neutral": 1,
        "negative": 1
    },
    "overall_sentiment": "neutral",
    "timeline_data": [],
    "detected_language": "en",
    "output_language": "en",
    "source": "whisper",
    "decision_path": "whisper"
}

SAMPLE_TRANSLATION_DATA = {
    **SAMPLE_ANALYSIS_DATA,
    "original_text": "זה טקסט לדוגמה",
    "translated_text": "This is a sample transcription.",
    "translation_model": "gpt-3.5-turbo",
    "detected_language": "he",
    "output_language": "en"
}


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def initialized_db(temp_db):
    """Create and initialize a test database with V2 schema."""
    init_database(temp_db)
    return temp_db


@pytest.fixture
def db_with_sample_data(initialized_db):
    """Create a database with sample test data."""
    # Add sample videos
    save_video_analysis(
        url="https://youtube.com/watch?v=test1",
        person_name="Person A",
        data=SAMPLE_ANALYSIS_DATA,
        date="2024-01-01",
        title="Test Video 1",
        db_path=initialized_db
    )

    save_video_analysis(
        url="https://youtube.com/watch?v=test2",
        person_name="Person A",
        data=SAMPLE_TRANSLATION_DATA,
        date="2024-01-02",
        title="Test Video 2",
        db_path=initialized_db
    )

    save_video_analysis(
        url="https://youtube.com/watch?v=test3",
        person_name="Person B",
        data=SAMPLE_ANALYSIS_DATA,
        date="2024-01-03",
        title="Test Video 3",
        db_path=initialized_db
    )

    return initialized_db


class TestSchemaInitialization:
    """Test database schema initialization."""

    def test_init_creates_all_tables(self, temp_db):
        """Test that init_database creates all required tables."""
        init_database(temp_db)

        conn = get_db_connection(temp_db)
        cursor = conn.cursor()

        # Check for all V2 tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'sentence_emotions',
            'sentences',
            'transcriptions',
            'translations',
            'video_emotion_summary',
            'video_metadata',
            'video_sentiments'
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not created"

        conn.close()

    def test_init_creates_indexes(self, initialized_db):
        """Test that indexes are created."""
        conn = get_db_connection(initialized_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        indexes = [row[0] for row in cursor.fetchall()]

        # Check for some key indexes
        assert 'idx_video_person_name' in indexes
        assert 'idx_video_date' in indexes
        assert 'idx_transcription_video_url' in indexes
        assert 'idx_sentence_video_url' in indexes

        conn.close()


class TestWriteFunctions:
    """Test data write functions."""

    def test_save_video_analysis_basic(self, initialized_db):
        """Test saving basic video analysis."""
        save_video_analysis(
            url="https://youtube.com/watch?v=test123",
            person_name="Test Person",
            data=SAMPLE_ANALYSIS_DATA,
            date="2024-01-01",
            title="Test Video",
            db_path=initialized_db
        )

        # Verify data was saved to video_metadata
        conn = get_db_connection(initialized_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM video_metadata WHERE url = ?
        """, ("https://youtube.com/watch?v=test123",))
        row = cursor.fetchone()

        assert row is not None
        assert row["person_name"] == "Test Person"
        assert row["status"] == "complete"

        # Verify transcription was saved
        cursor.execute("""
            SELECT * FROM transcriptions WHERE video_url = ?
        """, ("https://youtube.com/watch?v=test123",))
        trans_row = cursor.fetchone()

        assert trans_row is not None
        assert trans_row["text"] == "This is a sample transcription."
        assert trans_row["source"] == "whisper"

        # Verify sentences were saved
        cursor.execute("""
            SELECT COUNT(*) FROM sentences WHERE video_url = ?
        """, ("https://youtube.com/watch?v=test123",))
        sentence_count = cursor.fetchone()[0]

        assert sentence_count == 3

        # Verify sentiments were saved
        cursor.execute("""
            SELECT * FROM video_sentiments WHERE video_url = ?
        """, ("https://youtube.com/watch?v=test123",))
        sent_row = cursor.fetchone()

        assert sent_row is not None
        assert sent_row["overall_sentiment"] == "neutral"
        assert sent_row["positive_count"] == 1
        assert sent_row["neutral_count"] == 1
        assert sent_row["negative_count"] == 1

        conn.close()

    def test_save_video_with_translation(self, initialized_db):
        """Test saving video with translation data."""
        save_video_analysis(
            url="https://youtube.com/watch?v=translated",
            person_name="Hebrew Speaker",
            data=SAMPLE_TRANSLATION_DATA,
            date="2024-01-01",
            db_path=initialized_db
        )

        conn = get_db_connection(initialized_db)
        cursor = conn.cursor()

        # Verify translation was saved
        cursor.execute("""
            SELECT t.original_text, t.translated_text, t.model
            FROM translations t
            INNER JOIN transcriptions tr ON t.transcription_id = tr.id
            WHERE tr.video_url = ?
        """, ("https://youtube.com/watch?v=translated",))
        row = cursor.fetchone()

        assert row is not None
        assert row["original_text"] == "זה טקסט לדוגמה"
        assert row["translated_text"] == "This is a sample transcription."
        assert row["model"] == "gpt-3.5-turbo"

        conn.close()

    def test_save_video_with_emotions(self, initialized_db):
        """Test saving video with emotion data."""
        save_video_analysis(
            url="https://youtube.com/watch?v=emotions",
            person_name="Test Person",
            data=SAMPLE_ANALYSIS_DATA,
            date="2024-01-01",
            db_path=initialized_db
        )

        conn = get_db_connection(initialized_db)
        cursor = conn.cursor()

        # Verify emotions were saved
        cursor.execute("""
            SELECT COUNT(*) FROM sentence_emotions se
            INNER JOIN sentences s ON se.sentence_id = s.id
            WHERE s.video_url = ?
        """, ("https://youtube.com/watch?v=emotions",))
        emotion_count = cursor.fetchone()[0]

        # We have 3 sentences with emotions
        assert emotion_count > 0

        conn.close()


class TestQueryFunctions:
    """Test data query functions."""

    def test_get_all_results(self, db_with_sample_data):
        """Test retrieving all results."""
        results = get_all_results(db_with_sample_data)

        assert len(results) == 3
        assert results[0]["person_name"] in ["Person A", "Person B"]

    def test_get_videos_by_person(self, db_with_sample_data):
        """Test filtering videos by person."""
        results = get_videos_by_person("Person A", db_with_sample_data)

        assert len(results) == 2
        for result in results:
            assert result["person_name"] == "Person A"

    def test_get_person_summary(self, db_with_sample_data):
        """Test getting person summary statistics."""
        summary = get_person_summary("Person A", db_with_sample_data)

        assert summary["total_videos"] == 2
        assert summary["completed"] == 2
        assert summary["earliest_date"] == "2024-01-01"
        assert summary["latest_date"] == "2024-01-02"

    def test_get_sentiment_statistics(self, db_with_sample_data):
        """Test getting overall sentiment statistics."""
        stats = get_sentiment_statistics(db_with_sample_data)

        assert stats["total_videos"] == 3
        assert "Person A" in stats["by_person"]
        assert "Person B" in stats["by_person"]
        assert stats["by_person"]["Person A"]["count"] == 2


class TestNewQueryFunctions:
    """Test new query functions for V2 schema."""

    def test_get_video_timeline(self, db_with_sample_data):
        """Test timeline generation from sentences."""
        timeline = get_video_timeline(
            "https://youtube.com/watch?v=test1",
            bucket_size=3.0,
            db_path=db_with_sample_data
        )

        assert len(timeline) > 0
        assert all("positive" in bucket for bucket in timeline)
        assert all("neutral" in bucket for bucket in timeline)
        assert all("negative" in bucket for bucket in timeline)

    def test_get_person_sentiment_trends(self, db_with_sample_data):
        """Test sentiment trends over time."""
        trends = get_person_sentiment_trends("Person A", db_with_sample_data)

        assert len(trends) == 2
        assert trends[0]["date"] == "2024-01-01"
        assert trends[1]["date"] == "2024-01-02"
        assert all("overall_sentiment" in t for t in trends)
        assert all("positive_pct" in t for t in trends)

    def test_get_emotion_breakdown(self, db_with_sample_data):
        """Test emotion breakdown retrieval."""
        emotions = get_emotion_breakdown(
            "https://youtube.com/watch?v=test1",
            db_path=db_with_sample_data
        )

        assert "sentence_emotions" in emotions
        assert "video_summary" in emotions
        # Should have sentence-level emotions
        assert len(emotions["sentence_emotions"]) > 0


class TestMigration:
    """Test schema migration functionality."""

    @pytest.fixture
    def legacy_db(self, temp_db):
        """Create a legacy schema database with test data."""
        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Create legacy schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                url TEXT PRIMARY KEY,
                date TEXT,
                person_name TEXT,
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

        # Insert legacy data
        now = datetime.utcnow().isoformat()
        cursor.execute("""
            INSERT INTO videos (
                url, date, person_name, title, transcription, language,
                analysis_json, created_at, updated_at, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "https://youtube.com/watch?v=legacy1",
            "2024-01-01",
            "Legacy Person",
            "Legacy Video",
            "This is legacy transcription",
            "en",
            json.dumps(SAMPLE_ANALYSIS_DATA),
            now,
            now,
            "complete",
            None
        ))

        conn.commit()
        conn.close()
        return temp_db

    def test_migration_creates_backup(self, legacy_db):
        """Test that migration creates backup table."""
        run_migration(legacy_db, dry_run=False)

        conn = get_db_connection(legacy_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='videos_backup'
        """)
        assert cursor.fetchone() is not None

        conn.close()

    def test_migration_preserves_data(self, legacy_db):
        """Test that migration preserves all data."""
        # Get count before migration
        conn = get_db_connection(legacy_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM videos")
        original_count = cursor.fetchone()[0]
        conn.close()

        # Run migration
        report = run_migration(legacy_db, dry_run=False)

        # Verify counts
        assert report.total_videos == original_count
        assert report.migrated == original_count
        assert report.failed == 0

        # Verify data exists in new schema
        conn = get_db_connection(legacy_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM video_metadata")
        new_count = cursor.fetchone()[0]
        conn.close()

        assert new_count == original_count

    def test_rollback_migration(self, legacy_db):
        """Test rollback functionality."""
        # Run migration
        run_migration(legacy_db, dry_run=False)

        # Rollback
        rollback_migration(legacy_db)

        # Verify original table is back
        conn = get_db_connection(legacy_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='videos'
        """)
        assert cursor.fetchone() is not None

        # Verify new tables are gone
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='video_metadata'
        """)
        assert cursor.fetchone() is None

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
