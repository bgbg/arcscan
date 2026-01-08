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

# Default database path anchored to backend directory
DEFAULT_DB_PATH = str(Path(__file__).parent / "batch_results.db")


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

    Uses schema V2 with normalized tables for efficient querying.
    For backward compatibility, also maintains analysis_json column.

    Args:
        db_path: Path to the database file
    """
    schema_file = Path(__file__).parent / "schema_v2.sql"

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Load and execute schema from file
    if schema_file.exists():
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        cursor.executescript(schema_sql)
        logger.info(f"Database initialized with schema V2 at {db_path}")
    else:
        # Fallback to legacy schema if schema file not found
        logger.warning(f"Schema file not found: {schema_file}. Using legacy schema.")

        # Create videos table with hybrid schema (legacy)
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

        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_name
            ON videos(person_name)
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

    # Try new schema first, fall back to legacy if needed
    try:
        cursor.execute(
            "SELECT status FROM video_metadata WHERE url = ?",
            (url,)
        )
    except sqlite3.OperationalError:
        # Fall back to legacy schema
        cursor.execute(
            "SELECT status FROM videos WHERE url = ?",
            (url,)
        )

    result = cursor.fetchone()
    conn.close()

    if result:
        return result["status"] == "complete"
    return False


def save_transcription(
    conn: sqlite3.Connection,
    video_url: str,
    data: Dict[str, Any]
) -> int:
    """
    Save transcription data to normalized schema.

    Args:
        conn: Database connection
        video_url: Video URL
        data: Analysis data dictionary

    Returns:
        Transcription ID
    """
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    # Extract transcription fields
    text = data.get("transcription") or data.get("text", "")
    language = data.get("output_language", "")
    source = data.get("source", "whisper")
    detected_language = data.get("detected_language", "")
    subtitle_language = data.get("subtitle_language")
    decision_path = data.get("decision_path", "")

    cursor.execute("""
        INSERT INTO transcriptions (
            video_url, text, language, source, detected_language,
            subtitle_language, decision_path, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        video_url, text, language, source, detected_language,
        subtitle_language, decision_path, now
    ))

    return cursor.lastrowid


def save_translation(
    conn: sqlite3.Connection,
    transcription_id: int,
    data: Dict[str, Any]
) -> None:
    """
    Save translation data to normalized schema.

    Args:
        conn: Database connection
        transcription_id: ID of the transcription
        data: Analysis data dictionary
    """
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    original_text = data.get("original_text", "")
    translated_text = data.get("translated_text", "")
    model = data.get("translation_model", "gpt-3.5-turbo")

    cursor.execute("""
        INSERT INTO translations (
            transcription_id, original_text, translated_text, model, created_at
        ) VALUES (?, ?, ?, ?, ?)
    """, (transcription_id, original_text, translated_text, model, now))


def save_sentences(
    conn: sqlite3.Connection,
    video_url: str,
    data: Dict[str, Any]
) -> List[int]:
    """
    Save sentences with timestamps and sentiment to normalized schema.

    Args:
        conn: Database connection
        video_url: Video URL
        data: Analysis data dictionary

    Returns:
        List of sentence IDs
    """
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    sentences = data.get("sentences", [])
    sentence_ids = []

    for idx, sentence in enumerate(sentences):
        cursor.execute("""
            INSERT INTO sentences (
                video_url, sentence_index, text, start_time, end_time,
                sentiment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            video_url,
            idx,
            sentence.get("text", ""),
            sentence.get("start", 0.0),
            sentence.get("end", 0.0),
            sentence.get("sentiment", "neutral"),
            now
        ))
        sentence_ids.append(cursor.lastrowid)

    return sentence_ids


def save_sentiments(
    conn: sqlite3.Connection,
    video_url: str,
    data: Dict[str, Any]
) -> None:
    """
    Save aggregated sentiment statistics to normalized schema.

    Args:
        conn: Database connection
        video_url: Video URL
        data: Analysis data dictionary
    """
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    summary = data.get("summary", {})
    overall_sentiment = data.get("overall_sentiment", "neutral")

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
        video_url, overall_sentiment,
        positive_count, pos_pct,
        neutral_count, neu_pct,
        negative_count, neg_pct,
        now
    ))


def save_emotions(
    conn: sqlite3.Connection,
    sentence_ids: List[int],
    data: Dict[str, Any]
) -> None:
    """
    Save emotion analysis data to normalized schema.

    Args:
        conn: Database connection
        sentence_ids: List of sentence IDs
        data: Analysis data dictionary
    """
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    sentences = data.get("sentences", [])

    for sentence_id, sentence in zip(sentence_ids, sentences):
        emotions = sentence.get("emotions", {})
        for emotion_name, score in emotions.items():
            cursor.execute("""
                INSERT INTO sentence_emotions (
                    sentence_id, emotion_name, score, created_at
                ) VALUES (?, ?, ?, ?)
            """, (sentence_id, emotion_name, score, now))


def save_video_analysis(
    url: str,
    person_name: str,
    data: Dict[str, Any],
    date: Optional[str] = None,
    title: Optional[str] = None,
    duration: Optional[int] = None,
    status: str = "complete",
    error_message: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Save or update video analysis results.

    Writes to both normalized tables (V2 schema) and maintains analysis_json
    for backward compatibility during transition period.

    Args:
        url: Video URL (primary key)
        person_name: Name of person in the video
        data: Complete analysis results dictionary
        date: Video date (from CSV)
        title: Video title
        duration: Video duration in seconds
        status: Processing status (pending, processing, complete, error)
        error_message: Error message if status is error
        db_path: Path to the database file
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()

    # Serialize full analysis to JSON (for backward compatibility)
    analysis_json = json.dumps(data) if data else "{}"

    try:
        # Check which schema we're using
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='video_metadata'
        """)
        use_new_schema = cursor.fetchone() is not None

        if use_new_schema:
            # Use V2 normalized schema
            # Check if record exists
            cursor.execute(
                "SELECT url FROM video_metadata WHERE url = ?",
                (url,)
            )
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing record in video_metadata
                cursor.execute("""
                    UPDATE video_metadata
                    SET person_name = ?,
                        title = ?,
                        duration = ?,
                        updated_at = ?,
                        status = ?,
                        error_message = ?,
                        analysis_json = ?,
                        date = COALESCE(?, date)
                    WHERE url = ?
                """, (
                    person_name,
                    title,
                    duration,
                    now,
                    status,
                    error_message,
                    analysis_json,
                    date,
                    url
                ))

                # Delete existing normalized data for this video
                # (We'll re-insert fresh data)
                if status == "complete" and data:
                    cursor.execute(
                        "DELETE FROM transcriptions WHERE video_url = ?",
                        (url,)
                    )
                    cursor.execute(
                        "DELETE FROM video_sentiments WHERE video_url = ?",
                        (url,)
                    )
                    # Sentences will cascade delete via FK

                logger.info(f"Updated video analysis for {url}")

            else:
                # Insert new record into video_metadata
                cursor.execute("""
                    INSERT INTO video_metadata (
                        url, date, person_name, title, duration, created_at, updated_at,
                        status, error_message, analysis_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    url, date, person_name, title, duration, now, now,
                    status, error_message, analysis_json
                ))
                logger.info(f"Saved new video analysis for {url}")

            # Write to normalized tables if status is complete and we have data
            if status == "complete" and data:
                # Save transcription
                transcription_id = save_transcription(conn, url, data)

                # Save translation if exists
                if "translated_text" in data and "original_text" in data:
                    save_translation(conn, transcription_id, data)

                # Save sentences
                sentence_ids = save_sentences(conn, url, data)

                # Save sentiments
                save_sentiments(conn, url, data)

                # Save emotions if present
                if sentence_ids and data.get("sentences"):
                    save_emotions(conn, sentence_ids, data)

        else:
            # Fallback to legacy schema
            # Extract key fields from data
            transcription = data.get("transcription", "")
            if isinstance(transcription, dict):
                transcription = transcription.get("translated_text") or transcription.get("text", "")

            language = (
                data.get("output_language")
                or data.get("detected_language")
                or data.get("language", "")
            )

            # Check if record exists
            cursor.execute("SELECT url FROM videos WHERE url = ?", (url,))
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing record
                cursor.execute("""
                    UPDATE videos
                    SET person_name = ?,
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
                    person_name,
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
                logger.info(f"Updated video analysis for {url} (legacy schema)")
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO videos (
                        url, date, person_name, title, transcription,
                        language, analysis_json, created_at, updated_at,
                        status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    url,
                    date,
                    person_name,
                    title,
                    transcription,
                    language,
                    analysis_json,
                    now,
                    now,
                    status,
                    error_message
                ))
                logger.info(f"Saved new video analysis for {url} (legacy schema)")

        conn.commit()

    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving video analysis for {url}: {e}")
        raise
    finally:
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

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='video_metadata'
    """)
    use_new_schema = cursor.fetchone() is not None

    if use_new_schema:
        # Use V2 schema with JOIN
        cursor.execute("""
            SELECT
                v.url,
                v.date,
                v.person_name,
                v.title,
                v.duration,
                v.created_at,
                v.updated_at,
                v.status,
                v.error_message,
                t.text as transcription,
                t.language,
                v.analysis_json
            FROM video_metadata v
            LEFT JOIN transcriptions t ON v.url = t.video_url
                AND t.id = (
                    SELECT MAX(id) FROM transcriptions WHERE video_url = v.url
                )
            ORDER BY v.date DESC
        """)
    else:
        # Use legacy schema
        cursor.execute("""
            SELECT url, date, person_name, title, transcription,
                   language, analysis_json, created_at, updated_at,
                   status, error_message
            FROM videos
            ORDER BY date DESC
        """)

    results = []
    for row in cursor.fetchall():
        result = dict(row)
        # Parse JSON analysis data if present
        if result.get("analysis_json"):
            result["analysis"] = json.loads(result["analysis_json"])
        results.append(result)

    conn.close()
    return results


def get_videos_by_person(
    person_name: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Get all videos for a specific person.

    Args:
        person_name: Name of the person
        db_path: Path to the database file

    Returns:
        List of video analysis dictionaries
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='video_metadata'
    """)
    use_new_schema = cursor.fetchone() is not None

    if use_new_schema:
        # Use V2 schema with JOIN
        cursor.execute("""
            SELECT
                v.url,
                v.date,
                v.person_name,
                v.title,
                v.duration,
                v.created_at,
                v.updated_at,
                v.status,
                v.error_message,
                t.text as transcription,
                t.language,
                v.analysis_json
            FROM video_metadata v
            LEFT JOIN transcriptions t ON v.url = t.video_url
                AND t.id = (
                    SELECT MAX(id) FROM transcriptions WHERE video_url = v.url
                )
            WHERE v.person_name = ?
            ORDER BY v.date DESC
        """, (person_name,))
    else:
        # Use legacy schema
        cursor.execute("""
            SELECT url, date, person_name, title, transcription,
                   language, analysis_json, created_at, updated_at,
                   status, error_message
            FROM videos
            WHERE person_name = ?
            ORDER BY date DESC
        """, (person_name,))

    results = []
    for row in cursor.fetchall():
        result = dict(row)
        if result.get("analysis_json"):
            result["analysis"] = json.loads(result["analysis_json"])
        results.append(result)

    conn.close()
    return results


def get_person_summary(
    person_name: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Get summary statistics for a person.

    Args:
        person_name: Name of the person
        db_path: Path to the database file

    Returns:
        Summary dictionary with video count, date range, etc.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='video_metadata'
    """)
    use_new_schema = cursor.fetchone() is not None

    if use_new_schema:
        cursor.execute("""
            SELECT
                COUNT(*) as total_videos,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM video_metadata
            WHERE person_name = ?
        """, (person_name,))
    else:
        cursor.execute("""
            SELECT
                COUNT(*) as total_videos,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM videos
            WHERE person_name = ?
        """, (person_name,))

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

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='video_metadata'
    """)
    use_new_schema = cursor.fetchone() is not None

    if use_new_schema:
        # Use V2 schema with JOIN
        cursor.execute("""
            SELECT
                v.url,
                v.date,
                v.person_name,
                v.title,
                v.duration,
                v.created_at,
                v.updated_at,
                v.status,
                v.error_message,
                t.text as transcription,
                t.language,
                v.analysis_json
            FROM video_metadata v
            LEFT JOIN transcriptions t ON v.url = t.video_url
                AND t.id = (
                    SELECT MAX(id) FROM transcriptions WHERE video_url = v.url
                )
            WHERE v.date BETWEEN ? AND ?
            ORDER BY v.date DESC
        """, (start_date, end_date))
    else:
        # Use legacy schema
        cursor.execute("""
            SELECT url, date, person_name, title, transcription,
                   language, analysis_json, created_at, updated_at,
                   status, error_message
            FROM videos
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
        """, (start_date, end_date))

    results = []
    for row in cursor.fetchall():
        result = dict(row)
        if result.get("analysis_json"):
            result["analysis"] = json.loads(result["analysis_json"])
        results.append(result)

    conn.close()
    return results


def get_sentiment_statistics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Get overall sentiment statistics across all videos.

    Uses video_sentiments table if available (V2 schema), falls back to
    parsing analysis_json for legacy schema.

    Args:
        db_path: Path to the database file

    Returns:
        Statistics dictionary with sentiment distribution
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='video_sentiments'
    """)
    use_new_schema = cursor.fetchone() is not None

    stats = {
        "total_videos": 0,
        "by_person": {},
        "overall_sentiment": {"positive": 0, "neutral": 0, "negative": 0}
    }

    if use_new_schema:
        # Use V2 schema - query video_sentiments table directly
        cursor.execute("""
            SELECT
                v.person_name,
                s.overall_sentiment,
                s.positive_count,
                s.neutral_count,
                s.negative_count
            FROM video_metadata v
            INNER JOIN video_sentiments s ON v.url = s.video_url
            WHERE v.status = 'complete'
        """)

        for row in cursor.fetchall():
            stats["total_videos"] += 1
            person = row["person_name"]
            overall = row["overall_sentiment"].lower()

            if person not in stats["by_person"]:
                stats["by_person"][person] = {
                    "count": 0,
                    "sentiment": {"positive": 0, "neutral": 0, "negative": 0}
                }

            stats["by_person"][person]["count"] += 1

            # Increment sentiment counts
            if overall in ["positive", "neutral", "negative"]:
                stats["overall_sentiment"][overall] += 1
                stats["by_person"][person]["sentiment"][overall] += 1

    else:
        # Fallback to legacy schema - parse analysis_json
        cursor.execute("""
            SELECT person_name, analysis_json, status
            FROM videos
            WHERE status = 'complete'
        """)

        for row in cursor.fetchall():
            stats["total_videos"] += 1
            person = row["person_name"]

            if person not in stats["by_person"]:
                stats["by_person"][person] = {
                    "count": 0,
                    "sentiment": {"positive": 0, "neutral": 0, "negative": 0}
                }

            stats["by_person"][person]["count"] += 1

            # Parse analysis JSON and extract sentiment
            if row["analysis_json"]:
                try:
                    analysis = json.loads(row["analysis_json"])
                    overall_sentiment = analysis.get("overall_sentiment", "").lower()

                    if overall_sentiment in ["positive", "neutral", "negative"]:
                        stats["overall_sentiment"][overall_sentiment] += 1
                        stats["by_person"][person]["sentiment"][overall_sentiment] += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing analysis JSON: {e}")

    conn.close()
    return stats


def get_video_timeline(
    video_url: str,
    bucket_size: float = 30.0,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Get sentiment timeline for a video aggregated into time buckets.

    Aggregates sentence-level sentiment data from the sentences table into
    time buckets on read (flexible granularity).

    Args:
        video_url: Video URL
        bucket_size: Time bucket size in seconds (default: 30.0)
        db_path: Path to the database file

    Returns:
        List of timeline buckets with sentiment distribution
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='sentences'
    """)
    use_new_schema = cursor.fetchone() is not None

    if not use_new_schema:
        # Fallback: parse timeline from analysis_json
        cursor.execute(
            "SELECT analysis_json FROM videos WHERE url = ?",
            (video_url,)
        )
        row = cursor.fetchone()
        conn.close()

        if row and row["analysis_json"]:
            try:
                analysis = json.loads(row["analysis_json"])
                return analysis.get("timeline_data", [])
            except json.JSONDecodeError:
                return []
        return []

    # V2 schema: aggregate from sentences table
    cursor.execute("""
        SELECT
            start_time,
            end_time,
            sentiment
        FROM sentences
        WHERE video_url = ?
        ORDER BY start_time ASC
    """, (video_url,))

    sentences = cursor.fetchall()
    conn.close()

    if not sentences:
        return []

    # Determine video duration
    max_time = max(s["end_time"] for s in sentences)

    # Create time buckets
    timeline = []
    current_time = 0.0

    while current_time < max_time:
        bucket_end = current_time + bucket_size
        bucket = {
            "start": current_time,
            "end": bucket_end,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "total": 0
        }

        # Count sentences in this bucket
        for sentence in sentences:
            # Sentence overlaps with bucket if its start is within bucket range
            if current_time <= sentence["start_time"] < bucket_end:
                sentiment = sentence["sentiment"].lower()
                if sentiment in ["positive", "neutral", "negative"]:
                    bucket[sentiment] += 1
                    bucket["total"] += 1

        # Calculate percentages
        if bucket["total"] > 0:
            bucket["positive_pct"] = (bucket["positive"] / bucket["total"]) * 100
            bucket["neutral_pct"] = (bucket["neutral"] / bucket["total"]) * 100
            bucket["negative_pct"] = (bucket["negative"] / bucket["total"]) * 100
        else:
            bucket["positive_pct"] = 0.0
            bucket["neutral_pct"] = 0.0
            bucket["negative_pct"] = 0.0

        timeline.append(bucket)
        current_time = bucket_end

    return timeline


def get_person_sentiment_trends(
    person_name: str,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Get sentiment trends over time for a specific person.

    Returns time-series data showing how sentiment changes across their videos.

    Args:
        person_name: Name of the person
        db_path: Path to the database file

    Returns:
        List of sentiment data points sorted by date
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='video_sentiments'
    """)
    use_new_schema = cursor.fetchone() is not None

    if use_new_schema:
        # Use V2 schema
        cursor.execute("""
            SELECT
                v.url,
                v.date,
                v.title,
                s.overall_sentiment,
                s.positive_pct,
                s.neutral_pct,
                s.negative_pct
            FROM video_metadata v
            INNER JOIN video_sentiments s ON v.url = s.video_url
            WHERE v.person_name = ? AND v.status = 'complete'
            ORDER BY v.date ASC
        """, (person_name,))
    else:
        # Fallback to legacy schema
        cursor.execute("""
            SELECT url, date, title, analysis_json
            FROM videos
            WHERE person_name = ? AND status = 'complete'
            ORDER BY date ASC
        """, (person_name,))

    trends = []
    for row in cursor.fetchall():
        if use_new_schema:
            trends.append({
                "url": row["url"],
                "date": row["date"],
                "title": row["title"],
                "overall_sentiment": row["overall_sentiment"],
                "positive_pct": row["positive_pct"],
                "neutral_pct": row["neutral_pct"],
                "negative_pct": row["negative_pct"]
            })
        else:
            # Parse from analysis_json
            if row["analysis_json"]:
                try:
                    analysis = json.loads(row["analysis_json"])
                    summary = analysis.get("summary", {})
                    overall = analysis.get("overall_sentiment", "neutral")

                    positive = summary.get("positive", 0)
                    neutral = summary.get("neutral", 0)
                    negative = summary.get("negative", 0)
                    total = positive + neutral + negative

                    if total > 0:
                        pos_pct = (positive / total) * 100
                        neu_pct = (neutral / total) * 100
                        neg_pct = (negative / total) * 100
                    else:
                        pos_pct = neu_pct = neg_pct = 0.0

                    trends.append({
                        "url": row["url"],
                        "date": row["date"],
                        "title": row["title"],
                        "overall_sentiment": overall,
                        "positive_pct": pos_pct,
                        "neutral_pct": neu_pct,
                        "negative_pct": neg_pct
                    })
                except json.JSONDecodeError:
                    continue

    conn.close()
    return trends


def get_emotion_breakdown(
    video_url: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Get emotion analysis breakdown for a video.

    Returns both sentence-level emotions and video-level emotion summary.

    Args:
        video_url: Video URL
        db_path: Path to the database file

    Returns:
        Dictionary with sentence_emotions and video_summary
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Check which schema we're using
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='sentence_emotions'
    """)
    use_new_schema = cursor.fetchone() is not None

    result = {
        "sentence_emotions": [],
        "video_summary": {}
    }

    if not use_new_schema:
        # Fallback: parse from analysis_json
        cursor.execute(
            "SELECT analysis_json FROM videos WHERE url = ?",
            (video_url,)
        )
        row = cursor.fetchone()
        conn.close()

        if row and row["analysis_json"]:
            try:
                analysis = json.loads(row["analysis_json"])
                # Extract emotion data if available
                sentences = analysis.get("sentences", [])
                for sentence in sentences:
                    if "emotions" in sentence:
                        result["sentence_emotions"].append({
                            "text": sentence.get("text", ""),
                            "emotions": sentence["emotions"]
                        })
            except json.JSONDecodeError:
                pass

        return result

    # V2 schema: query emotion tables
    # Get sentence-level emotions
    cursor.execute("""
        SELECT
            s.sentence_index,
            s.text,
            se.emotion_name,
            se.score
        FROM sentences s
        INNER JOIN sentence_emotions se ON s.id = se.sentence_id
        WHERE s.video_url = ?
        ORDER BY s.sentence_index, se.score DESC
    """, (video_url,))

    # Group by sentence
    sentences_dict = {}
    for row in cursor.fetchall():
        idx = row["sentence_index"]
        if idx not in sentences_dict:
            sentences_dict[idx] = {
                "index": idx,
                "text": row["text"],
                "emotions": {}
            }
        sentences_dict[idx]["emotions"][row["emotion_name"]] = row["score"]

    result["sentence_emotions"] = [sentences_dict[i] for i in sorted(sentences_dict.keys())]

    # Get video-level emotion summary
    cursor.execute("""
        SELECT *
        FROM video_emotion_summary
        WHERE video_url = ?
    """, (video_url,))

    row = cursor.fetchone()
    if row:
        # Convert to dict and remove metadata columns
        summary = dict(row)
        summary.pop("video_url", None)
        summary.pop("created_at", None)
        result["video_summary"] = summary

    conn.close()
    return result


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

    report += "\n## By Person\n\n"
    for person, data in sorted(stats['by_person'].items()):
        report += f"### {person}\n\n"
        report += f"**Videos**: {data['count']}\n\n"
        report += "**Sentiment**:\n"
        for sentiment, count in data['sentiment'].items():
            if data['count'] > 0:
                percentage = (count / data['count']) * 100
                report += f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)\n"
        report += "\n"

    return report
