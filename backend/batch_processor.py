"""
Core batch processing logic for multiple video analysis.

This module orchestrates the batch processing of multiple videos,
handling caching, error recovery, and progress tracking.
"""

import os
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from datetime import datetime

# Import database functions
try:
    from .batch_db import (
        init_database,
        check_video_exists,
        save_video_analysis,
        DEFAULT_DB_PATH
    )
except ImportError:
    from batch_db import (
        init_database,
        check_video_exists,
        save_video_analysis,
        DEFAULT_DB_PATH
    )

logger = logging.getLogger(__name__)


def _import_app_functions():
    """Lazy import of app functions to avoid circular dependencies."""
    try:
        from .app import (
            download_youtube_audio,
            transcribe_audio,
            extract_sentences_with_timestamps,
            analyze_sentences,
            apply_smoothing,
            summarize_results,
            save_analysis as firebase_save_analysis,
            db as firebase_db
        )
    except ImportError:
        from app import (
            download_youtube_audio,
            transcribe_audio,
            extract_sentences_with_timestamps,
            analyze_sentences,
            apply_smoothing,
            summarize_results,
            save_analysis as firebase_save_analysis,
            db as firebase_db
        )

    return {
        'download_youtube_audio': download_youtube_audio,
        'transcribe_audio': transcribe_audio,
        'extract_sentences_with_timestamps': extract_sentences_with_timestamps,
        'analyze_sentences': analyze_sentences,
        'apply_smoothing': apply_smoothing,
        'summarize_results': summarize_results,
        'firebase_save_analysis': firebase_save_analysis,
        'firebase_db': firebase_db
    }


class BatchProcessingResult:
    """Container for batch processing results."""

    def __init__(self):
        self.total = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Batch Processing Results:\n"
            f"  Total: {self.total}\n"
            f"  Successful: {self.successful}\n"
            f"  Failed: {self.failed}\n"
            f"  Skipped (cached): {self.skipped}\n"
        )


def process_single_video(
    url: str,
    politician_name: str,
    date: str,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[bool, Optional[str]]:
    """
    Process a single video: download, transcribe, analyze.

    Args:
        url: YouTube video URL
        politician_name: Name of politician in video
        date: Video date
        db_path: Path to SQLite database

    Returns:
        Tuple of (success, error_message)
    """
    # Lazy load app functions
    funcs = _import_app_functions()
    download_youtube_audio = funcs['download_youtube_audio']
    transcribe_audio = funcs['transcribe_audio']
    extract_sentences_with_timestamps = funcs['extract_sentences_with_timestamps']
    analyze_sentences = funcs['analyze_sentences']
    apply_smoothing = funcs['apply_smoothing']
    summarize_results = funcs['summarize_results']

    try:
        logger.info(f"Processing video: {url}")

        # 1. Download audio
        logger.debug("Downloading audio...")
        try:
            audio_file = download_youtube_audio(url)
        except Exception as e:
            # Handle both HTTPException and regular exceptions
            error_detail = getattr(e, 'detail', str(e))
            raise Exception(f"Download failed: {error_detail}")

        # 2. Transcribe audio (with translation if Hebrew/Arabic)
        logger.debug("Transcribing audio...")
        try:
            whisper_response = transcribe_audio(audio_file)
        except Exception as e:
            error_detail = getattr(e, 'detail', str(e))
            raise Exception(f"Transcription failed: {error_detail}")
        finally:
            # Clean up audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)

        # Get the transcribed text
        text = whisper_response.get("text", "")

        # Check if we have a translation
        has_translation = isinstance(whisper_response, dict) and "translated_text" in whisper_response

        # Extract sentences with timestamps
        sentences = extract_sentences_with_timestamps(whisper_response)

        # 3. Analyze sentences
        logger.debug("Analyzing sentiment...")
        analysis = analyze_sentences(sentences)

        # 4. Create timeline
        logger.debug("Creating timeline...")
        timeline_data = apply_smoothing(analysis)

        # 5. Summarize results
        logger.debug("Summarizing results...")
        summary, overall = summarize_results(analysis)

        # 6. Save to SQLite
        result_data = {
            "video_url": url,
            "transcription": whisper_response if has_translation else text,
            "sentences": analysis,
            "summary": summary,
            "overall_sentiment": overall,
            "timeline_data": timeline_data,
            "detected_language": whisper_response.get("detected_language") if has_translation else None,
        }

        # Add translation info if available
        if has_translation:
            result_data.update({
                "original_text": whisper_response.get("original_text"),
                "translated_text": whisper_response.get("translated_text"),
            })

        save_video_analysis(
            url=url,
            politician_name=politician_name,
            data=result_data,
            date=date,
            title=None,  # We could extract from YouTube metadata if needed
            status="complete",
            db_path=db_path
        )

        logger.info(f"Successfully processed video: {url}")
        return True, None

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing video {url}: {error_msg}")

        # Save error to database
        try:
            save_video_analysis(
                url=url,
                politician_name=politician_name,
                data={},
                date=date,
                status="error",
                error_message=error_msg,
                db_path=db_path
            )
        except Exception as save_error:
            logger.error(f"Failed to save error to database: {save_error}")

        return False, error_msg


def process_video_batch(
    videos: List[Tuple[str, str, str]],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    db_path: str = DEFAULT_DB_PATH,
    skip_existing: bool = True
) -> BatchProcessingResult:
    """
    Process a batch of videos.

    Args:
        videos: List of tuples (date, politician_name, url)
        progress_callback: Optional callback function(current, total, message)
        db_path: Path to SQLite database
        skip_existing: If True, skip videos that already exist in database

    Returns:
        BatchProcessingResult with statistics
    """
    # Initialize database
    init_database(db_path)

    result = BatchProcessingResult()
    result.total = len(videos)

    logger.info(f"Starting batch processing of {result.total} videos")

    for idx, (date, politician_name, url) in enumerate(videos, 1):
        # Report progress
        if progress_callback:
            progress_callback(idx, result.total, f"Processing: {politician_name}")

        logger.info(f"[{idx}/{result.total}] Processing: {politician_name} - {date}")

        # Check if video already processed
        if skip_existing and check_video_exists(url, db_path):
            logger.info(f"Video already processed (cached), skipping: {url}")
            result.skipped += 1
            continue

        # Process the video
        success, error = process_single_video(url, politician_name, date, db_path)

        if success:
            result.successful += 1
        else:
            result.failed += 1
            result.errors.append({
                "url": url,
                "politician": politician_name,
                "date": date,
                "error": error
            })

        logger.info(
            f"Progress: {idx}/{result.total} - "
            f"Success: {result.successful}, Failed: {result.failed}, Skipped: {result.skipped}"
        )

    logger.info(f"Batch processing complete:\n{result}")
    return result


def sync_to_firebase(
    user_id: str,
    video_urls: Optional[List[str]] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Sync SQLite results to Firebase Firestore.

    Args:
        user_id: Firebase user ID
        video_urls: Optional list of specific URLs to sync (if None, sync all complete)
        db_path: Path to SQLite database

    Returns:
        Dictionary with sync statistics
    """
    # Lazy load functions
    funcs = _import_app_functions()
    firebase_save_analysis = funcs['firebase_save_analysis']

    try:
        from .batch_db import get_all_results, get_db_connection
    except ImportError:
        from batch_db import get_all_results, get_db_connection

    import json

    logger.info("Starting Firebase sync...")

    stats = {
        "total": 0,
        "synced": 0,
        "failed": 0,
        "errors": []
    }

    # Get videos to sync
    if video_urls:
        # Sync specific URLs
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT url, politician_name, analysis_json, status
            FROM videos
            WHERE url IN ({}) AND status = 'complete'
        """.format(','.join('?' * len(video_urls))), video_urls)
        videos = [dict(row) for row in cursor.fetchall()]
        conn.close()
    else:
        # Sync all complete videos
        all_results = get_all_results(db_path)
        videos = [v for v in all_results if v['status'] == 'complete']

    stats["total"] = len(videos)

    for video in videos:
        try:
            # Parse analysis JSON
            analysis = json.loads(video['analysis_json']) if video['analysis_json'] else {}

            # Prepare data for Firebase (match existing format)
            transcription = analysis.get("transcription", "")
            sentences = analysis.get("sentences", [])
            summary = analysis.get("summary", {})
            overall = analysis.get("overall_sentiment", "")
            timeline = analysis.get("timeline_data", [])

            # Add politician metadata
            analysis["politician_name"] = video["politician_name"]

            # Call existing Firebase save function
            firebase_save_analysis(
                user_id=user_id,
                video_url=video["url"],
                transcription=transcription,
                sentence_results=sentences,
                summary=summary,
                overall=overall,
                timeline_data=timeline,
                status="complete"
            )

            stats["synced"] += 1
            logger.info(f"Synced to Firebase: {video['url']}")

        except Exception as e:
            stats["failed"] += 1
            error_msg = str(e)
            stats["errors"].append({
                "url": video["url"],
                "error": error_msg
            })
            logger.error(f"Failed to sync {video['url']} to Firebase: {error_msg}")

    logger.info(
        f"Firebase sync complete: {stats['synced']}/{stats['total']} synced, "
        f"{stats['failed']} failed"
    )

    return stats
