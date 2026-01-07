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
            download_youtube_subtitles,
            validate_and_process_subtitles,
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
            download_youtube_subtitles,
            validate_and_process_subtitles,
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
        'download_youtube_subtitles': download_youtube_subtitles,
        'validate_and_process_subtitles': validate_and_process_subtitles,
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
    person_name: str,
    date: str,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[bool, Optional[str]]:
    """
    Process a single video: download, transcribe, analyze.

    Args:
        url: YouTube video URL
        person_name: Name of person in video
        date: Video date
        db_path: Path to SQLite database

    Returns:
        Tuple of (success, error_message)
    """
    # Lazy load app functions
    funcs = _import_app_functions()
    download_youtube_audio = funcs['download_youtube_audio']
    download_youtube_subtitles = funcs['download_youtube_subtitles']
    validate_and_process_subtitles = funcs['validate_and_process_subtitles']
    transcribe_audio = funcs['transcribe_audio']
    extract_sentences_with_timestamps = funcs['extract_sentences_with_timestamps']
    analyze_sentences = funcs['analyze_sentences']
    apply_smoothing = funcs['apply_smoothing']
    summarize_results = funcs['summarize_results']

    try:
        logger.info(f"Processing video: {url}")

        whisper_response = None
        decision_log = []
        audio_file = None

        # 1. Try to download and validate subtitles first (preferred: free, faster)
        logger.debug("Attempting to download subtitles...")
        raw_subtitles = download_youtube_subtitles(url)
        
        if raw_subtitles:
            subtitle_data = validate_and_process_subtitles(raw_subtitles)
            if subtitle_data:
                detected_lang = subtitle_data.get('detected_language', 'unknown')
                logger.info(f"✓ Found and validated subtitles (detected lang: {detected_lang})")
                decision_log.append(f"subtitle_{detected_lang}")
                
                # Translate to English if non-English
                if detected_lang not in ['en', 'unknown']:
                    logger.info(f"Translating subtitles from {detected_lang} to English...")
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        
                        translation_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "Translate to English. Output only translated text."},
                                {"role": "user", "content": subtitle_data['text']}
                            ],
                            temperature=0.3
                        )
                        translated = translation_response.choices[0].message.content.strip()
                        subtitle_data['original_text'] = subtitle_data['text']
                        subtitle_data['translated_text'] = translated
                        subtitle_data['text'] = translated
                        decision_log.append("translated")
                        logger.info("✓ Subtitle translation complete")
                    except Exception as e:
                        logger.warning(f"Subtitle translation failed: {e}. Falling back to Whisper.")
                        subtitle_data = None
                
                if subtitle_data:
                    whisper_response = subtitle_data
        
        # 2. Fallback to Whisper if no valid subtitles
        if not whisper_response:
            logger.info("No valid subtitles; using Whisper transcription")
            logger.debug("Downloading audio...")
            try:
                audio_file = download_youtube_audio(url)
            except Exception as e:
                error_detail = getattr(e, 'detail', str(e))
                raise Exception(f"Download failed: {error_detail}")

            logger.debug("Transcribing audio with Whisper API...")
            try:
                whisper_response = transcribe_audio(audio_file)
                decision_log.append("whisper")
                logger.info(f"✓ Whisper transcription complete (detected lang: {whisper_response.get('detected_language')})")
            except Exception as e:
                error_detail = getattr(e, 'detail', str(e))
                raise Exception(f"Transcription failed: {error_detail}")
            finally:
                if audio_file and os.path.exists(audio_file):
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
            person_name=person_name,
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
                person_name=person_name,
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
        videos: List of tuples (date, person_name, url)
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

    for idx, (date, person_name, url) in enumerate(videos, 1):
        # Report progress
        if progress_callback:
            progress_callback(idx, result.total, f"Processing: {person_name}")

        logger.info(f"[{idx}/{result.total}] Processing: {person_name} - {date}")

        # Check if video already processed
        if skip_existing and check_video_exists(url, db_path):
            logger.info(f"Video already processed (cached), skipping: {url}")
            result.skipped += 1
            continue

        # Process the video
        success, error = process_single_video(url, person_name, date, db_path)

        if success:
            result.successful += 1
        else:
            result.failed += 1
            result.errors.append({
                "url": url,
                "person": person_name,
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
            SELECT url, person_name, analysis_json, status
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

            # Add person metadata
            analysis["person_name"] = video["person_name"]

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
