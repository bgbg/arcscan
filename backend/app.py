from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI, APIError as OpenAIAPIError
import os
import yt_dlp
import uuid
import re
"""
Firebase imports are optional. In non-Firebase environments (CLI batch processing),
we avoid hard failures if the package isn't installed.
"""
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None
#from textblob import TextBlob
from collections import Counter
from transformers import pipeline as transformers_pipeline
from langdetect import detect
#from googletrans import Translator
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


# Load environment variables
load_dotenv()

# Initialize OpenAI client (>=1.0)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Optional: Setup LangSmith tracing if API key is configured
if os.getenv("LANGSMITH_API_KEY"):
    from langsmith.wrappers import wrap_openai
    try:
        client = wrap_openai(client)
    except Exception as e:
        print(f"Warning: LangSmith wrapping failed: {e}. Continuing without tracing.")

# Initialize Firebase (optional - only needed for web API, not batch processing)
db = None
try:
    if firebase_admin and credentials and firestore and os.path.exists("firebase_credentials.json"):
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    else:
        print("Warning: Firebase not configured or credentials missing. Firebase features disabled.")
except Exception as e:
    print(f"Warning: Firebase initialization failed: {e}. Firebase features disabled.")

# Initialize the translator
#translator = Translator()

# Setup EmoRoBERTa
emo_roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Define core emotions to track (based on Plutchik's wheel of emotions)
CORE_EMOTIONS = [
     "admiration", "approval", "neutral", "optimism",
    "confusion", "joy", "sadness", "anger",
    "fear", "surprise", "disgust", "trust",
    "anticipation", "gratitude", "caring", "realization",
    "curiosity", "excitement", "pride", "amusement",
    "nervousness", "disappointment", "disapproval",
    "relief", "grief", "love", "annoyance"
]

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body structure
class AnalyzeRequest(BaseModel):
    url: str
    user_id: str
    translated: bool = False  # Optional flag to use translated text for analysis

# Generate safe document ID based on video URL only

def clean_video_url(video_url: str) -> str:
    """Remove time markers and keep a clean video URL."""
    parsed = urlparse(video_url)
    query = parse_qs(parsed.query)
    query.pop("t", None)  # remove time parameter if exists
    cleaned_query = urlencode(query, doseq=True)
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, cleaned_query, parsed.fragment))
    return clean_url

def generate_doc_id(video_url):
    clean_url = clean_video_url(video_url)
    return re.sub(r'\W+', '_', clean_url)

# Check if analysis for this video already exists
def check_existing_analysis_by_video(video_url):
    doc_id = generate_doc_id(video_url)
    doc_ref = db.collection("analyses").document(doc_id).get()
    if doc_ref.exists:
        return doc_ref.to_dict()
    return None

# Save analysis to Firestore
def save_analysis(user_id, video_url, transcription, sentence_results, summary, overall, timeline_data, status="complete"):
    doc_id = generate_doc_id(video_url)
    data = {
        "user_id": user_id,
        "video_url": video_url,
        "transcription": transcription,
        "sentences": sentence_results,
        "summary": summary,
        "overall_sentiment": overall,
        "timeline_data": timeline_data,
        "status": status,
        "created_at": firestore.SERVER_TIMESTAMP
    }
    
    # If we have original text information from translation
    if isinstance(transcription, dict) and "original_text" in transcription:
        data["original_text"] = transcription.get("original_text")
        data["translated_text"] = transcription.get("translated_text")
        data["detected_language"] = transcription.get("detected_language")
    
    db.collection("analyses").document(doc_id).set(data)

# Update progress in Firestore
def update_progress(video_url, user_id, status, progress, message=""):
    doc_id = generate_doc_id(video_url)
    db.collection("analysis_progress").document(doc_id).set({
        "user_id": user_id,
        "video_url": video_url,
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": firestore.SERVER_TIMESTAMP
    })

# Download audio from YouTube
def download_youtube_audio(youtube_url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    output_path = os.path.join(output_dir, f"audio_{unique_id}")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_path + ".mp3"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {e}")


def download_youtube_subtitles(youtube_url, output_dir="downloads"):
    """
    Try to download YouTube subtitles if available.
    Returns subtitle data dict with text and segments, or None if no subtitles.
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())

    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,  # Also try auto-generated captions
        'subtitleslangs': ['en', 'he', 'ar'],  # Prefer these languages
        'skip_download': True,  # Don't download video
        'subtitlesformat': 'vtt/best',
        'outtmpl': os.path.join(output_dir, f'subtitle_{unique_id}'),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info to check for subtitles
            info = ydl.extract_info(youtube_url, download=False)

            # Check if subtitles are available
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})

            # Prefer manual subtitles over automatic
            available_subs = subtitles if subtitles else automatic_captions

            if not available_subs:
                return None

            # Try to find subtitles in order of preference
            preferred_langs = ['en', 'he', 'ar']
            selected_lang = None

            for lang in preferred_langs:
                if lang in available_subs:
                    selected_lang = lang
                    break

            # If no preferred language, take the first available
            if not selected_lang and available_subs:
                selected_lang = list(available_subs.keys())[0]

            if not selected_lang:
                return None

            # Download the subtitle
            ydl_opts['subtitleslangs'] = [selected_lang]
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([youtube_url])

            # Find the downloaded subtitle file
            subtitle_file = None
            for ext in ['.vtt', '.srt']:
                potential_file = os.path.join(output_dir, f'subtitle_{unique_id}.{selected_lang}{ext}')
                if os.path.exists(potential_file):
                    subtitle_file = potential_file
                    break

            if not subtitle_file or not os.path.exists(subtitle_file):
                return None

            # Parse the subtitle file
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Clean up subtitle file
            os.remove(subtitle_file)

            # Parse VTT/SRT format to extract text and timestamps
            import re

            # Remove VTT header
            content = re.sub(r'^WEBVTT\n\n', '', content)
            content = re.sub(r'^NOTE.*?\n\n', '', content, flags=re.MULTILINE)

            # Extract segments with timestamps
            segments = []
            full_text = []

            # Pattern for VTT timestamps: 00:00:00.000 --> 00:00:05.000
            pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)

            for start_time, end_time, text in matches:
                # Convert timestamp to seconds
                def timestamp_to_seconds(ts):
                    h, m, s = ts.split(':')
                    s, ms = s.split('.')
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

                # Clean text (remove HTML tags, extra whitespace)
                clean_text = re.sub(r'<[^>]+>', '', text)
                clean_text = re.sub(r'\n', ' ', clean_text)
                clean_text = clean_text.strip()

                if clean_text:
                    segments.append({
                        'start': timestamp_to_seconds(start_time),
                        'end': timestamp_to_seconds(end_time),
                        'text': clean_text
                    })
                    full_text.append(clean_text)

            if not segments:
                return None

            return {
                'text': ' '.join(full_text),
                'segments': segments,
                'language': selected_lang,
                'source': 'youtube_subtitles'
            }

    except Exception as e:
        # If subtitle download fails, return None to fall back to Whisper
        print(f"Subtitle download failed: {e}")
        return None


def validate_and_process_subtitles(subtitle_data):
    """
    Validate subtitle data and prepare for processing.
    Returns dict with text, language, and detected language, or None if invalid.
    """
    if not subtitle_data:
        return None
    
    text = subtitle_data.get('text', '').strip()
    subtitle_lang = subtitle_data.get('language', '')
    
    # Validate: subtitles must have meaningful text
    if not text or len(text) < 10:
        print(f"Subtitle validation failed: insufficient text length ({len(text)} chars)")
        return None
    
    try:
        detected_lang = detect(text)
    except Exception as e:
        print(f"Language detection failed for subtitles: {e}")
        return None
    
    # Return validated data with detected language
    return {
        'text': text,
        'subtitle_language': subtitle_lang,
        'detected_language': detected_lang,
        'source': 'youtube_subtitles'
    }


# Transcribe audio to text using Whisper
# Subtitle-first flow: prefer subtitles, translate if non-English, fallback to Whisper

def transcribe_audio(path):
    """
    Transcribe audio with subtitle-first flow.
    Decision path: Try subtitles → validate → detect language → translate if needed → fallback to Whisper.
    Returns dict with text, detected_language, source, and metadata.
    """
    result = {
        "source": None,
        "detected_language": None,
        "original_text": None,
        "translated_text": None,
    }
    
    try:
        # Note: Called from batch_processor with subtitles already attempted
        # This path handles pure Whisper transcription
        with open(path, "rb") as f:
            whisper_result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # Convert Pydantic model to dict if needed (openai >= 1.x)
        if hasattr(whisper_result, 'model_dump'):
            whisper_dict = whisper_result.model_dump()
        elif hasattr(whisper_result, 'dict'):
            whisper_dict = whisper_result.dict()
        else:
            whisper_dict = dict(whisper_result)

        text = whisper_dict.get("text", "").strip()
        if not text:
            raise ValueError("Whisper returned empty transcription")

        # Detect language from Whisper output
        try:
            detected_lang = detect(text)
        except Exception as e:
            print(f"Language detection failed: {e}")
            detected_lang = "unknown"

        # Translate if non-English
        if detected_lang not in ['en', 'unknown']:
            print(f"Translating from {detected_lang} to English via GPT...")
            translation_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Translate from {detected_lang} to English. Preserve meaning and tone. Output only translated text."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            translated_text = translation_response.choices[0].message.content.strip()
            result["original_text"] = text
            result["translated_text"] = translated_text
            result["text"] = translated_text
        else:
            result["text"] = text

        result["detected_language"] = detected_lang
        result["source"] = "whisper_transcription"
        result["segments"] = whisper_dict.get("segments", [])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

def extract_sentences_with_timestamps(response_data):
    """Extract sentences from Whisper output or subtitles, with best-effort timings."""
    sentences = []

    # Normalize response_data to dict if it's an object
    if hasattr(response_data, 'model_dump'):
        response_data = response_data.model_dump()
    elif hasattr(response_data, '__dict__') and not isinstance(response_data, dict):
        response_data = response_data.__dict__

    # Case 1: Whisper response with segments (has timestamps)
    if isinstance(response_data, dict) and response_data.get("segments"):
        for segment in response_data["segments"]:
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            if text:
                sentences.append({
                    "text": text,
                    "start_time": start,
                    "end_time": end
                })
    # Case 2: Subtitle text without segments (no timestamps)
    else:
        text = ""
        if isinstance(response_data, dict):
            text = response_data.get("text", "")
        elif isinstance(response_data, str):
            text = response_data

        text = text.strip()
        if text:
            import re
            sentence_texts = re.split(r'(?<=[.!?])\s+|[\n]+', text)
            cumulative_time = 0.0
            for sentence_text in sentence_texts:
                sentence_text = sentence_text.strip()
                if sentence_text:
                    duration = max(0.5, len(sentence_text) / 25.0)
                    sentences.append({
                        "text": sentence_text,
                        "start_time": cumulative_time,
                        "end_time": cumulative_time + duration
                    })
                    cumulative_time += duration

    return sentences

# Analyze each sentence with EmoRoBERTa
def analyze_sentences(sentences):
    results = []
    for i, sentence in enumerate(sentences, 1):
        try:
            analysis = emo_roberta(sentence["text"])[0]
            sentiment = label_map.get(analysis["label"], "Neutral")
        except:
            sentiment = "Neutral"
        
        # Store both sentiment and time information
        results.append({
            "index": i,
            "text": sentence["text"],
            "final_sentiment": sentiment,
            "start_time": sentence["start_time"],
            "end_time": sentence["end_time"]
        })
    return results

# Apply smoothing to sentiment values over time
def apply_smoothing(sentence_results, window_size=3):
    # Convert to sentiment values (1 for the assigned sentiment, 0 for others)
    timeline_data = []
    total_duration = max(s["end_time"] for s in sentence_results)
    
    # Create data points every second
    for t in range(int(total_duration) + 1):
        # Find the sentence that contains this timestamp
        current_sentence = None
        for s in sentence_results:
            if s["start_time"] <= t and s["end_time"] >= t:
                current_sentence = s
                break
        
        # If no sentence contains this timestamp, use the nearest one
        if not current_sentence and sentence_results:
            # Find nearest sentence by distance to midpoint
            current_sentence = min(
                sentence_results,
                key=lambda s: min(
                    abs(t - s["start_time"]),
                    abs(t - s["end_time"])
                )
            )
        
        # Add data point
        if current_sentence:
            sentiment = current_sentence["final_sentiment"]
            data_point = {
                "time": t,
                "Positive": 1 if sentiment == "Positive" else 0,
                "Negative": 1 if sentiment == "Negative" else 0,
                "Neutral": 1 if sentiment == "Neutral" else 0
            }
            timeline_data.append(data_point)
    
    # Apply smoothing
    smoothed_data = []
    for i, point in enumerate(timeline_data):
        # Calculate window boundaries
        window_start = max(0, i - window_size // 2)
        window_end = min(len(timeline_data), i + window_size // 2 + 1)
        window = timeline_data[window_start:window_end]
        
        # Calculate moving averages
        positive_avg = sum(p["Positive"] for p in window) / len(window)
        negative_avg = sum(p["Negative"] for p in window) / len(window)
        neutral_avg = sum(p["Neutral"] for p in window) / len(window)
        
        smoothed_data.append({
            "time": point["time"],
            "Positive": positive_avg,
            "Negative": negative_avg,
            "Neutral": neutral_avg
        })
    
    return smoothed_data

# Summarize overall sentiment
def summarize_results(sentences):
    counts = Counter([s["final_sentiment"] for s in sentences])
    total = sum(counts.values())
    summary = {
        key: {
            "count": val,
            "percentage": round((val / total) * 100, 1)
        } for key, val in counts.items()
    }
    overall = max(counts, key=counts.get)
    return summary, overall

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # Check if analysis already exists
    existing = check_existing_analysis_by_video(req.url)
    if existing:
        return existing

    # Create a progress entry
    doc_id = generate_doc_id(req.url)
    update_progress(req.url, req.user_id, "starting", 5, "Starting analysis...")

    try:
        # 1. Download audio
        update_progress(req.url, req.user_id, "downloading", 10, "Downloading audio...")
        audio_file = download_youtube_audio(req.url)
        update_progress(req.url, req.user_id, "downloaded", 20, "Audio downloaded successfully!")
        
        # 2. Transcribe audio (now with translation for Hebrew/Arabic)
        update_progress(req.url, req.user_id, "transcribing", 30, "Converting speech to text...")
        whisper_response = transcribe_audio(audio_file)
        
        # Get the transcribed text
        text = whisper_response.get("text", "")
        
        # Check if we have a translation
        has_translation = isinstance(whisper_response, dict) and "translated_text" in whisper_response
        
        # Store translation information
        translation_info = {}
        if has_translation:
            translation_info = {
                "original_text": whisper_response.get("original_text"),
                "translated_text": whisper_response.get("translated_text"),
                "detected_language": whisper_response.get("detected_language")
            }
        
        # Extract sentences with timestamps
        sentences = extract_sentences_with_timestamps(whisper_response)
        
        # Save partial result with just transcription
        if has_translation:
            save_analysis(
                req.user_id, req.url, whisper_response, [], {}, "", [], "transcribed"
            )
        else:
            save_analysis(
                req.user_id, req.url, text, [], {}, "", [], "transcribed"
            )
        
        update_progress(req.url, req.user_id, "transcribed", 50, "Speech successfully converted to text!")
        
        # 3. Analyze sentences
        update_progress(req.url, req.user_id, "analyzing", 60, "Analyzing emotional content...")
        analysis = analyze_sentences(sentences)
        
        # 4. Create timeline
        update_progress(req.url, req.user_id, "creating_timeline", 80, "Building sentiment timeline...")
        timeline_data = apply_smoothing(analysis)
        
        # 5. Summarize results
        update_progress(req.url, req.user_id, "summarizing", 90, "Creating emotional summary...")
        summary, overall = summarize_results(analysis)
        
        # Save complete analysis
        if has_translation:
            save_analysis(
                req.user_id, req.url, whisper_response, analysis, summary, overall, timeline_data
            )
        else:
            save_analysis(
                req.user_id, req.url, text, analysis, summary, overall, timeline_data
            )
        
        update_progress(req.url, req.user_id, "complete", 100, "Analysis complete!")
        
        # Clean up
        os.remove(audio_file)

        # Create response data
        response_data = {
            "user_id": req.user_id,
            "video_url": req.url,
            "transcription": text,
            "sentences": analysis,
            "summary": summary,
            "overall_sentiment": overall,
            "timeline_data": timeline_data,
            "status": "complete"
        }
        
        # Add translation info if available
        if translation_info:
            response_data.update(translation_info)
            
        return response_data
    except Exception as e:
        update_progress(req.url, req.user_id, "error", 0, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        update_progress(req.url, req.user_id, "analyzing", 60, "Analyzing emotional content...")
        analysis = analyze_sentences(sentences)
        
        # 4. Create timeline
        update_progress(req.url, req.user_id, "creating_timeline", 80, "Building sentiment timeline...")
        timeline_data = apply_smoothing(analysis)
        
        # 5. Summarize results
        update_progress(req.url, req.user_id, "summarizing", 90, "Creating emotional summary...")
        summary, overall = summarize_results(analysis)
        
        # Save complete analysis
        if has_translation:
            save_analysis(
                req.user_id, req.url, whisper_response, analysis, summary, overall, timeline_data
            )
        else:
            save_analysis(
                req.user_id, req.url, text, analysis, summary, overall, timeline_data
            )
        
        update_progress(req.url, req.user_id, "complete", 100, "Analysis complete!")
        
        # Clean up
        os.remove(audio_file)

        # Create response data
        response_data = {
            "user_id": req.user_id,
            "video_url": req.url,
            "transcription": text,
            "sentences": analysis,
            "summary": summary,
            "overall_sentiment": overall,
            "timeline_data": timeline_data,
            "status": "complete"
        }
        
        # Add translation info if available
        if translation_info:
            response_data.update(translation_info)
            
        return response_data
    except Exception as e:
        update_progress(req.url, req.user_id, "error", 0, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# Get analysis progress - FIXED to handle full URLs
@app.get("/progress/{video_url:path}")
async def get_progress(video_url: str):
    try:
        from urllib.parse import unquote
        decoded_url = unquote(video_url)
        print(f"Getting progress for URL: {decoded_url}")  # Add debug logging
        
        doc_id = generate_doc_id(decoded_url)
        print(f"Generated doc_id: {doc_id}")  # Add debug logging
        
        doc_ref = db.collection("analysis_progress").document(doc_id).get()
        
        if doc_ref.exists:
            progress_data = doc_ref.to_dict()
            print(f"Progress found: {progress_data}")  # Add debug logging
            return progress_data
        else:
            print("No progress found, returning not_started")  # Add debug logging
            return {"status": "not_started", "progress": 0}
    except Exception as e:
        print(f"Error getting progress: {e}")
        return {"status": "error", "progress": 0, "message": str(e)}
    
# Get all analyses done by a specific user
@app.get("/history/{user_id}")
async def get_user_history(user_id: str):
    results = []
    docs = db.collection("analyses").where("user_id", "==", user_id).stream()
    for doc in docs:
        results.append(doc.to_dict())
    return results

@app.get("/results/{video_url:path}")
async def get_results(video_url: str):
    existing = check_existing_analysis_by_video(video_url)
    if existing:
        return existing
    raise HTTPException(status_code=404, detail="Analysis not found")


# ------ ADVANCED EMOTIONS ANALYSIS - NEW CODE STARTS HERE -------

#Function to analyze complex emotions using GoEmotions model
def analyze_advanced_emotions(sentences):
    # Initialize the model only when needed (to save memory)
    emotion_classifier = transformers_pipeline(
        "text-classification",
        model="monologg/bert-base-cased-goemotions-original",
        return_all_scores=True
    )
    results = []
    for sentence in sentences:
        text = sentence["text"]
        if len(text.split()) < 3:
            continue
        try:
            emotion_scores = emotion_classifier(text)[0]
            emotions_dict = {item['label']: item['score'] for item in emotion_scores}
            top_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            formatted_emotions = [
                {"emotion": emotion, "score": round(score * 100, 1)}
                for emotion, score in top_emotions if score > 0.1
            ]
            results.append({
                "text": text,
                "start_time": sentence["start_time"],
                "end_time": sentence["end_time"],
                "emotions": formatted_emotions
            })
        except Exception as e:
            print(f"Error analyzing advanced emotions: {e}")
            continue
    return results

# Create an emotion timeline for visualization
def create_emotion_timeline(results, window_size=5):
    # First, identify which emotions are actually present in the data
    detected_emotions = set()
    for result in results:
        for emotion_data in result["emotions"]:
            detected_emotions.add(emotion_data["emotion"])
    
    # Use detected emotions plus core emotions
    tracked_emotions = list(detected_emotions.union(set(CORE_EMOTIONS)))
    
    # Initialize timeline
    timeline = []
    
    # Get total duration of video
    if not results:
        return []
    total_duration = max(r["end_time"] for r in results)
    
    # Create initial data structure with time points
    for t in range(int(total_duration) + 1):
        data_point = {"time": t}
        for emotion in tracked_emotions:
            data_point[emotion] = 0
        timeline.append(data_point)
    
    # Fill in emotion values for each second
    for result in results:
        start = int(result["start_time"])
        end = int(result["end_time"])
        
        for emotion_data in result["emotions"]:
            emotion = emotion_data["emotion"]
            score = emotion_data["score"] / 100  # Convert percentage back to 0-1 scale
            
            # Apply score to each second in the time range
            for t in range(start, end + 1):
                if t < len(timeline):
                    timeline[t][emotion] += score
    
    # Apply smoothing
    smoothed_timeline = []
    for i in range(len(timeline)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(timeline), i + window_size // 2 + 1)
        window = timeline[window_start:window_end]
        
        smoothed_point = {"time": timeline[i]["time"]}
        for emotion in tracked_emotions:
            values = [point.get(emotion, 0) for point in window]
            if values:
                smoothed_point[emotion] = sum(values) / len(values)
            else:
                smoothed_point[emotion] = 0
                
        smoothed_timeline.append(smoothed_point)
    
    return smoothed_timeline

# Summarize emotions across the video
def summarize_emotions(results):
    if not results:
        return {}

    # Collect all emotions with their scores
    all_emotions = []
    for result in results:
        for emotion_data in result["emotions"]:
            all_emotions.append({
                "emotion": emotion_data["emotion"],
                "score": emotion_data["score"],
                "duration": result["end_time"] - result["start_time"]
            })

    # Group by emotion
    emotion_groups = {}
    for item in all_emotions:
        emotion = item["emotion"]
        if emotion not in emotion_groups:
            emotion_groups[emotion] = []
        emotion_groups[emotion].append(item)

    # Calculate weighted average scores (by duration)
    summary = {}
    for emotion, items in emotion_groups.items():
        total_score = sum(item["score"] * item["duration"] for item in items)
        total_duration = sum(item["duration"] for item in items)
        
        if total_duration > 0:
            weighted_score = total_score / total_duration
        else:
            weighted_score = 0
        
        # Only include if significant
        if weighted_score > 5:  # 5% threshold
            summary[emotion] = {
                "average_score": round(weighted_score, 1),
                "occurrences": len(items)
            }

    return summary

# Add a new endpoint for complex emotions
@app.post("/analyze/advanced-emotions")
async def analyze_advanced(req: AnalyzeRequest):
    # First check if basic analysis exists
    doc_id = generate_doc_id(req.url)
    doc_ref = db.collection("analyses").document(doc_id).get()
    if not doc_ref.exists:
        raise HTTPException(status_code=404, detail="Basic analysis not found. Run basic analysis first.")
    
    basic_analysis = doc_ref.to_dict()

    # Check if advanced analysis already exists
    advanced_ref = db.collection("advanced_analyses").document(doc_id).get()
    if advanced_ref.exists:
        return advanced_ref.to_dict()

    try:
        # Get sentences from basic analysis
        sentences = basic_analysis.get("sentences", [])
        
        # Check if we should use translated text
        use_translated = req.translated and "translated_text" in basic_analysis
        
        # Extract just the text and timing info for processing
        if use_translated:
            # For non-English videos, use the translated text for analysis
            sentence_data = []
            for s in sentences:
                # Get translated text if available, otherwise use original
                text_to_use = basic_analysis.get("translated_text", basic_analysis.get("transcription", ""))
                
                # Since the translated text is a single block, use the original text
                # just for timing information
                sentence_data.append({
                    "text": text_to_use,
                    "start_time": s["start_time"], 
                    "end_time": s["end_time"]
                })
        else:
            # For English videos, use the original text
            sentence_data = [
                {
                    "text": s["text"], 
                    "start_time": s["start_time"], 
                    "end_time": s["end_time"]
                } 
                for s in sentences
            ]
        
        update_progress(req.url, req.user_id, "analyzing_advanced", 10, "Analyzing complex emotions...")
        
        # Perform advanced emotion analysis
        advanced_results = analyze_advanced_emotions(sentence_data)
        
        # Extract emotion timeline for visualization
        emotion_timeline = create_emotion_timeline(advanced_results)
        
        # Summarize dominant emotions
        emotion_summary = summarize_emotions(advanced_results)
        
        # Save results
        db.collection("advanced_analyses").document(doc_id).set({
            "user_id": req.user_id,
            "video_url": req.url,
            "sentence_emotions": advanced_results,
            "emotion_timeline": emotion_timeline,
            "emotion_summary": emotion_summary,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        
        update_progress(req.url, req.user_id, "complete_advanced", 100, "Advanced emotion analysis complete!")
        
        return {
            "user_id": req.user_id,
            "video_url": req.url,
            "sentence_emotions": advanced_results,
            "emotion_timeline": emotion_timeline,
            "emotion_summary": emotion_summary
        }
    except Exception as e:
        update_progress(req.url, req.user_id, "error_advanced", 0, f"Error in advanced analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# Add this endpoint to check advanced analysis progress
@app.get("/progress/advanced/{video_url:path}")
async def get_advanced_progress(video_url: str):
    try:
        from urllib.parse import unquote
        decoded_url = unquote(video_url)
        doc_id = generate_doc_id(decoded_url)
        
        # First check if analysis exists
        advanced_ref = db.collection("advanced_analyses").document(doc_id).get()
        if advanced_ref.exists:
            return {"status": "complete", "progress": 100}
        
        # Check progress
        doc_ref = db.collection("analysis_progress").document(doc_id).get()
        
        if doc_ref.exists:
            progress_data = doc_ref.to_dict()
            # Only return if it's advanced progress
            if progress_data.get("status", "").startswith("analyzing_advanced") or progress_data.get("status", "").startswith("complete_advanced"):
                return progress_data
            
        return {"status": "not_started", "progress": 0}
    except Exception as e:
        return {"status": "error", "progress": 0, "message": str(e)}


# Add this endpoint to get advanced results
@app.get("/results/advanced/{video_url:path}")
async def get_advanced_results(video_url: str):
    doc_id = generate_doc_id(video_url)
    doc_ref = db.collection("advanced_analyses").document(doc_id).get()
    if doc_ref.exists:
        return doc_ref.to_dict()

    raise HTTPException(status_code=404, detail="Advanced analysis not found")


# ============================================================================
# BATCH PROCESSING ENDPOINTS
# ============================================================================

# Import batch processing modules with flexible paths (support CLI context)
try:
    from .csv_parser import parse_csv, validate_csv_format
    from .batch_processor import process_video_batch, sync_to_firebase
    from .batch_db import (
        get_all_results,
        get_videos_by_person,
        get_person_summary,
        get_date_range_results,
        get_sentiment_statistics,
        DEFAULT_DB_PATH
    )
except ImportError:
    from csv_parser import parse_csv, validate_csv_format
    from batch_processor import process_video_batch, sync_to_firebase
    from batch_db import (
        get_all_results,
        get_videos_by_person,
        get_person_summary,
        get_date_range_results,
        get_sentiment_statistics,
        DEFAULT_DB_PATH
    )


class BatchAnalyzeRequest(BaseModel):
    csv_data: str
    user_id: Optional[str] = None
    sync_firebase: bool = False
    skip_existing: bool = True


@app.post("/batch/analyze")
async def batch_analyze(req: BatchAnalyzeRequest):
    """
    Batch analyze multiple videos from CSV data.

    CSV format: date,name,url

    Returns:
        Processing results with statistics
    """
    # Validate CSV format
    is_valid, error_msg = validate_csv_format(req.csv_data, is_file_path=False)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Parse CSV
    try:
        videos = parse_csv(req.csv_data, is_file_path=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not videos:
        raise HTTPException(status_code=400, detail="No valid videos found in CSV")

    # Process batch
    result = process_video_batch(
        videos=videos,
        db_path=DEFAULT_DB_PATH,
        skip_existing=req.skip_existing
    )

    # Optionally sync to Firebase
    sync_stats = None
    if req.sync_firebase and req.user_id:
        sync_stats = sync_to_firebase(
            user_id=req.user_id,
            db_path=DEFAULT_DB_PATH
        )

    response = result.to_dict()
    if sync_stats:
        response["firebase_sync"] = sync_stats

    return response


@app.post("/batch/upload")
async def batch_upload(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    sync_firebase: bool = Form(False),
    skip_existing: bool = Form(True)
):
    """
    Upload CSV file for batch analysis.

    Accepts multipart/form-data file upload.

    Returns:
        Processing results with statistics
    """
    # Read CSV file
    contents = await file.read()
    csv_data = contents.decode('utf-8')

    # Use the same logic as batch_analyze
    req = BatchAnalyzeRequest(
        csv_data=csv_data,
        user_id=user_id,
        sync_firebase=sync_firebase,
        skip_existing=skip_existing
    )

    return await batch_analyze(req)


@app.get("/batch/results")
async def get_batch_results(
    person: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Query batch analysis results from SQLite.

    Args:
        person: Filter by person name
        start_date: Filter by start date (YYYY-MM-DD)
        end_date: Filter by end date (YYYY-MM-DD)
        status: Filter by status (complete, error, pending)

    Returns:
        List of analysis results
    """
    # Apply filters
    if person:
        results = get_videos_by_person(person, DEFAULT_DB_PATH)
    elif start_date and end_date:
        results = get_date_range_results(start_date, end_date, DEFAULT_DB_PATH)
    else:
        results = get_all_results(DEFAULT_DB_PATH)

    # Filter by status if provided
    if status:
        results = [r for r in results if r.get("status") == status]

    return {
        "total": len(results),
        "results": results
    }


@app.get("/batch/people")
async def get_people():
    """
    Get list of all people with their video counts.

    Returns:
        List of people with statistics
    """
    stats = get_sentiment_statistics(DEFAULT_DB_PATH)

    people = []
    for name, data in stats.get("by_person", {}).items():
        summary = get_person_summary(name, DEFAULT_DB_PATH)
        people.append({
            "name": name,
            "video_count": data["count"],
            "sentiment_distribution": data["sentiment"],
            "date_range": {
                "earliest": summary.get("earliest_date"),
                "latest": summary.get("latest_date")
            },
            "status": {
                "completed": summary.get("completed", 0),
                "errors": summary.get("errors", 0)
            }
        })

    return {
        "total_people": len(people),
        "people": sorted(people, key=lambda x: x["video_count"], reverse=True)
    }


@app.get("/batch/statistics")
async def get_batch_statistics():
    """
    Get overall batch processing statistics.

    Returns:
        Statistics including sentiment distribution
    """
    return get_sentiment_statistics(DEFAULT_DB_PATH)