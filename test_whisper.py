import yt_dlp
import openai
import os
import uuid
import re
from dotenv import load_dotenv
from textblob import TextBlob
from collections import Counter
import firebase_admin
from firebase_admin import credentials, firestore
import subprocess
from transformers import pipeline

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load EmoRoBERTa sentiment analysis model
emo_roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Convert model output labels to readable sentiment names
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Download and extract audio from YouTube
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
        final_mp3 = output_path + ".mp3"
        print(f"[✓] Audio downloaded successfully: {final_mp3}")
        return final_mp3
    except Exception as e:
        print(f"[✗] Error during audio download: {e}")
        return None

# Transcribe audio using Whisper
def transcribe_audio(file_path):
    try:
        print(f"[…] Transcribing audio file: {file_path}")
        with open(file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        print("[✓] Transcription complete.")
        return response.get("text", "")
    except Exception as e:
        print(f"[✗] Error during transcription: {e}")
        return None

# Perform sentiment analysis for each sentence
def analyze_sentences(text):
    blob = TextBlob(text)
    raw_sentences = [str(sentence).strip() for sentence in blob.sentences]
    short_sentences = [s for s in raw_sentences if len(s.split()) <= 5]
    repeated_counts = Counter(short_sentences)

    results = []

    for i, sentence in enumerate(raw_sentences, 1):
        polarity = TextBlob(sentence).sentiment.polarity
        basic_sentiment = "Neutral"
        if polarity > 0:
            basic_sentiment = "Positive"
        elif polarity < 0:
            basic_sentiment = "Negative"
        
        if basic_sentiment == "Neutral" and len(sentence.split()) <= 5 and repeated_counts[sentence] >= 3:
            basic_sentiment = "Positive"

        try:
            roberta_result = emo_roberta(sentence)[0]
            final_sentiment = label_map.get(roberta_result["label"], "Neutral")
        except Exception as e:
            print(f"[!] EmoRoBERTa failed on sentence {i}: {e}")
            final_sentiment = "Neutral"

        results.append({
            "index": i,
            "text": sentence,
            "basic_sentiment": basic_sentiment,
            "final_sentiment": final_sentiment
        })

    return results

# Generate a summary and determine the overall dominant sentiment
def calculate_summary_and_overall(sentences):
    final_counts = Counter([s["final_sentiment"] for s in sentences])
    total = sum(final_counts.values())

    summary = {
        sentiment: {
            "count": count,
            "percentage": round((count / total) * 100, 1)
        }
        for sentiment, count in final_counts.items()
    }

    overall_sentiment = max(final_counts, key=final_counts.get)

    return summary, overall_sentiment

# Save all results to Firebase Firestore
def save_to_firestore(user_id, video_url, transcription, sentence_results, summary, overall):
    # Clean video URL to create a safe document ID
    safe_video_id = re.sub(r'\W+', '_', video_url)
    doc_id = f"{user_id}_{safe_video_id}"

    doc_ref = db.collection("analyses").document(doc_id)
    doc_ref.set({
        "user_id": user_id,
        "video_url": video_url,
        "transcription": transcription,
        "sentences": sentence_results,
        "summary": summary,
        "overall_sentiment": overall,
        "created_at": firestore.SERVER_TIMESTAMP
    })
    print(f"✅ Results saved to Firestore with ID: {doc_ref.id}")

# Run full analysis process
if __name__ == "__main__":
    print("Device set to use cpu")
    print("[🔄] Checking for yt-dlp updates...")
    subprocess.run(["yt-dlp", "-U"])

    user_id = input("Enter your user ID (from Firebase Auth): ").strip()
    youtube_url = input("Enter YouTube URL: ").strip()

    audio_file = download_youtube_audio(youtube_url)

    if audio_file and os.path.exists(audio_file):
        transcription = transcribe_audio(audio_file)
        if transcription:
            print("\n====== Transcription Result ======")
            print(transcription)

            sentence_results = analyze_sentences(transcription)
            summary, overall = calculate_summary_and_overall(sentence_results)

            print("\n====== Final Sentiment Summary (EmoRoBERTa) ======")
            for sentiment, data in summary.items():
                print(f"{sentiment}: {data['count']} sentence(s), {data['percentage']}%")
            print(f"\n🎯 Overall video sentiment: {overall}")

            save_to_firestore(user_id, youtube_url, transcription, sentence_results, summary, overall)
        else:
            print("✗ Transcription failed.")

        os.remove(audio_file)
    else:
        print("✗ Audio file could not be processed.")
