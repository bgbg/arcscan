import yt_dlp
import os

def download_youtube_audio(youtube_url, output_filename="output_audio.mp3"):
    """
    Downloads the audio from a YouTube video URL and saves it as an MP3 file.

    Args:
        youtube_url (str): The URL of the YouTube video.
        output_filename (str): The name of the output audio file.
    """
    try:
        # Ensure the output filename ends with .mp3
        if not output_filename.endswith(".mp3"):
            output_filename += ".mp3"

        # Define options for yt_dlp
        ydl_opts = {
            "format": "bestaudio/best",  # Get the best audio quality
            "outtmpl": output_filename.rsplit('.', 1)[0],  # Remove extension for yt-dlp handling
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",  # Extract audio using FFmpeg
                    "preferredcodec": "mp3",    # Convert to MP3
                    "preferredquality": "192",  # Audio quality
                }
            ],
        }

        # Download and process the YouTube video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from: {youtube_url}")
            ydl.download([youtube_url])

        print(f"Audio downloaded and saved as: {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
youtube_url = input("Enter the YouTube URL: ")
output_file = "youtube_audio1.mp3"  # Change this to your desired output file name
download_youtube_audio(youtube_url, output_file)
