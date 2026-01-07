# Batch Video Analysis Guide

This guide explains how to use ArcScan's batch processing capabilities to analyze multiple YouTube videos non-interactively.

## Overview

Batch processing allows you to analyze multiple YouTube videos from a CSV file, with results stored in a local SQLite database. This is useful for:

- Analyzing large collections of political speeches
- Building historical sentiment datasets
- Comparing rhetoric across politicians or time periods
- Automating regular analysis workflows

## Features

- **Dual Interface**: Use either REST API endpoints or standalone CLI script
- **SQLite Caching**: Avoid reprocessing videos with local database
- **Politician Tracking**: Automatically extract and track politician names
- **Resume Support**: Safely restart interrupted batch jobs
- **Export Formats**: Export results to JSON or CSV
- **Firebase Sync**: Optionally sync results to web app
- **Progress Tracking**: Real-time progress updates
- **Error Resilience**: Continue processing on individual video failures

## Quick Start

### 1. Prepare Your CSV File

Create a CSV file with `date,name,url` columns:

```csv
date,name,url
2011-05-24,נאום ראש הממשלה בנימין נתניהו בפני שני בתי הקונגרס של ארה"ב,https://www.youtube.com/watch?v=4H3Kyt1iGEE
2021-06-13,נאום ראש הממשלה נפתלי בנט במליאת הכנסת,https://www.youtube.com/watch?v=mqZam_BkTDM
2022-07-02,הצהרת ראש הממשלה יאיר לפיד עם כניסתו לתפקיד,https://www.youtube.com/watch?v=YYx2BHzxNKI
```

**Note**: The `name` field should contain the politician's name or a description that includes it. The system will attempt to extract the politician name automatically.

### 2. Run Batch Analysis

#### Option A: Using the CLI Script (Recommended for Development)

```bash
# Basic usage
python batch_analyze.py --input videos.csv

# With JSON export
python batch_analyze.py --input videos.csv --export-json results.json

# With Firebase sync
python batch_analyze.py --input videos.csv --sync-firebase --user-id YOUR_USER_ID

# Preview without processing
python batch_analyze.py --input videos.csv --dry-run

# Filter by politician
python batch_analyze.py --input videos.csv --politician "Benjamin Netanyahu"

# Generate report from existing results
python batch_analyze.py --report
```

#### Option B: Using the API

Start the backend server:

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Then make API requests:

```bash
# Analyze from CSV data
curl -X POST http://localhost:8000/batch/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "csv_data": "date,name,url\n2011-05-24,Netanyahu UN Speech,https://www.youtube.com/watch?v=...",
    "skip_existing": true
  }'

# Upload CSV file
curl -X POST http://localhost:8000/batch/upload \
  -F "file=@videos.csv" \
  -F "skip_existing=true"

# Query results
curl http://localhost:8000/batch/results?politician=Benjamin%20Netanyahu

# Get politician list
curl http://localhost:8000/batch/politicians

# Get statistics
curl http://localhost:8000/batch/statistics
```

## CLI Reference

### Commands

```
python batch_analyze.py [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--input`, `-i PATH` | Path to input CSV file |
| `--export-json PATH` | Export results to JSON file |
| `--export-csv PATH` | Export results to CSV file |
| `--sync-firebase` | Sync results to Firebase after processing |
| `--user-id ID` | Firebase user ID (required with `--sync-firebase`) |
| `--politician NAME` | Filter videos by politician name |
| `--dry-run` | Preview videos without processing |
| `--report` | Generate and print analysis report |
| `--config PATH` | Path to configuration file (default: `batch_config.json`) |
| `--db-path PATH` | Path to SQLite database (default: `batch_results.db`) |
| `--skip-existing` | Skip videos already in database (default: True) |
| `--verbose`, `-v` | Enable verbose logging |

### Configuration File

Create a `batch_config.json` file to set default options:

```json
{
  "input_file": "videos.csv",
  "user_id": "your-firebase-user-id",
  "db_path": "batch_results.db",
  "export_json": "results.json",
  "export_csv": null,
  "sync_firebase": false
}
```

CLI arguments override config file values.

## API Reference

### Endpoints

#### `POST /batch/analyze`

Process batch of videos from CSV data.

**Request Body**:
```json
{
  "csv_data": "date,name,url\n...",
  "user_id": "optional-user-id",
  "sync_firebase": false,
  "skip_existing": true
}
```

**Response**:
```json
{
  "total": 10,
  "successful": 8,
  "failed": 1,
  "skipped": 1,
  "errors": [
    {
      "url": "https://...",
      "politician": "Name",
      "date": "2023-01-01",
      "error": "Download failed: ..."
    }
  ]
}
```

#### `POST /batch/upload`

Upload CSV file for processing.

**Form Data**:
- `file`: CSV file
- `user_id`: Optional user ID
- `sync_firebase`: Boolean (default: false)
- `skip_existing`: Boolean (default: true)

**Response**: Same as `/batch/analyze`

#### `GET /batch/results`

Query analysis results.

**Query Parameters**:
- `politician`: Filter by politician name
- `start_date`: Filter by start date (YYYY-MM-DD)
- `end_date`: Filter by end date (YYYY-MM-DD)
- `status`: Filter by status (complete, error, pending)

**Response**:
```json
{
  "total": 5,
  "results": [
    {
      "url": "https://...",
      "date": "2023-01-01",
      "politician_name": "Name",
      "language": "he",
      "status": "complete",
      "analysis": { ... }
    }
  ]
}
```

#### `GET /batch/politicians`

Get list of all politicians with statistics.

**Response**:
```json
{
  "total_politicians": 3,
  "politicians": [
    {
      "name": "Benjamin Netanyahu",
      "video_count": 12,
      "sentiment_distribution": {
        "positive": 5,
        "neutral": 4,
        "negative": 3
      },
      "date_range": {
        "earliest": "2011-05-24",
        "latest": "2023-10-07"
      },
      "status": {
        "completed": 11,
        "errors": 1
      }
    }
  ]
}
```

#### `GET /batch/statistics`

Get overall sentiment statistics.

**Response**:
```json
{
  "total_videos": 20,
  "overall_sentiment": {
    "positive": 8,
    "neutral": 7,
    "negative": 5
  },
  "by_politician": { ... }
}
```

## SQLite Database Schema

The batch processor uses SQLite for local caching with a hybrid schema design:

### `videos` Table

| Column | Type | Description |
|--------|------|-------------|
| `url` | TEXT (PK) | YouTube video URL |
| `date` | TEXT | Video date (YYYY-MM-DD) |
| `politician_name` | TEXT | Extracted politician name |
| `title` | TEXT | Video title |
| `transcription` | TEXT | Full transcription text |
| `language` | TEXT | Detected language code |
| `analysis_json` | TEXT | Complete analysis results (JSON) |
| `created_at` | TEXT | Record creation timestamp |
| `updated_at` | TEXT | Last update timestamp |
| `status` | TEXT | Processing status (complete, error, pending) |
| `error_message` | TEXT | Error message if status=error |

### Indices

- `idx_politician_name` on `politician_name`
- `idx_date` on `date`
- `idx_status` on `status`

### Querying the Database

```bash
# Connect to database
sqlite3 batch_results.db

# Get all complete analyses
SELECT politician_name, date, url, language
FROM videos
WHERE status = 'complete'
ORDER BY date DESC;

# Get politician video counts
SELECT politician_name, COUNT(*) as count
FROM videos
GROUP BY politician_name
ORDER BY count DESC;

# Get videos in date range
SELECT * FROM videos
WHERE date BETWEEN '2020-01-01' AND '2023-12-31';
```

## Export Formats

### JSON Export

Grouped by politician with summary statistics:

```json
{
  "total_videos": 10,
  "politicians": {
    "Benjamin Netanyahu": [
      {
        "url": "https://...",
        "date": "2023-01-01",
        "title": "Speech Title",
        "language": "he",
        "status": "complete",
        "overall_sentiment": "Neutral",
        "created_at": "2024-01-01T12:00:00"
      }
    ]
  },
  "generated_at": "2024-01-01T12:00:00"
}
```

### CSV Export

Enhanced format with sentiment columns:

```csv
date,politician,url,language,overall_sentiment,status,created_at
2023-01-01,Benjamin Netanyahu,https://...,he,Neutral,complete,2024-01-01T12:00:00
```

## Politician Name Extraction

The system automatically extracts politician names from video titles using pattern matching:

**Supported Patterns**:
- Hebrew names: נתניהו → Benjamin Netanyahu
- Hebrew names: בנט → Naftali Bennett
- Hebrew names: לפיד → Yair Lapid
- Prime Minister titles: "רה\"מ [name]" or "ראש הממשלה [name]"

If no known pattern matches, the name is set to "Unknown" and logged.

**Custom Politicians**: To add support for additional politicians, edit the `patterns` list in [`backend/csv_parser.py:extract_politician_name()`](backend/csv_parser.py).

## Workflow Examples

### Example 1: One-Time Batch Analysis

```bash
# 1. Prepare CSV with video list
cat > videos.csv << EOF
date,name,url
2023-01-01,Netanyahu Speech,https://www.youtube.com/watch?v=...
2023-06-15,Bennett Address,https://www.youtube.com/watch?v=...
EOF

# 2. Run analysis
python batch_analyze.py --input videos.csv --export-json results.json

# 3. View results
cat results.json
```

### Example 2: Incremental Processing

```bash
# Day 1: Process initial batch
python batch_analyze.py --input batch1.csv

# Day 2: Add more videos (skips existing)
python batch_analyze.py --input batch2.csv --skip-existing

# Generate updated report
python batch_analyze.py --report
```

### Example 3: Analysis by Politician

```bash
# Process all videos
python batch_analyze.py --input all_videos.csv

# Export Netanyahu speeches only
python batch_analyze.py --export-csv netanyahu.csv --politician "Benjamin Netanyahu"

# Export Bennett speeches only
python batch_analyze.py --export-json bennett.json --politician "Naftali Bennett"
```

### Example 4: API Integration

```python
import requests

# Start batch job
response = requests.post('http://localhost:8000/batch/analyze', json={
    'csv_data': open('videos.csv').read(),
    'skip_existing': True
})

results = response.json()
print(f"Processed: {results['successful']} successful, {results['failed']} failed")

# Query results
response = requests.get('http://localhost:8000/batch/politicians')
politicians = response.json()

for pol in politicians['politicians']:
    print(f"{pol['name']}: {pol['video_count']} videos")
```

## Troubleshooting

### Issue: "No valid videos found in CSV"

**Cause**: CSV format is incorrect or URLs are invalid.

**Solution**:
- Verify CSV has headers: `date,name,url`
- Check URLs are valid YouTube links
- Use `--dry-run` to preview parsed videos

### Issue: "Transcription error: API rate limit"

**Cause**: OpenAI API rate limit exceeded.

**Solution**:
- Wait a few minutes between batches
- Process smaller batches
- Add delay between videos (future feature)

### Issue: Videos skipped unexpectedly

**Cause**: Videos already exist in database.

**Solution**:
- Check database: `sqlite3 batch_results.db "SELECT * FROM videos;"`
- Use `--skip-existing=false` to reprocess
- Delete database to start fresh: `rm batch_results.db`

### Issue: Politician name not extracted

**Cause**: Video title doesn't match known patterns.

**Solution**:
- Check logs for "Could not extract politician name" warnings
- Add custom pattern to [`backend/csv_parser.py`](backend/csv_parser.py)
- Manually edit database: `UPDATE videos SET politician_name = 'Name' WHERE url = '...';`

### Issue: Firebase sync fails

**Cause**: Missing Firebase credentials or invalid user ID.

**Solution**:
- Verify `firebase_credentials.json` exists in `backend/`
- Check user ID is valid Firebase auth UID
- Ensure `OPENAI_API_KEY` is set in `.env`

## Performance Considerations

**Processing Time**:
- ~3-5 minutes per video (download + transcription + analysis)
- 20 videos ≈ 1-2 hours
- Cached videos skip instantly

**Resource Usage**:
- Audio files: Temporary, deleted after transcription
- SQLite database: ~1-5 MB per video (with full analysis JSON)
- OpenAI API: 1 Whisper call + 0-1 GPT call per video (for translation)

**Rate Limits**:
- OpenAI Whisper: Check your account tier limits
- YouTube: No known limits for reasonable usage
- Recommendation: Process batches of <50 videos at once

## Next Steps

- **Extend politician patterns**: Edit [`backend/csv_parser.py`](backend/csv_parser.py) to add more names
- **Schedule regular runs**: Use cron or Task Scheduler with CLI script
- **Build reporting dashboard**: Query SQLite database for custom analyses
- **Integrate with web app**: Use Firebase sync to view results in dashboard

## Support

For issues or questions:
- Check [main README](README.md) for general setup
- Review [backend documentation](CLAUDE.md) for architecture details
- File issues at GitHub repository
