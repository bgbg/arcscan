-- Schema V2: Normalized database schema for video analysis
-- This schema replaces the monolithic analysis_json blob with structured tables
-- Enables efficient querying, time-series analysis, and SQL-based analytics

-- ============================================================================
-- VIDEO METADATA TABLE
-- ============================================================================
-- Stores core video metadata and processing status
CREATE TABLE IF NOT EXISTS video_metadata (
    url TEXT PRIMARY KEY,
    date TEXT,                  -- Video date from CSV input
    person_name TEXT NOT NULL,  -- Person appearing in the video
    title TEXT,                 -- Video title (from YouTube metadata)
    created_at TEXT NOT NULL,   -- When record was first created
    updated_at TEXT NOT NULL,   -- When record was last updated
    status TEXT NOT NULL,       -- 'pending', 'processing', 'complete', 'error'
    error_message TEXT,         -- Error details if status='error'

    -- Backward compatibility: keep analysis_json during migration period
    analysis_json TEXT          -- Will be removed in future after validation
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_video_person_name ON video_metadata(person_name);
CREATE INDEX IF NOT EXISTS idx_video_date ON video_metadata(date);
CREATE INDEX IF NOT EXISTS idx_video_status ON video_metadata(status);
CREATE INDEX IF NOT EXISTS idx_video_person_date ON video_metadata(person_name, date);


-- ============================================================================
-- TRANSCRIPTIONS TABLE
-- ============================================================================
-- Stores transcription data with language detection metadata
-- Supports multiple transcriptions per video (e.g., different sources, corrections)
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_url TEXT NOT NULL,
    text TEXT NOT NULL,                     -- Final transcription text (translated if applicable)
    language TEXT,                          -- Final output language (usually 'en')
    source TEXT,                            -- 'whisper' or 'subtitles'
    detected_language TEXT,                 -- Original detected language (e.g., 'he', 'ar', 'en')
    subtitle_language TEXT,                 -- Subtitle language code if source='subtitles'
    decision_path TEXT,                     -- Processing path (e.g., 'subtitle_he_translated', 'whisper')
    created_at TEXT NOT NULL,

    FOREIGN KEY (video_url) REFERENCES video_metadata(url) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcription_video_url ON transcriptions(video_url);
CREATE INDEX IF NOT EXISTS idx_transcription_language ON transcriptions(language);
CREATE INDEX IF NOT EXISTS idx_transcription_source ON transcriptions(source);


-- ============================================================================
-- TRANSLATIONS TABLE
-- ============================================================================
-- Stores translation data when transcription was in non-English language
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcription_id INTEGER NOT NULL,
    original_text TEXT NOT NULL,            -- Original non-English text
    translated_text TEXT NOT NULL,          -- English translation
    model TEXT,                             -- Translation model used (e.g., 'gpt-3.5-turbo')
    created_at TEXT NOT NULL,

    FOREIGN KEY (transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_translation_transcription_id ON translations(transcription_id);


-- ============================================================================
-- SENTENCES TABLE
-- ============================================================================
-- Stores sentence-level data with timestamps and sentiment
-- Enables time-series analysis and timeline aggregation at any granularity
CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_url TEXT NOT NULL,
    sentence_index INTEGER NOT NULL,        -- Order of sentence in video (0-based)
    text TEXT NOT NULL,
    start_time REAL NOT NULL,               -- Start timestamp in seconds
    end_time REAL NOT NULL,                 -- End timestamp in seconds
    sentiment TEXT NOT NULL,                -- 'positive', 'neutral', 'negative'
    created_at TEXT NOT NULL,

    FOREIGN KEY (video_url) REFERENCES video_metadata(url) ON DELETE CASCADE,

    -- Ensure one sentence per index per video
    UNIQUE(video_url, sentence_index)
);

CREATE INDEX IF NOT EXISTS idx_sentence_video_url ON sentences(video_url);
CREATE INDEX IF NOT EXISTS idx_sentence_sentiment ON sentences(sentiment);
CREATE INDEX IF NOT EXISTS idx_sentence_start_time ON sentences(start_time);
CREATE INDEX IF NOT EXISTS idx_sentence_end_time ON sentences(end_time);
CREATE INDEX IF NOT EXISTS idx_sentence_time_range ON sentences(video_url, start_time, end_time);


-- ============================================================================
-- VIDEO SENTIMENTS TABLE
-- ============================================================================
-- Stores aggregated sentiment statistics per video
-- One row per video with overall sentiment distribution
CREATE TABLE IF NOT EXISTS video_sentiments (
    video_url TEXT PRIMARY KEY,
    overall_sentiment TEXT NOT NULL,        -- 'positive', 'neutral', 'negative'
    positive_count INTEGER NOT NULL DEFAULT 0,
    positive_pct REAL NOT NULL DEFAULT 0.0,
    neutral_count INTEGER NOT NULL DEFAULT 0,
    neutral_pct REAL NOT NULL DEFAULT 0.0,
    negative_count INTEGER NOT NULL DEFAULT 0,
    negative_pct REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,

    FOREIGN KEY (video_url) REFERENCES video_metadata(url) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_video_sentiment_overall ON video_sentiments(overall_sentiment);


-- ============================================================================
-- SENTENCE EMOTIONS TABLE
-- ============================================================================
-- Stores emotion analysis at sentence level (GoEmotions model with 28 emotions)
-- Each sentence can have multiple emotions with scores
CREATE TABLE IF NOT EXISTS sentence_emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id INTEGER NOT NULL,
    emotion_name TEXT NOT NULL,             -- e.g., 'joy', 'sadness', 'anger', 'fear', etc.
    score REAL NOT NULL,                    -- Emotion score (0.0-1.0)
    created_at TEXT NOT NULL,

    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,

    -- Ensure one score per emotion per sentence
    UNIQUE(sentence_id, emotion_name)
);

CREATE INDEX IF NOT EXISTS idx_sentence_emotion_sentence_id ON sentence_emotions(sentence_id);
CREATE INDEX IF NOT EXISTS idx_sentence_emotion_name ON sentence_emotions(emotion_name);
CREATE INDEX IF NOT EXISTS idx_sentence_emotion_score ON sentence_emotions(score);


-- ============================================================================
-- VIDEO EMOTION SUMMARY TABLE
-- ============================================================================
-- Stores aggregated emotion statistics per video
-- One row per video with average scores for each of the 28 emotions
-- Note: Emotion columns added dynamically based on GoEmotions taxonomy
CREATE TABLE IF NOT EXISTS video_emotion_summary (
    video_url TEXT PRIMARY KEY,

    -- Plutchik's basic emotions (GoEmotions subset)
    joy REAL DEFAULT 0.0,
    sadness REAL DEFAULT 0.0,
    anger REAL DEFAULT 0.0,
    fear REAL DEFAULT 0.0,
    surprise REAL DEFAULT 0.0,
    disgust REAL DEFAULT 0.0,

    -- Additional GoEmotions emotions
    admiration REAL DEFAULT 0.0,
    amusement REAL DEFAULT 0.0,
    annoyance REAL DEFAULT 0.0,
    approval REAL DEFAULT 0.0,
    caring REAL DEFAULT 0.0,
    confusion REAL DEFAULT 0.0,
    curiosity REAL DEFAULT 0.0,
    desire REAL DEFAULT 0.0,
    disappointment REAL DEFAULT 0.0,
    disapproval REAL DEFAULT 0.0,
    embarrassment REAL DEFAULT 0.0,
    excitement REAL DEFAULT 0.0,
    gratitude REAL DEFAULT 0.0,
    grief REAL DEFAULT 0.0,
    love REAL DEFAULT 0.0,
    nervousness REAL DEFAULT 0.0,
    optimism REAL DEFAULT 0.0,
    pride REAL DEFAULT 0.0,
    realization REAL DEFAULT 0.0,
    relief REAL DEFAULT 0.0,
    remorse REAL DEFAULT 0.0,

    -- Neutral emotion
    neutral REAL DEFAULT 0.0,

    created_at TEXT NOT NULL,

    FOREIGN KEY (video_url) REFERENCES video_metadata(url) ON DELETE CASCADE
);

-- Indexes for emotion-based queries
CREATE INDEX IF NOT EXISTS idx_video_emotion_joy ON video_emotion_summary(joy);
CREATE INDEX IF NOT EXISTS idx_video_emotion_sadness ON video_emotion_summary(sadness);
CREATE INDEX IF NOT EXISTS idx_video_emotion_anger ON video_emotion_summary(anger);


-- ============================================================================
-- MIGRATION HELPERS
-- ============================================================================

-- Backup table for safe migration (created by migration script)
-- CREATE TABLE IF NOT EXISTS videos_backup AS SELECT * FROM videos;

-- View to simulate old schema for backward compatibility during transition
CREATE VIEW IF NOT EXISTS videos_legacy AS
SELECT
    v.url,
    v.date,
    v.person_name,
    v.title,
    t.text as transcription,
    t.language,
    v.analysis_json,
    v.created_at,
    v.updated_at,
    v.status,
    v.error_message
FROM video_metadata v
LEFT JOIN transcriptions t ON v.url = t.video_url
WHERE t.id = (
    -- Get most recent transcription for each video
    SELECT MAX(id) FROM transcriptions WHERE video_url = v.url
);
