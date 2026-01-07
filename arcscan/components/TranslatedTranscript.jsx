import React, { useState } from 'react';

const TranslatedTranscript = ({ 
  sentenceResults, 
  activeSentence, 
  youtubePlayerRef,
  originalText,
  translatedText,
  detectedLanguage
}) => {
  const [showOriginal, setShowOriginal] = useState(false);
  
  // Only show translation toggle if we have both versions
  const hasTranslation = originalText && translatedText && originalText !== translatedText;
  
  const getSentimentColor = (sentiment, opacity = 0.2) => {
    const colors = {
      'Positive': `rgba(16, 185, 129, ${opacity})`,
      'Negative': `rgba(239, 68, 68, ${opacity})`,
      'Neutral': `rgba(251, 191, 36, ${opacity})`
    };
    return colors[sentiment] || colors['Neutral'];
  };
  
  const getLanguageName = (code) => {
    if (code === 'he') return 'Hebrew';
    if (code === 'ar') return 'Arabic';
    return code;
  };

  return (
    <div className="bg-white rounded-xl shadow-md p-4">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-xl font-semibold">📝 Transcript</h2>
        
        {hasTranslation && (
          <div className="flex items-center gap-2">
            <div className="text-sm text-gray-500 flex items-center">
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-md mr-2">
                {getLanguageName(detectedLanguage)} detected
              </span>
              <label className="inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={showOriginal}
                  onChange={() => setShowOriginal(!showOriginal)}
                  className="sr-only peer"
                />
                <div className="relative w-10 h-5 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-500"></div>
                <span className="ml-2 text-sm font-medium">
                  {showOriginal ? 'Original' : 'Translated'}
                </span>
              </label>
            </div>
          </div>
        )}
      </div>
      
      <div className="max-h-[calc(100vh-24rem)] overflow-y-auto text-sm">
        {hasTranslation && showOriginal ? (
          // Show original text (Hebrew/Arabic)
          <p className="whitespace-pre-wrap text-right" dir="rtl">{originalText}</p>
        ) : (
          // Show translated or original text with sentiment highlighting
          sentenceResults.map((sentence, index) => (
            <span
              key={index}
              className={`inline transition-colors duration-200 cursor-pointer hover:bg-opacity-40 ${
                activeSentence && activeSentence.index === sentence.index 
                  ? 'bg-blue-100 rounded' 
                  : ''
              }`}
              style={{ 
                backgroundColor: activeSentence && activeSentence.index === sentence.index 
                  ? 'rgba(59, 130, 246, 0.2)' 
                  : getSentimentColor(sentence.final_sentiment)
              }}
              onClick={() => {
                if (youtubePlayerRef.current) {
                  youtubePlayerRef.current.seekTo(sentence.start_time);
                }
              }}
            >
              {sentence.text}{' '}
            </span>
          ))
        )}
      </div>
      
      <div className="mt-2 text-xs text-gray-500">
        {hasTranslation && showOriginal ? 
          "Original text shown above" : 
          "Click on any sentence to jump to that point in the video"
        }
      </div>
    </div>
  );
};

export default TranslatedTranscript;