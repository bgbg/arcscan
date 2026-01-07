// components/AnalysisProgress.jsx
'use client'

import React, { useState, useEffect } from 'react';

const AnalysisProgress = ({ progress, partialResults }) => {
  // Set initial state based on props or default
  const [displayProgress, setDisplayProgress] = useState({ 
    status: progress?.status || "starting", 
    progress: progress?.progress || 5, 
    message: progress?.message || "Starting analysis..." 
  });
  
  // Track when the last update happened
  const [lastUpdateTime, setLastUpdateTime] = useState(Date.now());
  
  // Log and update when progress changes
  useEffect(() => {
    console.log("Progress prop received:", progress);
    
    // Only update if we have valid progress data and it's different from what we're showing
    if (progress && progress.status) {
      console.log("Updating displayed progress to:", progress);
      setDisplayProgress(progress);
      setLastUpdateTime(Date.now());
    }
  }, [progress]);
  
  const getStepEmoji = (status) => {
    switch (status) {
      case 'starting': return '🔍';
      case 'downloading': return '📥';
      case 'downloaded': return '✅';
      case 'transcribing': return '🔊';
      case 'transcribed': return '📝';
      case 'analyzing': return '🧠';
      case 'creating_timeline': return '📊';
      case 'summarizing': return '📊';
      case 'complete': return '✅';
      case 'error': return '❌';
      default: return '⏳';
    }
  };
  
  const getStepDescription = (status) => {
    switch (status) {
      case 'starting': return 'Starting analysis';
      case 'downloading': return 'Downloading audio from YouTube';
      case 'downloaded': return 'Audio downloaded successfully';
      case 'transcribing': return 'Converting speech to text';
      case 'transcribed': return 'Speech converted to text';
      case 'analyzing': return 'Analyzing emotional content';
      case 'creating_timeline': return 'Building sentiment timeline';
      case 'summarizing': return 'Creating emotional summary';
      case 'complete': return 'Analysis complete!';
      case 'error': return 'An error occurred';
      default: return 'Processing...';
    }
  };

  // Animated dots component
  const LoadingDots = () => {
    const [dots, setDots] = useState('.');
    
    useEffect(() => {
      const interval = setInterval(() => {
        setDots(prev => prev.length >= 3 ? '.' : prev + '.');
      }, 500);
      
      return () => clearInterval(interval);
    }, []);
    
    return <span className="ml-2">{dots}</span>;
  };
  
  // Define the order and steps
  const stepOrder = {
    'not_started': -1,
    'starting': 0,
    'downloading': 1,
    'downloaded': 2,
    'transcribing': 3,
    'transcribed': 4,
    'analyzing': 5,
    'creating_timeline': 6,
    'summarizing': 7,
    'complete': 8,
    'error': 999
  };

  // The steps we want to display in the UI
  const visibleSteps = ['downloading', 'transcribing', 'analyzing', 'creating_timeline', 'complete'];
  
  // Determine if a step is completed, current, or upcoming
  const getStepStatus = (step) => {
    const currentStepValue = stepOrder[displayProgress.status] || 0;
    const thisStepValue = stepOrder[step] || 0;
    
    // If we're on this step
    if (displayProgress.status === step) {
      return 'current';
    } 
    // If we're on complete, mark all as complete
    else if (displayProgress.status === 'complete') {
      return 'completed';
    }
    // If we're past this step
    else if (currentStepValue > thisStepValue) {
      return 'completed';
    } 
    // Otherwise, it's upcoming
    else {
      return 'upcoming';
    }
  };
  
  // Manually override the status for demonstration
  const status = displayProgress.status === 'not_started' ? 'starting' : displayProgress.status;
  
  return (
    <div className="bg-white rounded-xl shadow-md p-6 max-w-2xl mx-auto">
      <h3 className="text-xl font-semibold mb-4">Analyzing Your Video</h3>
      
      {/* Progress bar */}
      <div className="relative h-3 bg-gray-200 rounded-full mb-6 overflow-hidden">
        <div 
          className="absolute h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
          style={{ width: `${displayProgress.progress || 5}%` }}
        >
          {/* Animated shine effect */}
          <div className="absolute inset-0 w-full h-full">
            <div className="absolute inset-0 animate-pulse bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
          </div>
        </div>
      </div>
      
      {/* Current step */}
      <div className="flex items-center mb-6">
        <div className="text-3xl mr-4">{getStepEmoji(status)}</div>
        <div className="flex-1">
          <p className="font-medium text-lg flex items-center">
            {getStepDescription(status)}
            {status !== 'complete' && status !== 'error' && <LoadingDots />}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            {displayProgress.message || (status === 'error' 
              ? 'Please try again or choose a different video' 
              : status === 'complete'
                ? 'Your results are ready!'
                : 'Please wait while we process your video')}
          </p>
        </div>
        <div className="text-xl font-bold text-blue-500">
          {displayProgress.progress || 0}%
        </div>
      </div>
      
      {/* Steps list */}
      <div className="space-y-2">
        {visibleSteps.map(step => {
          // Force all steps to show as upcoming unless we have confirmed progress
          const shouldShowProgress = displayProgress.status !== 'not_started';
          const status = shouldShowProgress ? getStepStatus(step) : 'upcoming';
          
          return (
            <div key={step} className="flex items-center">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-3 ${
                status === 'current' ? 'bg-blue-500 text-white' :
                status === 'completed' ? 'bg-green-500 text-white' : 
                'bg-gray-200 text-gray-400'
              }`}>
                {status === 'completed' ? '✓' : 
                 status === 'current' ? '•' : ''}
              </div>
              <div className={`${
                status === 'current' ? 'text-blue-500 font-medium' :
                status === 'completed' ? 'text-green-500' : 
                'text-gray-400'
              }`}>
                {getStepDescription(step)}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Partial transcript */}
      {partialResults && partialResults.transcription && status !== 'complete' && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-100">
          <h4 className="font-medium text-gray-700 mb-2">Transcript Preview:</h4>
          <div className="text-sm text-gray-600 max-h-32 overflow-y-auto">
            {partialResults.transcription.substring(0, 300)}
            {partialResults.transcription.length > 300 && '...'}
          </div>
        </div>
      )}
      
      {/* Debug information - will help identify progress update issues 
      <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-100 text-xs text-gray-500">
        <details>
          <summary className="cursor-pointer">Debug Info</summary>
          <div className="mt-2">
            <p>Last updated: {new Date(lastUpdateTime).toLocaleTimeString()}</p>
            <p>Time since update: {((Date.now() - lastUpdateTime) / 1000).toFixed(1)} seconds</p>
            <pre className="whitespace-pre-wrap mt-2">
              Current Progress: {JSON.stringify(displayProgress, null, 2)}
            </pre>
          </div>
        </details>
      </div>
      */}
    </div>
  );
  
};

export default AnalysisProgress;