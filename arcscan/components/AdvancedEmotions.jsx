// components/AdvancedEmotions.jsx
'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

// Color palette for emotions - optimized for distinguishability
// Based on iwanthue principles - maximizing perceptual distances between colors
const EMOTION_COLORS = {
  // Primary emotions with highly distinguishable colors
  joy: '#ffd700',            // Bright yellow - universally recognized for joy
  sadness: '#3c78d8',        // Clear blue - commonly associated with sadness
  anger: '#cc0000',          // True red - universally recognized for anger
  fear: '#8000b0',           // Deep purple - commonly associated with fear
  surprise: '#00c6ff',       // Bright cyan - high contrast and visibility
  disgust: '#61a500',        // Olive green - distinctly different from others
  trust: '#ff9966',          // Coral - warm and distinctive
  anticipation: '#ff3399',   // Pink - bright and distinctive
  
  // Secondary emotions with maximized perceptual distance
  admiration: '#9966ff',     // Lavender purple
  approval: '#009e73',       // Distinct teal
  neutral: '#6baed6',        // Light blue
  optimism: '#ff9900',       // Orange - visually distinct
  confusion: '#9c9c9c',      // Mid-gray
  gratitude: '#d462ff',      // Bright magenta
  caring: '#ff6b6b',         // Salmon red
  realization: '#5954c9',    // Indigo
  annoyance: '#e6394a',      // Raspberry
  disappointment: '#9f5000', // Brown
  disapproval: '#1a9850',    // Forest green 
  grief: '#466a8f',          // Steel blue
  love: '#ff0099',           // Hot pink
  nervousness: '#e69900',    // Amber
  relief: '#45b5aa',         // Turquoise
  curiosity: '#00c0b0',      // Aqua
  excitement: '#ff6000',     // Bright orange
  pride: '#a53d13',          // Rust
  amusement: '#ff70cf'       // Light pink
}

export default function AdvancedEmotions({ videoUrl, userId }) {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [advancedData, setAdvancedData] = useState(null)
  const [activeTab, setActiveTab] = useState('timeline')

  const fetchAdvancedAnalysis = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Start the analysis
      const response = await fetch('/api/analyze/advanced-emotions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          url: videoUrl, 
          user_id: userId,
          translated: true // Flag to tell backend to use translated text if available
        }),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch advanced analysis')
      }
      
      // If we get an immediate response, use it
      const data = await response.json()
      
      // Check if analysis is complete or if we need to poll for progress
      if (data.status === 'complete' || data.emotion_timeline) {
        setAdvancedData(data)
        setLoading(false)
      } else {
        // Start polling for progress
        pollForProgress()
      }
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }
  
  // Poll for analysis progress
  const pollForProgress = async () => {
    const encodedUrl = encodeURIComponent(videoUrl)
    const pollInterval = setInterval(async () => {
      try {
        const progressRes = await fetch(`/api/progress/advanced/${encodedUrl}`)
        const progressData = await progressRes.json()
        
        setProgress(progressData.progress || 0)
        
        // If complete, fetch the results and stop polling
        if (progressData.status === 'complete' || progressData.status === 'complete_advanced') {
          clearInterval(pollInterval)
          
          // Fetch the completed analysis
          const resultsRes = await fetch(`/api/results/advanced/${encodedUrl}`)
          if (resultsRes.ok) {
            const resultsData = await resultsRes.json()
            setAdvancedData(resultsData)
            setLoading(false)
          } else {
            throw new Error('Failed to retrieve analysis results')
          }
        } else if (progressData.status === 'error' || progressData.status === 'error_advanced') {
          clearInterval(pollInterval)
          throw new Error(progressData.message || 'Analysis failed')
        }
      } catch (err) {
        clearInterval(pollInterval)
        setError(err.message)
        setLoading(false)
      }
    }, 2000) // Poll every 2 seconds
    
    // Clean up interval on component unmount
    return () => clearInterval(pollInterval)
  }
  
  // Format timestamp (seconds) to MM:SS format
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }
  
  // Prepare data for pie chart
  const preparePieData = () => {
    if (!advancedData?.emotion_summary) return []
    
    return Object.entries(advancedData.emotion_summary)
      .sort((a, b) => b[1].occurrences - a[1].occurrences) // Sort by occurrences
      .map(([emotion, data]) => ({
        name: emotion,
        value: data.average_score,
        count: data.occurrences
      }))
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 my-8">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold mb-2">Advanced Emotion Analysis</h3>
        <p className="text-gray-600">
          Discover the complex emotional journey in your video content
        </p>
        
        {!advancedData && !loading && (
          <button
            onClick={fetchAdvancedAnalysis}
            className="mt-4 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-500 text-white rounded-full font-semibold shadow-md transition hover:shadow-lg hover:scale-105"
          >
            Show Complex Emotions
          </button>
        )}
        
        {loading && (
          <div className="mt-6">
            <div className="relative pt-1">
              <div className="flex mb-2 items-center justify-between">
                <div>
                  <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-purple-600 bg-purple-200">
                    Analyzing
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-xs font-semibold inline-block text-purple-600">
                    {progress}%
                  </span>
                </div>
              </div>
              <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-purple-200">
                <div 
                  style={{ width: `${progress}%` }}
                  className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-purple-500 transition-all duration-500"
                ></div>
              </div>
            </div>
          </div>
        )}
        
        {error && (
          <div className="bg-red-50 text-red-500 p-4 rounded-lg mt-4">
            {error}
          </div>
        )}
      </div>
      
      {advancedData && (
        <>
          {/* Tab Navigation */}
          <div className="flex justify-center mb-6 border-b">
            <button
              className={`px-4 py-2 ${activeTab === 'timeline' ? 'border-b-2 border-purple-500 text-purple-500' : 'text-gray-500'}`}
              onClick={() => setActiveTab('timeline')}
            >
              Emotion Timeline
            </button>
            <button
              className={`px-4 py-2 ${activeTab === 'summary' ? 'border-b-2 border-purple-500 text-purple-500' : 'text-gray-500'}`}
              onClick={() => setActiveTab('summary')}
            >
              Emotion Summary
            </button>
            <button
              className={`px-4 py-2 ${activeTab === 'transcript' ? 'border-b-2 border-purple-500 text-purple-500' : 'text-gray-500'}`}
              onClick={() => setActiveTab('transcript')}
            >
              Emotional Transcript
            </button>
          </div>
          
          {/* Timeline View - UPDATED WITH OPTIMIZED COLORS */}
          {activeTab === 'timeline' && (
            <>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={advancedData.emotion_timeline}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      tickFormatter={formatTime}
                      label={{ value: 'Time (MM:SS)', position: 'insideBottom', offset: -5 }}
                      // Reduce number of ticks - show approximately 8-10 ticks
                      interval={Math.ceil(advancedData.emotion_timeline.length / 10)}
                    />
                    <YAxis 
                      label={{ value: 'Intensity', angle: -90, position: 'insideLeft' }}
                      domain={[0, 1]}
                      ticks={[0, 0.25, 0.5, 0.75, 1]} 
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                    <Tooltip
                      formatter={(value, name) => [
                        `${(value * 100).toFixed(1)}%`, 
                        name.charAt(0).toUpperCase() + name.slice(1)
                      ]}
                      labelFormatter={(value) => `Time: ${formatTime(value)}`}
                      // Custom content for tooltip to sort and show only top 2 emotions
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          // Sort payload by value (highest first)
                          const sortedPayload = [...payload].sort((a, b) => b.value - a.value);
                          
                          // Take only top 2 emotions with non-zero values
                          const topEmotions = sortedPayload
                            .filter(entry => entry.value > 0.01)
                            .slice(0, 2);
                          
                          return (
                            <div className="bg-white p-2 border border-gray-200 rounded shadow-sm">
                              <p className="font-medium text-gray-800">Time: {formatTime(label)}</p>
                              <div className="mt-1">
                                {topEmotions.map((entry, index) => (
                                  <div key={index} className="flex items-center gap-2 text-sm">
                                    <span
                                      className="w-3 h-3 rounded-full"
                                      style={{ backgroundColor: entry.stroke }}
                                    ></span>
                                    <span className="font-medium">{entry.name}</span>
                                    <span>{(entry.value * 100).toFixed(1)}%</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Legend />
                    {/* Dynamically render only detected emotions with data - with null check fix */}
                    {Object.keys(advancedData.emotion_timeline[0] || {})
                      .filter(key => key !== 'time')
                      .map(emotion => {
                        const hasData = advancedData.emotion_timeline.some(point => point[emotion] > 0.01);
                        if (!hasData) return null;
                        
                        // Calculate max value for this emotion with null/undefined check
                        const maxValue = Math.max(...advancedData.emotion_timeline.map(point => 
                          (point[emotion] !== undefined && point[emotion] !== null) ? point[emotion] : 0
                        ));
                        return { emotion, maxValue };
                      })
                      .filter(item => item !== null) // Filter out any null items
                      .sort((a, b) => b.maxValue - a.maxValue) // Sort by max value
                      .slice(0, 10) // Take only top 10 emotions to avoid visual clutter
                      .map(({ emotion }) => {
                        const color = EMOTION_COLORS[emotion] || '#999999';

                        return (
                          <Line
                            key={emotion}
                            type="monotone"
                            dataKey={emotion}
                            stroke={color}
                            activeDot={{ r: 8 }}
                            strokeWidth={2}
                            name={emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                          />
                        );
                      })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              {/* Additional chart - Simplified to ensure lines are visible */}
              <div className="mt-8 border-t pt-6">
                <h3 className="text-lg font-semibold mb-4">Simpler Emotion Distribution Over Time</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart 
                      data={advancedData.emotion_timeline}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="time" 
                        tickFormatter={formatTime}
                        interval={Math.ceil(advancedData.emotion_timeline.length / 8)}
                      />
                      <YAxis 
                        domain={[0, 1]}
                        ticks={[0, 0.25, 0.5, 0.75, 1]} 
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      />
                      <Tooltip 
                        formatter={(value, name) => [`${(value * 100).toFixed(1)}%`, name]}
                        labelFormatter={(value) => `Time: ${formatTime(value)}`}
                      />
                      <Legend />
                      
                      {/* Direct rendering of top 5 emotions without complex calculations */}
                      {Object.keys(advancedData.emotion_timeline[0] || {})
                        .filter(key => key !== 'time')
                        .map(emotion => {
                          // Simple check if this emotion has any significant data
                          const hasData = advancedData.emotion_timeline.some(point => 
                            point[emotion] !== undefined && point[emotion] > 0.05
                          );
                          
                          if (!hasData) return null;
                          
                          return {
                            emotion,
                            color: EMOTION_COLORS[emotion] || '#999999'
                          };
                        })
                        .filter(item => item !== null)
                        .slice(0, 5) // Take top 5
                        .map(({ emotion, color }) => (
                          <Line
                            key={emotion}
                            type="monotone"
                            dataKey={emotion}
                            stroke={color}
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 6 }}
                            name={emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                          />
                        ))
                      }
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-sm text-gray-500 mt-2">
                  This chart shows the distribution of the top 5 emotions over time, helping identify when different emotions occur simultaneously.
                </p>
              </div>
            </>
          )}
          
          {/* Summary View */}
          {activeTab === 'summary' && (
            <div className="flex flex-col md:flex-row items-center">
              {/* Pie Chart */}
              <div className="w-full md:w-1/2 h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={preparePieData()}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      nameKey="name"
                      label={({name, percent}) => `${name} (${(percent * 100).toFixed(0)}%)`}
                    >
                      {preparePieData().map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={EMOTION_COLORS[entry.name] || '#CCCCCC'} 
                        />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value) => `${value.toFixed(1)}%`}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              {/* Text Summary */}
              <div className="w-full md:w-1/2 p-4">
                <h4 className="text-xl font-bold mb-4">Emotional Profile</h4>
                <div className="space-y-3">
                  {Object.entries(advancedData.emotion_summary || {})
                    .sort((a, b) => b[1].occurrences - a[1].occurrences) // Sort by occurrences instead of score
                    .map(([emotion, data]) => (
                      <div key={emotion} className="flex items-center">
                        <div 
                          className="w-4 h-4 rounded-full mr-2" 
                          style={{ backgroundColor: EMOTION_COLORS[emotion] || '#CCCCCC' }}
                        ></div>
                        <div className="flex-1">
                          <div className="flex justify-between">
                            <span className="font-medium capitalize">{emotion}</span>
                            <span>{data.average_score}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                            <div 
                              className="h-2 rounded-full" 
                              style={{ 
                                width: `${data.average_score}%`, 
                                backgroundColor: EMOTION_COLORS[emotion] || '#CCCCCC' 
                              }}
                            ></div>
                          </div>
                          <div className="text-sm text-gray-500 mt-1">
                            Detected {data.occurrences} {data.occurrences === 1 ? 'time' : 'times'}
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          )}
          
          {/* Transcript View */}
          {activeTab === 'transcript' && (
            <div className="max-h-96 overflow-y-auto">
              {(advancedData.sentence_emotions || []).map((item, index) => (
                <div key={index} className="mb-4 p-3 rounded-lg border border-gray-200 hover:bg-gray-50">
                  <div className="flex justify-between text-sm text-gray-500 mb-1">
                    <span>{formatTime(item.start_time)} - {formatTime(item.end_time)}</span>
                  </div>
                  <p className="mb-2">{item.text}</p>
                  <div className="flex flex-wrap gap-2">
                    {(item.emotions || []).map((emotion, idx) => (
                      <span
                        key={idx}
                        className="text-xs px-2 py-1 rounded-full text-white"
                        style={{ 
                          backgroundColor: EMOTION_COLORS[emotion.emotion] || '#888',
                          opacity: Math.max(0.5, emotion.score / 100)  // Ensure minimum opacity for readability
                        }}
                      >
                        {emotion.emotion}: {emotion.score}%
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}