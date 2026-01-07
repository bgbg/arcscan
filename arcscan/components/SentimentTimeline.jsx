// components/SentimentTimeline.jsx
'use client'

import React, { useEffect, useRef } from 'react';

const SentimentTimeline = ({ timelineData, videoPlayer, currentTime }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !timelineData || timelineData.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Get the actual pixel dimensions of the canvas
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    
    // Set canvas resolution to match display size for crisp rendering
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
    
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, width, height);

    // Get the total video duration
    const totalDuration = timelineData[timelineData.length - 1].time;
    const secondsPerPixel = width / totalDuration;
    
    // Draw grid lines (vertical time markers)
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= totalDuration; i += 60) { // Every minute
      const x = i * secondsPerPixel;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      
      // Add time labels
      const minutes = Math.floor(i / 60);
      const seconds = i % 60;
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`${minutes}:${seconds.toString().padStart(2, '0')}`, x, height - 5);
    }

    // Find the current time index
    const currentTimeIndex = Math.min(
      Math.floor(currentTime),
      timelineData.length - 1
    );
    
    // Only process data up to the current time
    const visibleData = timelineData.slice(0, currentTimeIndex + 1);
    
    if (visibleData.length > 0) {
      // Process data to create a sequential sentiment chart
      // Each point in time will show the dominant sentiment with a block of color
      const segmentWidth = Math.max(2, secondsPerPixel); // Ensure at least 2px width for visibility
      
      for (let i = 0; i < visibleData.length; i++) {
        const point = visibleData[i];
        const x = point.time * secondsPerPixel;
        
        // Find the dominant sentiment
        let dominantSentiment = 'Neutral';
        let maxValue = point.Neutral;
        
        if (point.Positive > maxValue) {
          dominantSentiment = 'Positive';
          maxValue = point.Positive;
        }
        
        if (point.Negative > maxValue) {
          dominantSentiment = 'Negative';
          maxValue = point.Negative;
        }
        
        // Set color based on dominant sentiment
        let color;
        switch (dominantSentiment) {
          case 'Positive':
            color = 'rgba(16, 185, 129, 0.7)'; // Green
            break;
          case 'Negative':
            color = 'rgba(239, 68, 68, 0.7)'; // Red
            break;
          default:
            color = 'rgba(251, 191, 36, 0.7)'; // Yellow (Neutral)
            break;
        }
        
        // Calculate the intensity of the sentiment (0.3 to 1.0 range)
        const intensity = 0.3 + (maxValue * 0.7);
        
        // Draw the sentiment bar
        const barHeight = height * intensity;
        const y = height - barHeight;
        
        ctx.fillStyle = color;
        ctx.fillRect(x, y, segmentWidth, barHeight);
      }
      
      // Add a softer transition overlay
      ctx.globalCompositeOperation = 'source-atop';
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, 'rgba(255, 255, 255, 0.1)');
      gradient.addColorStop(1, 'rgba(255, 255, 255, 0.4)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      ctx.globalCompositeOperation = 'source-over';
    }

    // Draw current time marker
    if (currentTime !== undefined) {
      const timeX = currentTime * secondsPerPixel;
      
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)'; // Blue
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(timeX, 0);
      ctx.lineTo(timeX, height);
      ctx.stroke();
      
      // Add current time label
      const minutes = Math.floor(currentTime / 60);
      const seconds = Math.floor(currentTime % 60);
      ctx.fillStyle = '#3b82f6';
      ctx.font = 'bold 11px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`${minutes}:${seconds.toString().padStart(2, '0')}`, timeX, 15);
    }
    
    // Draw sentiment labels directly on the chart
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#10b981'; // Green
    ctx.fillText('Positive', 10, 20);
    ctx.fillStyle = '#ef4444'; // Red
    ctx.fillText('Negative', 10, 40);
    ctx.fillStyle = '#fbbf24'; // Yellow
    ctx.fillText('Neutral', 10, 60);
  }, [timelineData, currentTime]);

  // Add click handler to jump to specific time in the video
  const handleCanvasClick = (e) => {
    if (!videoPlayer || !timelineData || timelineData.length === 0) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    
    const totalDuration = timelineData[timelineData.length - 1].time;
    const secondsPerPixel = canvas.width / totalDuration;
    const clickedTime = x / secondsPerPixel;
    
    if (videoPlayer && typeof videoPlayer.seekTo === 'function') {
      videoPlayer.seekTo(clickedTime);
    }
  };

  return (
    <div className="w-full h-full relative">
      <canvas 
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="w-full h-full cursor-pointer rounded-lg"
      />
      <div className="text-xs text-gray-500 text-center mt-1">
        Click on the timeline to jump to that point in the video
      </div>
    </div>
  );
};

export default SentimentTimeline;