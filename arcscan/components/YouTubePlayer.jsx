// components/YouTubePlayer.jsx
'use client'

import React, { useEffect, useRef } from 'react';

const YouTubePlayer = ({ videoUrl, onTimeUpdate, onReady }) => {
  const playerRef = useRef(null);
  const playerInstanceRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    // Load YouTube iframe API script if not already loaded
    if (!window.YT) {
      const tag = document.createElement('script');
      tag.src = 'https://www.youtube.com/iframe_api';
      
      const firstScriptTag = document.getElementsByTagName('script')[0];
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
    }

    // Initialize player when API is ready
    const initPlayer = () => {
      if (!playerRef.current) return;
      
      const videoId = extractVideoId(videoUrl);
      if (!videoId) return;
      
      playerInstanceRef.current = new window.YT.Player(playerRef.current, {
        height: '360',
        width: '100%',
        videoId: videoId,
        playerVars: {
          autoplay: 0,
          controls: 1,
          rel: 0,
          modestbranding: 1
        },
        events: {
          onReady: handlePlayerReady,
          onStateChange: handlePlayerStateChange
        }
      });
    };

    // Setup YouTube API callback
    if (window.YT && window.YT.Player) {
      initPlayer();
    } else {
      // Define callback for when API loads
      window.onYouTubeIframeAPIReady = initPlayer;
    }

    // Clean up
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      
      window.onYouTubeIframeAPIReady = null;
      
      if (playerInstanceRef.current) {
        playerInstanceRef.current.destroy();
      }
    };
  }, [videoUrl]);

  const handlePlayerReady = (event) => {
    if (onReady) {
      onReady(event.target);
    }
  };

  const handlePlayerStateChange = (event) => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // If playing, start interval to track time
    if (event.data === window.YT.PlayerState.PLAYING) {
      intervalRef.current = setInterval(() => {
        if (playerInstanceRef.current && onTimeUpdate) {
          const currentTime = playerInstanceRef.current.getCurrentTime();
          onTimeUpdate(currentTime);
        }
      }, 100); // Update every 100ms for smoother updates
    }
  };

  // Extract YouTube video ID from URL
  const extractVideoId = (url) => {
    if (!url) return null;
    
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[7].length === 11) ? match[7] : null;
  };

  return (
    <div className="youtube-player-container w-full aspect-video bg-black rounded-lg overflow-hidden">
      <div ref={playerRef} id="youtube-player" className="w-full h-full"></div>
    </div>
  );
};

export default YouTubePlayer;