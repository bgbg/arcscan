// components/FirebaseProgressTracker.jsx
import React, { useEffect } from 'react';
import { doc, onSnapshot } from 'firebase/firestore';
import { db } from '@/lib/firebaseConfig';

const FirebaseProgressTracker = ({ videoUrl, userId, onProgressUpdate, onComplete }) => {
  // Generate the same document ID that your backend uses
  const generateDocId = (url) => {
    return url.replace(/\W+/g, '_');
  };
  
  useEffect(() => {
    if (!videoUrl) return;
    
    const docId = generateDocId(videoUrl);
    console.log("Listening for progress updates on doc:", docId);
    
    // Set up a real-time listener on the progress document
    const unsubscribe = onSnapshot(
      doc(db, "analysis_progress", docId),
      (docSnapshot) => {
        if (docSnapshot.exists()) {
          const data = docSnapshot.data();
          console.log("Real-time progress update:", data);
          
          // Notify parent component
          if (onProgressUpdate) {
            onProgressUpdate(data);
          }
          
          // Check for completion
          if (data.status === 'complete' && onComplete) {
            onComplete(videoUrl);
          }
        } else {
          console.log("No progress document found yet");
        }
      },
      (err) => {
        console.error("Error listening to progress updates:", err);
      }
    );
    
    // Cleanup listener on unmount
    return () => unsubscribe();
  }, [videoUrl, userId, onProgressUpdate, onComplete]);

  // This component doesn't render anything visible
  return null;
};

export default FirebaseProgressTracker;