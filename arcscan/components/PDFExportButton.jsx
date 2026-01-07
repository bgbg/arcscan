import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Download, Loader2 } from 'lucide-react';
import { jsPDF } from 'jspdf'; // هذا هو الصحيح
import { noto_hebrew } from '@/app/fonts/noto_hebrew-normal';




const PDFExportButton = ({ 
  videoUrl, 
  sentimentSummary, 
  timelineData, 
  sentenceResults, 
  advancedData, 
  originalText, 
  translatedText, 
  detectedLanguage 
}) => {
  // Allow directly passing advanced data or fetch it when generating PDF
  const [fetchedAdvancedData, setFetchedAdvancedData] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Try to fetch advanced emotions data if not already provided
  const fetchAdvancedData = async () => {
    try {
      console.log("Attempting to fetch advanced emotions data for:", videoUrl);
const cleanUrl = new URL(videoUrl);
cleanUrl.searchParams.delete('t'); // Removes &t=54s or similar
const encodedUrl = encodeURIComponent(cleanUrl.toString());      const response = await fetch(`/api/results/advanced/${encodedUrl}`);
      
      if (response.ok) {
        const data = await response.json();
        console.log("Successfully fetched advanced data:", data);
        return data;
      } else {
        console.log("API returned non-ok response:", response.status);
        
        // If the API fails, try to fetch directly from Firestore
        // You may need to import the Firestore setup
        try {
          // This assumes you have imported { db } from '@/lib/firebaseConfig'
          // and have the necessary imports from 'firebase/firestore'
          const { doc, getDoc } = await import('firebase/firestore');
          const { db } = await import('@/lib/firebaseConfig');
          
          // Generate the same document ID used in your backend
const cleanUrl = new URL(videoUrl);
cleanUrl.searchParams.delete('t');
const docId = cleanUrl.toString().replace(/\W+/g, '_');
          console.log("Trying to fetch from Firestore with ID:", docId);
          
          const docRef = doc(db, "advanced_analyses", docId);
          const docSnap = await getDoc(docRef);
          
          if (docSnap.exists()) {
            console.log("Found data in Firestore:", docSnap.data());
            return docSnap.data();
          } else {
            console.log("No advanced emotions data found in Firestore");
          }
        } catch (firestoreError) {
          console.log("Error fetching from Firestore:", firestoreError);
        }
      }
    } catch (error) {
      console.log("Could not fetch advanced emotions data:", error);
    }
    return null;
  };

  const generatePDF = async () => {
    if (isGenerating) return;
    setIsGenerating(true);

    try {
      // Try to fetch advanced emotions data if not provided
      let emotionData = advancedData;
      console.log("Initial emotion data:", emotionData);
      
      if (!emotionData || !emotionData.emotion_summary) {
        console.log("No emotion data provided directly, attempting to fetch it");
        emotionData = await fetchAdvancedData();
        console.log("Fetched emotion data:", emotionData);
        setFetchedAdvancedData(emotionData);
      }

      // Create a new PDF document
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      // Register the font into jsPDF
pdf.addFileToVFS('noto_hebrew.ttf', noto_hebrew);
pdf.addFont('noto_hebrew.ttf', 'noto_hebrew', 'normal');
pdf.setFont('noto_hebrew');

      // Set up variables for positioning content
      const pageWidth = pdf.internal.pageSize.getWidth();
      const margin = 15;
      const contentWidth = pageWidth - (margin * 2);
      let yPosition = margin;

      // Define brand colors (replace with your actual brand colors)
      const brandPrimaryColor = [130, 45, 178]; // Purple
      const brandSecondaryColor = [180, 45, 178]; // Magenta
      const brandTextColor = [50, 50, 60]; // Dark gray

      // Add logo - if you have one
      /*
      const logoUrl = '/images/your-company-logo.png'; // Update with your logo path
      try {
        pdf.addImage(logoUrl, 'PNG', margin, yPosition, 40, 15);
        yPosition += 20;
      } catch (err) {
        console.log("Could not add logo:", err);
        yPosition = margin;
      }
      */

      // Add title and video URL
      pdf.setFontSize(18);
      pdf.setTextColor(...brandPrimaryColor);
      pdf.text("Video Analysis Report", margin, yPosition);
      yPosition += 10;

      // Add video URL
      pdf.setFontSize(10);
      pdf.setTextColor(...brandSecondaryColor);
      const videoTitle = videoUrl.includes('youtube.com') || videoUrl.includes('youtu.be') ? 
        'YouTube Video' : 'Video URL';
      pdf.text(`Source: ${videoTitle}`, margin, yPosition);
      yPosition += 6;
      
      // Wrap long URLs
      const wrappedUrl = pdf.splitTextToSize(videoUrl, contentWidth);
      pdf.text(wrappedUrl, margin, yPosition);
      yPosition += wrappedUrl.length * 5 + 3;

      // Add QR code
      const qrCodeUrl = `https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=${encodeURIComponent(videoUrl)}`;
      try {
        pdf.addImage(qrCodeUrl, 'PNG', pageWidth - 50, 30, 30, 30);
      } catch (err) {
        console.log("Could not add QR code:", err);
      }

      // Add date
      pdf.setTextColor(...brandSecondaryColor);
      const today = new Date();
      pdf.text(`Generated: ${today.toLocaleDateString()} ${today.toLocaleTimeString()}`, margin, yPosition);
      yPosition += 15;

      // Add sentiment summary section
      pdf.setFontSize(14);
      pdf.setTextColor(...brandPrimaryColor);
      pdf.text("Sentiment Summary", margin, yPosition);
      yPosition += 8;

      // Add sentiment summary data
      pdf.setFontSize(10);
      pdf.setTextColor(...brandTextColor);

      if (sentimentSummary) {
        Object.entries(sentimentSummary)
          .sort((a, b) => b[1] - a[1])
          .forEach(([sentiment, value]) => {
            pdf.text(`${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}: ${value}%`, margin, yPosition);
            yPosition += 5;
          });
      }
      yPosition += 10;

      // Check if we have advanced emotions data
      if (emotionData && emotionData.emotion_summary) {
        console.log("Adding advanced emotions to PDF");
        // Add advanced emotions section
        pdf.setFontSize(14);
        pdf.setTextColor(...brandPrimaryColor);
        pdf.text("Advanced Emotions Analysis", margin, yPosition);
        yPosition += 8;

        // Add top emotions
        pdf.setFontSize(10);
        pdf.setTextColor(...brandTextColor);
        
        // Sort emotions by average score (highest first)
        const sortedEmotions = Object.entries(emotionData.emotion_summary)
          .sort((a, b) => {
            // Handle different data structures
            const scoreA = typeof a[1] === 'object' ? a[1].average_score : a[1];
            const scoreB = typeof b[1] === 'object' ? b[1].average_score : b[1];
            return scoreB - scoreA;
          });
        
        console.log("Sorted emotions:", sortedEmotions);
        
        // Display the emotions
        sortedEmotions.slice(0, 8).forEach(([emotion, data]) => {
          // Handle different data structures
          let score, occurrences;
          if (typeof data === 'object') {
            score = data.average_score;
            occurrences = data.occurrences;
          } else {
            score = data;
            occurrences = 1; // Default if not provided
          }
          
          pdf.text(`${emotion.charAt(0).toUpperCase() + emotion.slice(1)}: ${score}% (detected ${occurrences} times)`, margin, yPosition);
          yPosition += 5;
        });
        
        // If there are emotional sentences, add a section for them
        if (emotionData.sentence_emotions && emotionData.sentence_emotions.length > 0) {
          console.log("Adding emotional sentences to PDF");
          yPosition += 8;
          pdf.setFontSize(12);
          pdf.setTextColor(...brandPrimaryColor);
          pdf.text("Key Emotional Moments:", margin, yPosition);
          yPosition += 6;
          
          // Format timestamp (seconds) to MM:SS format
          const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
          };
          
          // Get top emotional sentences (limit to 5 for space)
          const topEmotionalSentences = emotionData.sentence_emotions
            .slice(0, 5)
            .map(sentence => {
              // Get the top emotion for this sentence
              const topEmotion = sentence.emotions && sentence.emotions.length > 0 
                ? sentence.emotions[0] 
                : null;
                
              return {
                text: sentence.text,
                time: formatTime(sentence.start_time),
                emotion: topEmotion ? `${topEmotion.emotion} (${topEmotion.score}%)` : 'neutral'
              };
            });
            
          // Add them to the PDF
          pdf.setFontSize(9);
          topEmotionalSentences.forEach(sentence => {
            // Add timestamp and emotion
            pdf.setTextColor(...brandSecondaryColor);
            pdf.text(`[${sentence.time}] - ${sentence.emotion}`, margin, yPosition);
            yPosition += 4;
            
            // Add the sentence text (with word wrapping)
            pdf.setTextColor(...brandTextColor);
            const textLines = pdf.splitTextToSize(sentence.text, contentWidth - 5);
            
            // Check if we need a new page
            if (yPosition + (textLines.length * 4) > pdf.internal.pageSize.getHeight() - margin) {
              pdf.addPage();
              yPosition = margin;
            }
            
            pdf.text(textLines, margin + 5, yPosition);
            yPosition += textLines.length * 4 + 4;
          });
        }
        
        yPosition += 5;
      }

      // Add transcript section with translation support
      if (sentenceResults && sentenceResults.length > 0) {
        // Check if we need a new page
        if (yPosition > pdf.internal.pageSize.getHeight() - 60) {
          pdf.addPage();
          yPosition = margin;
        }

        pdf.setFontSize(14);
        pdf.setTextColor(...brandPrimaryColor);
        pdf.text("Emotional Transcript", margin, yPosition);
        yPosition += 8;

        // Check if there's a translation
        const hasTranslation = originalText && translatedText && originalText !== translatedText;
        if (hasTranslation) {
          const languageName = detectedLanguage === 'he' ? 'Hebrew' : 
                              detectedLanguage === 'ar' ? 'Arabic' : 
                              detectedLanguage;
          pdf.setFontSize(9);
          pdf.setTextColor(...brandSecondaryColor);
          pdf.text(`Original language: ${languageName} (translated to English)`, margin, yPosition);
          yPosition += 5;

          // Add the original text section
          pdf.setFontSize(10);
          pdf.setTextColor(...brandPrimaryColor);
          pdf.text("Original Text:", margin, yPosition);
          yPosition += 6;
          
          // Format original text with word wrapping
          pdf.setFontSize(9);
          pdf.setTextColor(...brandTextColor);
          const originalTextLines = pdf.splitTextToSize(originalText, contentWidth);
          
          // Check if we need a new page
          if (yPosition + (originalTextLines.length * 4) > pdf.internal.pageSize.getHeight() - margin) {
            pdf.addPage();
            yPosition = margin;
          }
          
          // If the text is in Hebrew or Arabic, align it to the right
         if (detectedLanguage === 'he' || detectedLanguage === 'ar') {
  const rightMargin = pageWidth - margin;
  const reverseLines = (lines) => lines.map(line => line.split('').reverse().join(''));
  const reversedTextLines = reverseLines(originalTextLines);
  pdf.setFont('noto_hebrew');
  pdf.setFontSize(10);
  pdf.text(reversedTextLines, rightMargin, yPosition, { align: 'right' });
}

          
          yPosition += originalTextLines.length * 4 + 10;
          
          // Check if we need a new page for the translated text
          if (yPosition > pdf.internal.pageSize.getHeight() - 40) {
            pdf.addPage();
            yPosition = margin;
          }
          
          // Add the translated text section
          pdf.setFontSize(10);
          pdf.setTextColor(...brandPrimaryColor);
          pdf.text("Translated Text:", margin, yPosition);
          yPosition += 6;
        }

        // Add transcript text with timestamps
        pdf.setFontSize(9);
        pdf.setTextColor(...brandTextColor);

        // Format timestamp (seconds) to MM:SS format
        const formatTime = (seconds) => {
          const mins = Math.floor(seconds / 60);
          const secs = Math.floor(seconds % 60);
          return `${mins}:${secs.toString().padStart(2, '0')}`;
        };

        // For translated content, use the translated text directly
        if (hasTranslation && translatedText) {
          const translatedTextLines = pdf.splitTextToSize(translatedText, contentWidth);
          
          // Check if we need a new page
          if (yPosition + (translatedTextLines.length * 4) > pdf.internal.pageSize.getHeight() - margin) {
            pdf.addPage();
            yPosition = margin;
          }
          
          pdf.text(translatedTextLines, margin, yPosition);
          yPosition += translatedTextLines.length * 4 + 8;
        } else {
          // Group transcript into paragraphs for better readability
          const paragraphSize = 4;
          for (let i = 0; i < sentenceResults.length; i += paragraphSize) {
            const paragraph = sentenceResults.slice(i, i + paragraphSize);
            
            // Create paragraph text with timestamps
            let paragraphText = '';
            paragraph.forEach(sentence => {
              paragraphText += `[${formatTime(sentence.start_time)}] ${sentence.text} `;
            });

            // Format long text to fit within page width (word wrapping)
            const textLines = pdf.splitTextToSize(paragraphText, contentWidth);
            
            // Check if we need a new page
            if (yPosition + (textLines.length * 5) > pdf.internal.pageSize.getHeight() - margin) {
              pdf.addPage();
              yPosition = margin;
            }
            
            pdf.text(textLines, margin, yPosition);
            yPosition += (textLines.length * 5) + 2;
          }
        }
      }

      // Add page numbers
      const totalPages = pdf.internal.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setTextColor(150, 150, 150);
        pdf.text(`Page ${i} of ${totalPages}`, pageWidth - margin - 15, pdf.internal.pageSize.getHeight() - 10);
      }

      // Save and download the PDF
      pdf.save(`video-analysis-${new Date().getTime()}.pdf`);
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Button
      onClick={generatePDF}
      disabled={isGenerating}
      className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white py-2 px-4 rounded-lg shadow-md transition-all"
      size="lg"
    >
      {isGenerating ? (
        <>
          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
          Generating PDF...
        </>
      ) : (
        <>
          <Download className="mr-2 h-5 w-5" />
          Download Analysis as PDF
        </>
      )}
    </Button>
  );
};

export default PDFExportButton;