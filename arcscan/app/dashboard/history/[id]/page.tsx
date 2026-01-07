'use client'
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Download, Calendar, ExternalLink, Play } from 'lucide-react';
import { useParams } from "next/navigation";
import { doc, getDoc } from "firebase/firestore";
import { db } from "@/lib/firebaseConfig";
import { Label } from '@radix-ui/react-label';
import { useRouter } from 'next/navigation';
// @ts-ignore
import html2pdf from 'html2pdf.js';
import ProtectedRoute from '@/components/ProtectedRoute';

interface AnalysisData {
  title: string;
  video_url: string;
  created_at: any;
  user_id: string;
  emotion_summary: {
    [emotion: string]: {
      average_score: number;
      occurrences: number;
    };
  };
  sentence_emotions: {
    text: string;
    emotion: string;
    score: number;
    end_time: number;
  }[];
}

const SentimentAnalysisResults = () => {
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [dominantEmotion, setDominantEmotion] = useState<string>('Unknown');
  const [topEmotions, setTopEmotions] = useState<[string, number][]>([]);
  const router = useRouter();
  const params = useParams();
  const id = params.id as string;

  const getBarColorCode = (emotion: string) => {
    const colorMap: Record<string, string> = {
      joy: '#34d399',
      pride: '#6366f1',
      neutral: '#9ca3af',
      anger: '#dc2626',
      annoyance: '#fb923c',
      anticipation: '#60a5fa',
      surprise: '#facc15',
      disapproval: '#f87171',
      disgust: '#f472b6',
      sadness: '#93c5fd',
      fear: '#a5b4fc',
      love: '#f43f5e',
      amusement: '#c084fc',
      realization: '#22d3ee',
      trust: '#22c55e',
      caring: '#a3e635',
      confusion: '#a1a1aa',
      nervousness: '#e879f9',
      approval: '#14b8a6',
      optimism: '#4ade80',
      disappointment: '#fca5a5',
      grief: '#94a3b8',
      excitement: '#a78bfa',
      curiosity: '#38bdf8',
      gratitude: '#fbbf24',
      relief: '#5eead4',
      time: '#d1d5db',
      unknown: '#cbd5e1',
    };

    const key = emotion.toLowerCase().trim();
    return colorMap[key] || '#cbd5e1';
  };

  const getBarColor = (emotion: string) => {
    const colorMap: Record<string, string> = {
      joy: 'bg-emerald-400',
      pride: 'bg-indigo-500',
      neutral: 'bg-gray-400',
      anger: 'bg-red-600',
      annoyance: 'bg-orange-400',
      anticipation: 'bg-blue-400',
      surprise: 'bg-yellow-400',
      disapproval: 'bg-red-400',
      disgust: 'bg-pink-400',
      sadness: 'bg-blue-300',
      fear: 'bg-indigo-300',
      love: 'bg-rose-400',
      amusement: 'bg-purple-400',
      realization: 'bg-cyan-400',
      trust: 'bg-green-500',
      caring: 'bg-lime-400',
      confusion: 'bg-zinc-400',
      nervousness: 'bg-fuchsia-400',
      approval: 'bg-teal-400',
      optimism: 'bg-green-400',
      disappointment: 'bg-red-300',
      grief: 'bg-slate-400',
      excitement: 'bg-violet-400',
      curiosity: 'bg-sky-400',
      gratitude: 'bg-amber-400',
      relief: 'bg-teal-300',
      time: 'bg-gray-300',
      unknown: 'bg-slate-300',
    };
    const key = emotion.toLowerCase().trim();
    return colorMap[key] || 'bg-slate-300';
  };

  const getTopEmotionForSentence = (sentence: any) => {
    if (!sentence.emotions || sentence.emotions.length === 0) return 'Unknown';
    const sorted = sentence.emotions.sort((a: any, b: any) => b.score - a.score);
    return sorted[0].emotion;
  };

  useEffect(() => {
    const fetchAnalysis = async () => {
      if (!id) return;

      const ref = doc(db, "advanced_analyses", id);
      const snap = await getDoc(ref);

      if (snap.exists()) {
        const fetched = snap.data() as AnalysisData;
        const createdAt = fetched.created_at.toDate();
        setAnalysis({
          ...fetched,
          created_at: createdAt,
        });

        const entries = Object.entries(fetched.emotion_summary || {});
        const nonZero = entries.filter(
          ([_, val]: any) => val?.average_score > 0
        ) as [string, { average_score: number }][];

        const sorted: [string, number][] = nonZero
          .sort((a, b) => b[1].average_score - a[1].average_score)
          .map(([label, val]) => [label, val.average_score]);
        const topThree = sorted.slice(0, 3);
        setTopEmotions(topThree);

        const dominant = topThree[0]?.[0] || 'Unknown';
        setDominantEmotion(dominant);
      } else {
        setAnalysis(null);
      }
      setIsLoading(false);
    };

    fetchAnalysis();
  }, [id]);

  const getSentimentEmoji = (sentiment: string) => {
    const emojiMap: Record<string, string> = {
      joy: '😊', pride: '🏅', neutral: '😐', anger: '😡', annoyance: '😒',
      anticipation: '🤞', surprise: '😲', disapproval: '👎', disgust: '🤢',
      sadness: '😢', fear: '😱', love: '❤️', amusement: '😂', realization: '💡',
      trust: '🤝', caring: '💖', confusion: '😕', nervousness: '😬',
      approval: '👍', optimism: '🌞', disappointment: '😞', grief: '🖤',
      excitement: '🤩', curiosity: '🧐', gratitude: '🙏', relief: '😌',
      time: '⏳', unknown: '❓',
    };
    return emojiMap[sentiment.toLowerCase().trim()] || '❓';
  };

  const getSentimentColor = (sentiment: string) => {
    const colorMap: Record<string, string> = {
      joy: 'text-emerald-700 bg-emerald-50 border-emerald-200',
      pride: 'text-indigo-700 bg-indigo-50 border-indigo-200',
      neutral: 'text-gray-700 bg-gray-50 border-gray-200',
      anger: 'text-red-700 bg-red-50 border-red-200',
      annoyance: 'text-orange-700 bg-orange-50 border-orange-200',
      anticipation: 'text-blue-700 bg-blue-50 border-blue-200',
      surprise: 'text-yellow-700 bg-yellow-50 border-yellow-200',
      disapproval: 'text-red-800 bg-red-100 border-red-300',
      disgust: 'text-pink-700 bg-pink-50 border-pink-200',
      sadness: 'text-blue-700 bg-blue-100 border-blue-300',
      fear: 'text-indigo-700 bg-indigo-100 border-indigo-300',
      love: 'text-rose-700 bg-rose-50 border-rose-200',
      amusement: 'text-purple-700 bg-purple-50 border-purple-200',
      realization: 'text-cyan-700 bg-cyan-50 border-cyan-200',
      trust: 'text-green-700 bg-green-50 border-green-200',
      caring: 'text-lime-700 bg-lime-50 border-lime-200',
      confusion: 'text-zinc-700 bg-zinc-50 border-zinc-200',
      nervousness: 'text-fuchsia-700 bg-fuchsia-50 border-fuchsia-200',
      approval: 'text-teal-700 bg-teal-50 border-teal-200',
      optimism: 'text-green-700 bg-green-50 border-green-200',
      disappointment: 'text-red-700 bg-red-50 border-red-200',
      grief: 'text-slate-700 bg-slate-50 border-slate-200',
      excitement: 'text-violet-700 bg-violet-50 border-violet-200',
      curiosity: 'text-sky-700 bg-sky-50 border-sky-200',
      gratitude: 'text-amber-700 bg-amber-50 border-amber-200',
      relief: 'text-teal-700 bg-teal-50 border-teal-200',
      time: 'text-gray-700 bg-gray-50 border-gray-200',
      unknown: 'text-slate-700 bg-slate-50 border-slate-200',
    };
    return colorMap[sentiment.toLowerCase().trim()] || 'text-slate-700 bg-slate-50 border-slate-200';
  };

  const getSentimentBadgeColor = (sentiment: string) => {
    const badgeMap: Record<string, string> = {
      joy: 'bg-emerald-100 text-emerald-800 border-emerald-300',
      pride: 'bg-indigo-100 text-indigo-800 border-indigo-300',
      neutral: 'bg-gray-100 text-gray-800 border-gray-300',
      anger: 'bg-red-100 text-red-800 border-red-300',
      annoyance: 'bg-orange-100 text-orange-800 border-orange-300',
      anticipation: 'bg-blue-100 text-blue-800 border-blue-300',
      surprise: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      disapproval: 'bg-red-200 text-red-900 border-red-300',
      disgust: 'bg-pink-100 text-pink-800 border-pink-300',
      sadness: 'bg-blue-200 text-blue-900 border-blue-300',
      fear: 'bg-indigo-200 text-indigo-900 border-indigo-300',
      love: 'bg-rose-100 text-rose-800 border-rose-300',
      amusement: 'bg-purple-100 text-purple-800 border-purple-300',
      realization: 'bg-cyan-100 text-cyan-800 border-cyan-300',
      trust: 'bg-green-100 text-green-800 border-green-300',
      caring: 'bg-lime-100 text-lime-800 border-lime-300',
      confusion: 'bg-zinc-100 text-zinc-800 border-zinc-300',
      nervousness: 'bg-fuchsia-100 text-fuchsia-800 border-fuchsia-300',
      approval: 'bg-teal-100 text-teal-800 border-teal-300',
      optimism: 'bg-green-100 text-green-800 border-green-300',
      disappointment: 'bg-red-100 text-red-800 border-red-300',
      grief: 'bg-slate-100 text-slate-800 border-slate-300',
      excitement: 'bg-violet-100 text-violet-800 border-violet-300',
      curiosity: 'bg-sky-100 text-sky-800 border-sky-300',
      gratitude: 'bg-amber-100 text-amber-800 border-amber-300',
      relief: 'bg-teal-100 text-teal-800 border-teal-300',
      time: 'bg-gray-100 text-gray-800 border-gray-300',
      unknown: 'bg-slate-100 text-slate-800 border-slate-300',
    };
    return badgeMap[sentiment.toLowerCase().trim()] || 'bg-slate-100 text-slate-800 border-slate-300';
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric", month: "long", day: "numeric",
      hour: "2-digit", minute: "2-digit", timeZoneName: "short",
    }).format(date);
  };

  const truncateUrl = (url: string, maxLength = 60) => {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + '...';
  };

  const handleBackToHistory = () => {
    router.push('/dashboard/history');
  };

  const handleDownloadPDF = () => {
    const element = document.getElementById('analysis-report');
    if (!element) return;
    const opt = {
      margin: 0.5, filename: 'ArcScan_Report.pdf',
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(element).save();
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading analysis results...</p>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return <p>No analysis found.</p>;
  }

  const data = analysis;

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div id="analysis-report">
            {/* Header Section */}
            <div className="bg-white rounded-2xl shadow-lg p-6 sm:p-8 mb-8">
              <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
                <div className="flex-1">
                  <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4 leading-tight">
                    {data.title}
                  </h1>
                  <div className="flex flex-col sm:flex-row sm:items-center gap-4 text-sm text-gray-600">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-blue-500" />
                      <span className="font-medium">Analyzed on {formatDate(data.created_at)}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Play className="w-4 h-4 text-red-500" />
                      <a href={data.video_url} target="_blank" rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 hover:underline font-medium transition-colors duration-200">
                        {truncateUrl(data.video_url)}
                      </a>
                      <ExternalLink className="w-3 h-3 text-gray-400" />
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4 bg-gradient-to-r from-gray-50 to-gray-100 px-6 py-4 rounded-xl border border-gray-200">
                  <span className="text-3xl">{getSentimentEmoji(dominantEmotion)}</span>
                  <div>
                    <p className="text-sm text-gray-600 font-medium">Overall Sentiment</p>
                    <p className="font-bold text-xl text-gray-800">{dominantEmotion}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Sentiment Distribution Section */}
            <div className="bg-white rounded-2xl shadow-lg p-6 sm:p-8 mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-8">Sentiment Distribution</h2>

              {/* Visual Bars Instead of Chart */}
              <div className="space-y-6 mb-8">
                {topEmotions.map(([label, score]) => (
                  <div key={label} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">{getSentimentEmoji(label)}</span>
                        <span className="font-semibold capitalize text-gray-800 text-lg">{label}</span>
                      </div>
                      <span className="text-2xl font-bold text-gray-700">{score.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div 
                        className="h-4 rounded-full transition-all duration-1000 ease-out"
                        style={{ 
                          width: `${score}%`, 
                          backgroundColor: getBarColorCode(label) 
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Statistics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {topEmotions.map(([label, score]) => (
                  <div key={label} className="flex items-center justify-between p-5 bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl border-2 border-slate-200">
                    <div className="flex items-center gap-3">
                      <div className={`w-5 h-5 rounded-full shadow-sm ${getBarColor(label)}`}></div>
                      <span className="font-semibold capitalize text-slate-800">{label}</span>
                    </div>
                    <span className="text-xl font-bold text-slate-700">{score.toFixed(1)}%</span>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-5 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-200">
                <h3 className="font-semibold text-blue-800 mb-2 flex items-center gap-2">
                  <span className="text-blue-600">📊</span>
                  Analysis Summary
                </h3>
                <p className="text-sm text-blue-700 leading-relaxed">
                  This {data.sentence_emotions.length}-sentence analysis reveals a <span className="font-semibold">{dominantEmotion}</span> overall tone.
                  The content is mainly characterized by{" "}
                  {topEmotions.map(([label], index) => {
                    if (index === 0) return <span key={label} className="font-medium">{label}</span>;
                    if (index === topEmotions.length - 1) return <span key={label}>, and {label}</span>;
                    return <span key={label}>, {label}</span>;
                  })}.
                </p>
              </div>
            </div>

            {/* Transcript Analysis Section */}
            <div className="bg-white rounded-2xl shadow-lg p-6 sm:p-8 mb-8">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-2xl sm:text-3xl font-bold text-gray-900">Detailed Transcript Analysis</h2>
                <div className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
                  {data.sentence_emotions.length} sentences analyzed
                </div>
              </div>

              <div className="space-y-4">
                {data.sentence_emotions.map((sentence, index) => (
                  <div key={index} className={`p-5 rounded-xl border-2 transition-all duration-300 hover:shadow-md hover:scale-[1.01] ${getSentimentColor(getTopEmotionForSentence(sentence))}`}>
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full bg-white shadow-sm">
                        <span className="text-xl">{getSentimentEmoji(getTopEmotionForSentence(sentence))}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-gray-800 leading-relaxed mb-3">{sentence.text}</p>
                        <div className="flex items-center gap-3">
                          <span className={`text-xs px-3 py-1 rounded-full border font-medium ${getSentimentBadgeColor(getTopEmotionForSentence(sentence))}`}>
                            {getTopEmotionForSentence(sentence)}
                          </span>
                          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                            Sentence {index + 1}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <button onClick={handleBackToHistory}
              className="flex items-center gap-2 px-8 py-3 bg-white hover:bg-gray-50 text-gray-700 rounded-xl border-2 border-gray-200 transition-all duration-200 font-semibold shadow-md hover:shadow-lg">
              <ArrowLeft className="w-5 h-5" />
              Back to History
            </button>

            <button onClick={handleDownloadPDF}
              className="flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl transition-all duration-200 font-semibold shadow-lg hover:shadow-xl transform hover:scale-105">
              <Download className="w-5 h-5" />
              Download PDF Report
            </button>
          </div>
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default SentimentAnalysisResults;