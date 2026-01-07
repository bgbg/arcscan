'use client'

import React, { useEffect, useState } from 'react';
import { auth, db } from '@/lib/firebaseConfig';
import { collection, getDocs, query, where } from 'firebase/firestore';
import { User, onAuthStateChanged } from 'firebase/auth';
import { useRouter } from 'next/navigation';
import ProtectedRoute from '@/components/ProtectedRoute';
interface Sentiment {
type: string;
emoji: string;
color: string;
}

interface Emotions {
  label: string;
  percentage: number;
}


interface AnalysisItem {
  id: string;
  title: string;
  thumbnail: string;
  date: string;
  duration: string;
  sentiment: Sentiment;
  emotions: Emotions[]; // لأنه صار مصفوفة
}


interface EmotionBarProps {
  label: string;
  percentage: number;
  cardId: string;
}


interface StatItem {
number: string;
label: string;
color: string;
}

interface StatCardProps {
stat: StatItem;
}

interface AnalysisCardProps {
analysis: AnalysisItem;
}

const ArcScanHistory: React.FC = () => {
const router = useRouter();
const [animatedBars, setAnimatedBars] = useState(new Set());
const [analyses, setAnalyses] = useState<AnalysisItem[]>([]);
const [user, setUser] = useState<User | null>(null);
const [loading, setLoading] = useState(true);
const [sortDescending, setSortDescending] = useState(true);
const [filterSentiment, setFilterSentiment] = useState<'All' | 'Positive' | 'Neutral' | 'Negative'>('All');
const emotionCount: Record<string, number> = {};
// أعلى ايموشن سكور في كل التحليلات
let maxEmotionScore = 0;
let maxEmotionLabel = 'None';
const sortedAnalyses = analyses
.slice()
.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

// 2. نجيب أحدث شعور
const latestSentiment = sortedAnalyses.length > 0
? sortedAnalyses[0].sentiment.type
: 'N/A';

// 3. نجيب تاريخ آخر تحليل
const lastAnalysisDate = sortedAnalyses.length > 0
? sortedAnalyses[0].date
: 'N/A';
analyses.forEach((a) => {
  a.emotions.forEach((e) => {
    if (!emotionCount[e.label]) emotionCount[e.label] = 0;
    emotionCount[e.label]++;
  });
});

const mostFrequentEmotion = Object.entries(emotionCount)
  .sort((a, b) => b[1] - a[1])[0]?.[0] || 'Unknown';


// لو في تحليلات، نرتبها حسب التاريخ تنازلياً

const stats: StatItem[] = [
  { number: String(analyses.length), label: 'Total Analyses', color: 'text-purple-600' },
  { number: mostFrequentEmotion, label: 'Most Frequent Emotion', color: 'text-indigo-600' },
  { number: latestSentiment, label: 'Last Sentiment', color: 'text-amber-500' }, 
{
number: lastAnalysisDate,
label: 'Last Analysis Date',
color: 'text-slate-600'
}

];


useEffect(() => {
  const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
    if (currentUser) {
      setUser(currentUser);

      const q = query(
        collection(db, 'advanced_analyses'),
        where('user_id', '==', currentUser.uid)
      );

      const snapshot = await getDocs(q);

      const userData = snapshot.docs.map((doc) => {
        const data = doc.data();
             console.log('🎯 doc.id:', doc.id);
console.log('📊 emotion_summary:', data.emotion_summary);



        const getSentimentEmoji = (type: string) => {
          const map: Record<string, string> = {
         joy: '😊',
    pride: '😌',
    neutral: '😐',
    anger: '😡',
    annoyance: '😠',
    anticipation: '🤔',
    surprise: '😲',
    disapproval: '😞',
    disgust: '🤢',
    sadness: '😢',
    fear: '😨',
    love: '❤️',
    amusement: '😂',
    realization: '💡',
    trust: '🤝',
    caring: '💙',
    confusion: '😵',
    nervousness: '😬',
    approval: '👍',
    optimism: '🌤️',
    disappointment: '🙁',
    grief: '🖤',
    excitement: '🤩',
    curiosity: '🔍',
    gratitude: '🙏',
    relief: '😌',
    time: '⏳',
    admiration: '👏',
    unknown: '❓',
    'no clear emotion': '❔'
          };
          return map[type] || '❓';
        };

        const getSentimentColor = (type: string) => {
          const map: Record<string, string> = {
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
    Unknown: 'bg-slate-300'
  };
console.log("💥 type in emoji:", type);

const key = type.toLowerCase().trim();
return map[key] || `❓ (${key})`;
        };

        const dateObj = new Date(data.created_at?.seconds * 1000);
        const formattedDate = dateObj.toLocaleDateString('en-GB', {
          day: 'numeric',
          month: 'long',
          year: 'numeric'
        });

        // تحليل المشاعر
        const emotionSummary = data.emotion_summary || {};

    const rawEmotions = data.emotion_summary || {};
const emotionEntries = Object.entries(rawEmotions) as [string, { average_score: number }][];
const nonZeroEmotions = emotionEntries
  .filter(([_, val]) => typeof val.average_score === 'number' && val.average_score > 0)
  .map(([label, val]) => [label, val.average_score] as [string, number]);
const sorted = nonZeroEmotions.sort((a, b) => b[1] - a[1]);
const topThree = sorted.slice(0, 3);

const emotions = topThree.map(([label, value]) => ({
  label,
  percentage: value
}));

const dominant = topThree[0]?.[0] || 'No clear emotion';

console.log('✅ topThree:', topThree);

        return {
          id: doc.id,
          title: data.title || data.video_url || 'Untitled Analysis',
          thumbnail: '📹',
          date: formattedDate,
          duration: '–',
          sentiment: {
            type: dominant,
            emoji: getSentimentEmoji(dominant),
            color: getSentimentColor(dominant)
          },
          emotions
        };
      });

      setAnalyses(userData);
const sortedByDate = [...userData].sort((a, b) =>
new Date(b.date).getTime() - new Date(a.date).getTime()
);
const lastAnalysisDate = sortedByDate[0]?.date || 'N/A';


    }
    
    setLoading(false);
  });

  return () => unsubscribe();
}, []);

const EmotionBar: React.FC<EmotionBarProps> = ({ label, percentage }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 200);
    return () => clearTimeout(timer);
  }, []);

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
    Unknown: 'bg-slate-300'
  };

  const barColor = colorMap[label] || 'bg-slate-300';

  return (
    <div className="flex items-center gap-3">
      <div className="min-w-[70px] text-sm font-medium capitalize text-slate-700">{label}</div>
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColor}`}
          style={{ width: isVisible ? `${percentage}%` : '0%' }}
        />
      </div>
      <div className="min-w-[35px] text-sm text-slate-500 text-right">
        {percentage.toFixed(1)}%
      </div>
    </div>
  );
};


const AnalysisCard: React.FC<AnalysisCardProps> = ({ analysis }) => {
const [isHovered, setIsHovered] = useState(false);
const [buttonClicked, setButtonClicked] = useState(false);

const handleViewDetails = () => {
setButtonClicked(true);
setTimeout(() => setButtonClicked(false), 1500);
router.push(`/dashboard/history/${analysis.id}`);
};

return (
<div
className={`bg-white rounded-2xl shadow-lg border border-slate-200/60 overflow-hidden transition-all duration-300 ease-out ${
isHovered ? 'transform -translate-y-1 shadow-2xl shadow-slate-900/10' : ''
}`}
onMouseEnter={() => setIsHovered(true)}
onMouseLeave={() => setIsHovered(false)}
>
<div className="p-6 pb-0">
<div className="flex items-center gap-3 mb-4">
<div className="w-12 h-12 bg-gradient-to-br from-slate-100 to-slate-200 rounded-lg flex items-center justify-center text-xl">
{analysis.thumbnail}
</div>
<div className="flex-1">
<h3 className="font-semibold text-slate-800 text-base leading-snug mb-1">{analysis.title}</h3>
<div className="flex items-center gap-4 text-sm text-slate-500">
<span>📅 {analysis.date}</span>
</div>
</div>
</div>

<div className="flex items-center justify-between mb-6">
<div className="flex items-center gap-2 font-semibold">
<span className="text-xl">{analysis.sentiment.emoji}</span>
<span className={analysis.sentiment.color}>{analysis.sentiment.type}</span>
</div>
</div>
</div>

<div className="px-6 mb-6 space-y-3">
{analysis.emotions.map((e, idx) => (
  <EmotionBar key={idx} label={e.label} percentage={e.percentage} cardId={analysis.id} />
))}

</div>

<div className="px-6 pb-6">
<button
onClick={handleViewDetails}
className="w-full py-3 px-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-medium text-sm transition-all duration-200 hover:transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-indigo-500/25"
>
{buttonClicked ? '✓ Loading...' : '🔍 View Details'}
</button>
</div>
</div>
);
};

const StatCard: React.FC<StatCardProps> = ({ stat }) => (
<div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200/80 text-center">
<div className={`text-3xl font-bold mb-1 ${stat.color}`}>{stat.number}</div>
<div className="text-sm text-slate-500 uppercase tracking-wide font-medium">{stat.label}</div>
</div>
);
const filteredAnalyses = analyses
.filter((a) =>
  filterSentiment === 'All' || a.sentiment.type.toLowerCase() === filterSentiment.toLowerCase()
)
.sort((a, b) => {
const dateA = new Date(a.date);
const dateB = new Date(b.date);
return sortDescending
? dateB.getTime() - dateA.getTime()
: dateA.getTime() - dateB.getTime();
});

return (
    <ProtectedRoute>
<div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
<div className="text-center mb-12">
<h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
Analysis History
</h1>
<p className="text-lg text-slate-600 font-normal">
Track your emotional insights and content analysis over time
</p>
</div>
<div className="flex justify-center flex-wrap gap-4 mb-8">

<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
{stats.map((stat, index) => (
<StatCard key={index} stat={stat} />
))}
</div>
</div>
{loading && (
<div className="text-center py-20 text-slate-500 text-lg">Loading analyses...</div>
)}
<div className="flex flex-wrap items-center gap-2 mb-6">
<button
onClick={() => setSortDescending(!sortDescending)}
className="flex items-center gap-2 py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white text-sm rounded-lg transition"
>
{sortDescending ? '🔽 Newest First' : '🔼 Oldest First'}
</button>

<select
  value={filterSentiment}
  onChange={(e) => setFilterSentiment(e.target.value as any)}
  className="py-2 px-3 border border-slate-300 text-sm rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
>
  <option value="All">All Emotions</option>
  <option value="joy">Joy</option>
  <option value="pride">Pride</option>
  <option value="neutral">Neutral</option>
  <option value="anger">Anger</option>
  <option value="annoyance">Annoyance</option>
  <option value="anticipation">Anticipation</option>
  <option value="surprise">Surprise</option>
  <option value="disapproval">Disapproval</option>
  <option value="disgust">Disgust</option>
  <option value="sadness">Sadness</option>
  <option value="fear">Fear</option>
  <option value="love">Love</option>
  <option value="amusement">Amusement</option>
  <option value="realization">Realization</option>
  <option value="trust">Trust</option>
  <option value="caring">Caring</option>
  <option value="confusion">Confusion</option>
  <option value="nervousness">Nervousness</option>
  <option value="approval">Approval</option>
  <option value="optimism">Optimism</option>
  <option value="disappointment">Disappointment</option>
  <option value="grief">Grief</option>
  <option value="excitement">Excitement</option>
  <option value="curiosity">Curiosity</option>
  <option value="gratitude">Gratitude</option>
  <option value="relief">Relief</option>
  <option value="admiration">Admiration</option>
</select>

<button
onClick={() => {
setSortDescending(true);
setFilterSentiment('All');
}}
className="py-2 px-4 bg-slate-200 hover:bg-slate-300 text-slate-700 text-sm rounded-lg transition"
>
Reset Filters
</button>
</div>

<div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">

{filteredAnalyses.map((analysis) => (
<AnalysisCard key={analysis.id} analysis={analysis} />
))}


</div>

{filteredAnalyses.length === 0 && !loading && (
<div className="text-center py-16 text-slate-500">
<div className="text-6xl mb-4 opacity-50">🤷‍♀️</div>
<h3 className="text-xl font-medium mb-2">No analyses found</h3>
<p>
{filterSentiment === 'All'
? 'Start analyzing content to see your history here.'
: `No analyses found with sentiment: ${filterSentiment}.`}
</p>
</div>
)}


</div>
</div>
</ProtectedRoute>
);
};

export default ArcScanHistory;