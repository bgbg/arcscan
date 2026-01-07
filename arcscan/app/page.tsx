'use client'

import Link from 'next/link'
import Image from 'next/image'
import logo from './logo.png' // ✅ لأن logo.png في نفس مجلد page.tsx
import { useEffect, useRef, useState } from 'react'

// الإيموجي العائمة
const emojis = [
  { emoji: '😊', style: 'top-[8%] left-[12%]' },
  { emoji: '😢', style: 'top-[12%] right-[14%]' },
  { emoji: '😡', style: 'top-[30%] left-[16%]' },
  { emoji: '😮', style: 'top-[40%] right-[10%]' }
]

export default function Home() {
  const [fadeIn, setFadeIn] = useState(false)
  const cardsRef = useRef<HTMLDivElement[]>([])
  const [visibleCards, setVisibleCards] = useState([false, false, false])

  useEffect(() => {
    // تفعيل أنيميشن العنوان عند تحميل الصفحة
    const timer = setTimeout(() => setFadeIn(true), 200)
    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry, idx) => {
          if (entry.isIntersecting) {
            setVisibleCards((prev) => {
              const updated = [...prev]
              updated[idx] = true
              return updated
            })
          }
        })
      },
      { threshold: 0.3 }
    )

    cardsRef.current.forEach((el) => {
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [])

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-500 via-pink-500 to-orange-400 text-white relative overflow-hidden">
      {/* الإيموجي الطائرة */}
      {emojis.map(({ emoji, style }, idx) => (
        <div
          key={idx}
          className={`floating-emoji absolute text-3xl animate-float ${style}`}
        >
          {emoji}
        </div>
      ))}

      {/* الهيدر */}
      <header className="fixed top-0 left-0 w-full z-50 bg-white/90 backdrop-blur-md shadow">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-extrabold bg-gradient-to-r from-purple-600 via-pink-500 to-orange-500 bg-clip-text text-transparent flex items-center gap-2">
  <Image 
    src={logo} 
    alt="ArcScan Logo" 
    className="h-12 w-auto"
    priority 
  />
  ArcScan
</h1>

          <nav className="hidden md:flex space-x-8 text-gray-800 font-medium">
            <a href="#features" className="hover:text-purple-600">Features</a>
            <a href="#about" className="hover:text-purple-600">About</a>
            <a href="#contact" className="hover:text-purple-600">Contact</a>
          </nav>
        </div>
      </header>

      {/* قسم البطل */}
      <section className="flex items-center justify-center min-h-screen text-center px-6">
        <div className={`max-w-3xl transition-all duration-1000 ease-out ${fadeIn ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
          <h1 className="text-3xl md:text-6xl font-extrabold mb-6">
            Understand Emotion in Every Frame 
          </h1>
          <p className="text-lg md:text-xl mb-8 text-white/90">
            ArcScan uses advanced AI to analyze the emotional journey within your videos. Paste a YouTube link or upload a file and let us decode tone, mood, and sentiment with clarity.
          </p>
          <div className="flex justify-center flex-wrap gap-4">
            <Link href="/dashboard">
              <button className="bg-white text-purple-600 font-bold px-6 py-3 rounded-full shadow hover:shadow-lg transition">
                Start Analyzing
              </button>
            </Link>
           <Link href="/login">
  <button className="bg-white/20 text-white font-semibold px-6 py-3 rounded-full border border-white hover:bg-white/30 transition">
    Sign in
  </button>
</Link>
          </div>
        </div>
      </section>

      {/* الميزات */}
      <section id="features" className="bg-white text-gray-800 py-20 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <h3 className="text-3xl font-bold mb-6">What ArcScan Offers</h3>
          <p className="text-gray-600 mb-12 max-w-xl mx-auto">
            Built with state-of-the-art NLP models and emotion recognition tools, ArcScan provides sentence-level insights, visual breakdowns, and rich analytics.
          </p>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: '🧠',
                title: 'Emotion Detection',
                text: 'Identify positive, negative, and neutral tones across your video timeline.',
                bg: 'bg-purple-50'
              },
              {
                icon: '📈',
                title: 'Timeline Visuals',
                text: 'View how emotions shift sentence by sentence, with beautiful charts.',
                bg: 'bg-pink-50'
              },
              {
                icon: '🔒',
                title: 'Private & Secure',
                text: 'Uploaded content is analyzed locally or securely processed and removed.',
                bg: 'bg-orange-50'
              }
            ].map((item, idx) => (
              <div
                key={idx}
                ref={(el) => (cardsRef.current[idx] = el!)}
                className={`p-6 ${item.bg} rounded-2xl shadow transform transition-all duration-700 ease-out
                ${visibleCards[idx] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}
              >
                <div className="text-4xl mb-4">{item.icon}</div>
                <h4 className="text-lg font-bold mb-2">{item.title}</h4>
                <p className="text-sm text-gray-700">{item.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* الفوتر */}
      <footer id="contact" className="bg-gray-900 text-white py-10 text-center mt-10">
        <p className="text-sm text-white/70">
          © 2025 ArcScan — Graduation Project by Hiba
        </p>
      </footer>

      {/* أنيميشن الإيموجي */}
      <style jsx>{`
        .floating-emoji {
          animation: float 6s ease-in-out infinite;
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-15px); }
        }
      `}</style>
    </main>
  )
}
