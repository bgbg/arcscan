import './globals.css'
import { Inter } from 'next/font/google'
import { Navbar } from '@/components/navbar'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'ArcScan - Video Analysis Platform',
  description: 'Analyze videos and YouTube links with advanced AI technology',
 icons: {
    icon: '/icon.png', // This explicitly tells Next.js to use icon.png
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Navbar />
        <main className="min-h-screen bg-gray-50 pt-16">
          {children}
        </main>
      </body>
    </html>
  )
}

