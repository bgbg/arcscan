'use client'

import { DashboardNav } from '@/components/dashboard-nav'
import { Navbar } from '@/components/navbar'
import { auth } from '@/lib/firebaseConfig'
import { useAuthState } from 'react-firebase-hooks/auth'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const [user, loading] = useAuthState(auth)

  if (loading) return null

  return (
    <div className="min-h-screen bg-white text-gray-900">
      {/* Show either authenticated nav or guest nav */}
      {user ? <DashboardNav /> : <Navbar />}

      {/* Don't force container styling — let each page control its own layout */}
      <div className="overflow-x-hidden">
        {children}
      </div>
    </div>
  )
}
