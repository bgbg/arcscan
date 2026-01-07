'use client'

import { useAuthState } from 'react-firebase-hooks/auth'
import { auth } from '@/lib/firebaseConfig'
import { useRouter } from 'next/navigation'
import { useEffect, ReactNode } from 'react'

export default function ProtectedRoute({ children }: { children: ReactNode }) {
  const [user, loading] = useAuthState(auth)
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login')
    }
  }, [user, loading, router])

  if (loading || !user) return null // أو ممكن تعرض سبينر

  return <>{children}</>
}
