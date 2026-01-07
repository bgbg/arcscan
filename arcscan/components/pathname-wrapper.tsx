'use client'

import { usePathname } from 'next/navigation'

export function PathnameWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const showPadding = !pathname.startsWith('/login') && !pathname.startsWith('/register') && !pathname.startsWith('/reset-password')&& !pathname.startsWith('/verify')

  return (
    <main className={`min-h-screen bg-gray-50 ${showPadding ? 'pt-16' : ''}`}>
      {children}
    </main>
  )
}
