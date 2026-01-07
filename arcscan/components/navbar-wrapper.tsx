'use client'

import { usePathname } from 'next/navigation'
import { Navbar } from './navbar'

export function NavbarWrapper() {
  const pathname = usePathname()
  const showNavbar = !pathname.startsWith('/login') && !pathname.startsWith('/register') && !pathname.startsWith('/reset-password') && !pathname.startsWith('/verify')

  return showNavbar ? <Navbar /> : null
}
