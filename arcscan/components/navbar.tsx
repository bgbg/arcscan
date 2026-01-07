'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { User, UserCircle2, Settings, LogOut, ChevronDown } from 'lucide-react'
import { auth } from '@/lib/firebaseConfig'
import { useAuthState } from 'react-firebase-hooks/auth'
import logo from '/app/logo.png' // ✅ الاستيراد الصحيح للصورة

export function Navbar() {
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [user] = useAuthState(auth)
  const dropdownRef = useRef(null)

  // إغلاق القائمة عند الضغط خارجها
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !(dropdownRef.current as any).contains(event.target)) {
        setDropdownOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white shadow-md border-b border-purple-100">
      <div className="max-w-7xl mx-auto px-4 flex items-center justify-between h-16">
        
        {/* Logo + عنوان */}
        <Link href="/" className="flex items-center gap-2 hover:opacity-90 transition">
          <Image
            src={logo}
            alt="ArcScan Logo"
            className="h-10 w-auto"
            priority
          />
          <h1 className="text-3xl font-extrabold bg-gradient-to-r from-purple-600 via-pink-500 to-orange-500 bg-clip-text text-transparent">
            ArcScan
          </h1>
        </Link>

        {user && (
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-2 bg-gradient-to-r from-purple-500 to-pink-500 px-3 py-2 rounded-full text-white font-semibold shadow hover:scale-105 transition"
            >
              <User className="w-4 h-4" />
              <ChevronDown className="w-4 h-4" />
            </button>

            {dropdownOpen && (
              <div className="absolute right-0 mt-3 w-48 bg-white rounded-xl shadow-lg border border-gray-200 animate-fade-in z-50">
                <Link
                  href="/dashboard/profile"
                  className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-purple-50"
                >
                  <UserCircle2 className="w-4 h-4" /> Profile
                </Link>
             
                <button
                  onClick={() => auth.signOut()}
                  className="flex w-full items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                >
                  <LogOut className="w-4 h-4" /> Logout
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </nav>
  )
}
