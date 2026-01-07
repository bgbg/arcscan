'use client'

import { useState } from 'react'
import { sendPasswordResetEmail } from 'firebase/auth'
import { auth } from '@/lib/firebaseConfig'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { useRouter } from 'next/navigation'

export default function ResetPasswordPage() {
    const router = useRouter()

  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setMessage('')

    try {
      await sendPasswordResetEmail(auth, email)
      setMessage('✅ Check your inbox for reset instructions.')
    } catch (err: any) {
      setError(err.message || 'Something went wrong.')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-pink-50 to-orange-50 px-4">
      <div className="max-w-md w-full bg-white shadow-xl rounded-2xl p-8">
        <h1 className="text-2xl font-bold mb-4 text-center text-purple-600">Reset Your Password</h1>

        {message && <p className="text-green-600 text-sm mb-4 text-center">{message}</p>}
        {error && <p className="text-red-500 text-sm mb-4 text-center">{error}</p>}

        <form onSubmit={handleReset} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Your email address
            </label>
            <Input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1"
            />
          </div>
          <Button
            type="submit"
            className="w-full bg-gradient-to-br from-purple-500 to-pink-500 text-white font-semibold hover:from-purple-600 hover:to-pink-600"
          >
            Send Reset Email
          </Button>
          <p className="text-xs text-gray-500 mt-2 text-center">
  Didn't receive the email? Check your spam folder.
</p>
          <Button
  type="button"
  variant="outline"
  onClick={() => router.push('/login')}
  className="w-full"
>
  ← Back to Login
</Button>

        </form>
      </div>
    </div>
  )
}
