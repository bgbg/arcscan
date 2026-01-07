// ✅ صفحة Login معدّلة لتتناسق مع ستايل ArcScan العام (ألوان + أزرار + تدرجات)

'use client'

import { useState } from 'react'
import {
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
} from 'firebase/auth'
import { auth } from '@/lib/firebaseConfig'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { useRouter } from 'next/navigation'

export default function Login() {
  const router = useRouter()

  const [formData, setFormData] = useState({ email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      )
      console.log('Logged in:', userCredential.user)
      setFormData({ email: '', password: '' })
      router.push('/dashboard')
    } catch (error: any) {
      console.error('Login error:', error)
      setError(error.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleLogin = async () => {
    setLoading(true)
    setError('')

    try {
      const provider = new GoogleAuthProvider()
      const result = await signInWithPopup(auth, provider)
      console.log('Google login:', result.user)
      router.push('/dashboard')
    } catch (error: any) {
      console.error('Google login error:', error)
      setError(error.message || 'Google login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-orange-50 flex items-center justify-center px-4 py-16">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-3xl font-extrabold text-center mb-6 text-transparent bg-gradient-to-r from-purple-600 via-pink-500 to-orange-500 bg-clip-text">
          Welcome Back
        </h1>

        {error && (
          <div className="text-red-600 text-sm mb-4 text-center font-medium">{error}</div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="email">Email</Label>
            <Input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              className="mt-1"
            />
          </div>

          <div>
            <Label htmlFor="password">Password</Label>
            <Input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              className="mt-1"
            />
          </div>
          <div className="text-right">
  <a
    href="/reset-password"
    className="text-sm text-purple-600 hover:underline font-medium"
  >
    Forgot your password?
  </a>
</div>


          <Button
            type="submit"
            className="w-full bg-gradient-to-br from-purple-500 to-pink-500 text-white font-semibold hover:from-purple-600 hover:to-pink-600 transition"
            disabled={loading}
          >
            {loading ? 'Logging in...' : '🚀 Log In'}
          </Button>
        </form>

        <div className="mt-6">
          <Button
            onClick={handleGoogleLogin}
            variant="outline"
            className="w-full border-gray-300 text-gray-700 font-medium hover:bg-gray-50"
            disabled={loading}
          >
            Continue with Google
          </Button>
        </div>

        <p className="text-center text-sm text-gray-600 mt-6">
          Don't have an account?{' '}
          <a href="/register" className="text-purple-600 font-medium hover:underline">
            Register here
          </a>
        </p>
      </div>
    </div>
  )
}
