'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { createUserWithEmailAndPassword, updateProfile, GoogleAuthProvider, signInWithPopup, sendEmailVerification } from 'firebase/auth'
import { auth, db } from '@/lib/firebaseConfig'
import { doc, setDoc } from 'firebase/firestore'
import { useRouter } from 'next/navigation'
import { FcGoogle } from 'react-icons/fc'

export default function Register() {
  const router = useRouter()
  const [formData, setFormData] = useState({ name: '', email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prevState => ({ ...prevState, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setSuccess('')

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password)
      const user = userCredential.user

      await updateProfile(user, { displayName: formData.name })

      await setDoc(doc(db, 'users', user.uid), {
        name: formData.name,
        email: formData.email,
        createdAt: new Date(),
      })

      await sendEmailVerification(user)

router.push('verify')    } catch (error: any) {
      console.error('Signup error:', error)
      setError(error.message || 'Signup failed')
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleSignup = async () => {
    setLoading(true)
    setError('')
    try {
      const provider = new GoogleAuthProvider()
      const result = await signInWithPopup(auth, provider)
      const user = result.user

      await setDoc(doc(db, 'users', user.uid), {
        name: user.displayName,
        email: user.email,
        createdAt: new Date(),
      })

      router.push('/dashboard')
    } catch (err: any) {
      console.error('Google signup error:', err)
      setError(err.message || 'Google signup failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-pink-50 to-orange-50 px-4">
      <div className="max-w-md w-full bg-white/90 backdrop-blur shadow-xl rounded-2xl p-8">
        <h1 className="text-3xl font-extrabold text-center text-purple-600 mb-2">Create an Account</h1>
<p className="text-center text-transparent bg-clip-text bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 mb-6 text-base font-medium">
  Join ArcScan and start analyzing emotions in your videos today!
</p>
        {error && <p className="text-red-500 text-sm mb-4 text-center">{error}</p>}
        {success && <p className="text-green-600 text-sm mb-4 text-center">{success}</p>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="name">Name</Label>
            <Input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </div>

          <div>
            <Label htmlFor="email">Email</Label>
            <Input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
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
            />
          </div>

          <Button
            type="submit"
            className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold hover:from-purple-600 hover:to-pink-600"
            disabled={loading}
          >
            {loading ? 'Registering...' : 'Register'}
          </Button>
        </form>

        <div className="mt-6">
          <Button
            onClick={handleGoogleSignup}
            className="w-full bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 flex items-center justify-center gap-2"
            disabled={loading}
          >
            <FcGoogle className="text-xl" /> Continue with Google
          </Button>
        </div>

        <p className="text-center text-sm text-gray-600 mt-6">
          Already have an account?{' '}
          <a href="/login" className="text-purple-600 hover:underline">
            Log in here
          </a>
        </p>
      </div>
    </div>
  )
}
