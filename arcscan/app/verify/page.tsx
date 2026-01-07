'use client'

import { useEffect } from 'react'
import { auth } from '@/lib/firebaseConfig'
import { useRouter } from 'next/navigation'
import { sendEmailVerification } from 'firebase/auth'
export default function VerifyPage() {
const router = useRouter()

useEffect(() => {
const interval = setInterval(async () => {
await auth.currentUser?.reload()
if (auth.currentUser?.emailVerified) {
router.push('/dashboard')
}
}, 3000)

return () => clearInterval(interval)
}, [router])

const resendVerification = async () => {
const user = auth.currentUser
if (user && !user.emailVerified) {
try {
await sendEmailVerification(user)
alert('Verification email sent again!')
} catch (error) {
alert('Failed to resend verification email.')
}
}
}

return (
<div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-pink-50 to-orange-50 px-4">
<div className="max-w-md w-full bg-white/90 backdrop-blur shadow-xl rounded-2xl p-8 text-center">
<h1 className="text-2xl font-bold text-purple-600 mb-4">🔐 Verify Your Email</h1>
<p className="text-gray-700 mb-3">
We've sent you a verification email. Please check your inbox and click the confirmation link.
</p>
<p className="text-sm text-gray-500">
Once verified, you'll be redirected to your dashboard automatically.
</p>
<button
onClick={resendVerification}
className="mt-6 px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-pink-600 transition"
>
Resend Verification Email
</button>
</div>
</div>
)
}
