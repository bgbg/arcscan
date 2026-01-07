"use client";

import { useState } from "react";
import { signInWithPopup } from "firebase/auth";
import { auth, googleProvider } from "@/lib/firebaseConfig";

export default function LoginButton() {
  const [userName, setUserName] = useState<string | null>(null);

  const handleLogin = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      console.log("User:", user.displayName);
      setUserName(user.displayName);
    } catch (error) {
      console.error("Login failed", error);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <button
        onClick={handleLogin}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Sign in with Google
      </button>

      {userName && (
        <p className="text-lg font-medium text-green-600">
          👋 Welcome, {userName}!
        </p>
      )}
    </div>
  );
}
