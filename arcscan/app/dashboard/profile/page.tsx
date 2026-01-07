'use client'


import React, { useState, useEffect } from 'react';
import { User, Mail, Shield, Settings, AlertTriangle, Camera, Eye, EyeOff, Check, X } from 'lucide-react';
import { getAuth, updateProfile, updatePassword, EmailAuthProvider, reauthenticateWithCredential } from "firebase/auth";
import { RefreshCw } from "lucide-react";
import { sendEmailVerification } from "firebase/auth";
import ProtectedRoute from '@/components/ProtectedRoute';

export default function ArcScanProfilePage() {
const [isEditing, setIsEditing] = useState(false);
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [profileData, setProfileData] = useState({
    displayName: "",
    fullName: "", // لو عندك Full Name في Firestore ممكن تضيفيه هنا
    email: "",
    emailVerified: false,
    accountCreated: ""
  });

  const [editData, setEditData] = useState({
    displayName: "",
    fullName: ""
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: ""
  });

  // جلب بيانات المستخدم عند تحميل الصفحة
  useEffect(() => {
    const fetchProfile = async () => {
      const auth = getAuth();
      const user = auth.currentUser;
      if (user) {
        await user.reload(); // تأكد تجيب أحدث حالة
        let formattedDate = "";
if (user.metadata.creationTime) {
  formattedDate = new Date(user.metadata.creationTime).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric"
  });
}

        setProfileData({
          displayName: user.displayName || "",
          fullName: "", // لو بدك تجيبيه من Firestore
          email: user.email || "",
          emailVerified: user.emailVerified,
accountCreated: formattedDate
        });
        setEditData({
          displayName: user.displayName || "",
          fullName: ""
        });
      }
    };

    fetchProfile();
  }, []);

  // تحديث Display Name
  const handleSaveProfile = async () => {
    const auth = getAuth();
    const user = auth.currentUser;
    if (user) {
      await updateProfile(user, {
        displayName: editData.displayName
      });
      setProfileData(prev => ({
        ...prev,
        displayName: editData.displayName
      }));
      setIsEditing(false);
    }
  };

  const handleCancelEdit = () => {
    setEditData({
      displayName: profileData.displayName,
      fullName: profileData.fullName
    });
    setIsEditing(false);
  };

  // تحديث حالة التحقق
  const refreshEmailVerification = async () => {
    const auth = getAuth();
    const user = auth.currentUser;
    if (user) {
      await user.reload();
      setProfileData(prev => ({
        ...prev,
        emailVerified: user.emailVerified
      }));
    }
  };

  // تغيير كلمة المرور
  const handlePasswordChange = async () => {
    const auth = getAuth();
    const user = auth.currentUser;
    if (user) {
      const credential = EmailAuthProvider.credential(
        user.email || "",
        passwordData.currentPassword
      );
      try {
        await reauthenticateWithCredential(user, credential);
        await updatePassword(user, passwordData.newPassword);
        setShowPasswordModal(false);
        setPasswordData({
          currentPassword: "",
          newPassword: "",
          confirmPassword: ""
        });
        alert("Password updated successfully!");
      } catch (error) {
        console.error("Error updating password:", error);
        alert("Failed to update password. Please check your current password.");
      }
    }
  };

  const handleResendVerification = async () => {
        const auth = getAuth();

  const user = auth.currentUser;
  if (user) {
    await sendEmailVerification(user);
    alert("Verification email sent again.");
  }
};


  // حذف الحساب
  const handleDeleteAccount = async () => {
    const auth = getAuth();
    const user = auth.currentUser;
    if (user) {
      try {
        await user.delete();
        alert("Account deleted.");
        // هنا تقدرِ توجهي المستخدم للصفحة الرئيسية
      } catch (error) {
        console.error("Error deleting account:", error);
        alert("Failed to delete account.");
      }
    }
  };

  


return (
  <ProtectedRoute>
<div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50/20 p-6">
<div className="max-w-4xl mx-auto">
{/* Header Section */}
<div className="mb-8">
<h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-500 bg-clip-text text-transparent mb-2">
Your Profile
</h1>
<p className="text-slate-600 text-lg">
Manage your personal information and preferences.
</p>
</div>

<div className="grid grid-cols-1 gap-6 max-w-xl mx-auto">
{/* Profile Overview Card */}
<div className="lg:col-span-2">
  <div className="bg-white/90 rounded-2xl shadow-md border border-slate-200/70 p-5">
    <div className="flex flex-col sm:flex-row items-center gap-6">
      <div className="relative">
        <div className="w-24 h-24 bg-gradient-to-r from-purple-600 to-pink-500 rounded-full flex items-center justify-center shadow-lg">
          <User className="w-12 h-12 text-white" />
        </div>
      </div>

      <div className="flex-1 text-center sm:text-left">
        <h2 className="text-2xl font-bold text-slate-800 mb-1">
          {profileData.displayName}
        </h2>

        <div className="flex flex-col sm:flex-row sm:items-center gap-2 mb-2 flex-wrap">
          <div className="flex items-center gap-2">
            <Mail className="w-4 h-4 text-slate-500" />
            <span className="text-slate-600">{profileData.email}</span>
          </div>

          {profileData.emailVerified ? (
            <div className="flex items-center gap-1">
              <Check className="w-4 h-4 text-green-500" />
              <span className="text-green-600 text-sm font-medium">Verified</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-1">
                <X className="w-4 h-4 text-red-500" />
                <span className="text-red-600 text-sm font-medium">Not Verified</span>
              </div>
              <button
                onClick={handleResendVerification}
                className="text-purple-600 text-xs underline hover:text-purple-800"
              >
                Resend Email
              </button>
           <button
  onClick={refreshEmailVerification}
  className="flex items-center gap-1 text-slate-500 hover:text-purple-600 text-xs"
>
  <RefreshCw className="w-4 h-4" />
  Refresh
</button>

            </div>
          )}
        </div>

        <p className="text-slate-500 text-sm">
          Member since {profileData.accountCreated}
        </p>
      </div>
    </div>
  </div>
</div>


{/* Personal Info Section */}
<div className="bg-white/90 rounded-2xl shadow-md border border-slate-200/70 p-5 max-w-xl mx-auto">
  <div className="flex items-center gap-2 mb-4">
    <User className="w-5 h-5 text-purple-600" />
    <h3 className="text-xl font-bold text-slate-800">Personal Info</h3>
  </div>

  <div className="space-y-4">
    {/* Display Name */}
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">Display Name</label>
      <input
        type="text"
        value={isEditing ? editData.displayName : profileData.displayName}
        onChange={(e) => setEditData({ ...editData, displayName: e.target.value })}
        disabled={!isEditing}
        className="w-full px-3 py-2 border border-slate-300 rounded-md shadow-inner focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 disabled:bg-slate-50 disabled:text-slate-500"
      />
    </div>

 

    {/* Email */}
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">Email</label>
      <input
        type="email"
        value={profileData.email}
        disabled
        className="w-full px-3 py-2 border border-slate-300 rounded-md shadow-inner bg-slate-50 text-slate-500"
      />
    </div>

    {/* Change Password */}
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
      <button
        onClick={() => setShowPasswordModal(true)}
        className="text-purple-600 hover:text-purple-700 font-medium text-sm"
      >
        Change Password
      </button>
    </div>

    {/* Buttons */}
    <div className="flex gap-2">
      {isEditing ? (
        <>
          <button
            onClick={handleSaveProfile}
            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-500 text-white px-4 py-2 rounded-md font-medium hover:from-purple-700 hover:to-pink-600 transition-all duration-200 transform hover:scale-105"
          >
            Save Changes
          </button>
          <button
            onClick={handleCancelEdit}
            className="px-4 py-2 border border-slate-300 rounded-md text-slate-600 hover:bg-slate-50 transition-colors"
          >
            Cancel
          </button>
        </>
      ) : (
        <button
          onClick={() => setIsEditing(true)}
          className="flex-1 bg-gradient-to-r from-purple-600 to-pink-500 text-white px-4 py-2 rounded-md font-medium hover:from-purple-700 hover:to-pink-600 transition-all duration-200 transform hover:scale-105"
        >
          Edit Profile
        </button>
      )}
    </div>

    {/* Danger Zone */}
    <div className="mt-8 border-t border-slate-200 pt-6">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="w-5 h-5 text-red-600" />
        <h3 className="text-xl font-bold text-red-800">Danger Zone</h3>
      </div>

      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <h4 className="font-medium text-red-800 mb-2">Delete Account</h4>
        <p className="text-red-600 text-sm mb-4">
          Once you delete your account, there is no going back. Please be certain.
        </p>
        <button
          onClick={() => setShowDeleteModal(true)}
          className="bg-red-600 text-white px-4 py-2 rounded-md font-medium hover:bg-red-700 transition-colors"
        >
          Delete Account
        </button>
      </div>
    </div>
  </div>
</div>





</div>
</div>

{/* Password Change Modal */}
{showPasswordModal && (
<div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
<div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
<div className="flex items-center gap-2 mb-4">
<Shield className="w-5 h-5 text-purple-600" />
<h3 className="text-xl font-bold text-slate-800">Change Password</h3>
</div>

<div className="space-y-4">
<div>
<label className="block text-sm font-medium text-slate-700 mb-2">
Current Password
</label>
<div className="relative">
<input
type={showCurrentPassword ? "text" : "password"}
value={passwordData.currentPassword}
onChange={(e) => setPasswordData({...passwordData, currentPassword: e.target.value})}
className="w-full px-3 py-2 pr-10 border border-slate-300 rounded-md shadow-inner focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
/>
<button
type="button"
onClick={() => setShowCurrentPassword(!showCurrentPassword)}
className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600"
>
{showCurrentPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
</button>
</div>
</div>

<div>
<label className="block text-sm font-medium text-slate-700 mb-2">
New Password
</label>
<div className="relative">
<input
type={showNewPassword ? "text" : "password"}
value={passwordData.newPassword}
onChange={(e) => setPasswordData({...passwordData, newPassword: e.target.value})}
className="w-full px-3 py-2 pr-10 border border-slate-300 rounded-md shadow-inner focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
/>
<button
type="button"
onClick={() => setShowNewPassword(!showNewPassword)}
className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600"
>
{showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
</button>
</div>
</div>

<div>
<label className="block text-sm font-medium text-slate-700 mb-2">
Confirm New Password
</label>
<div className="relative">
<input
type={showConfirmPassword ? "text" : "password"}
value={passwordData.confirmPassword}
onChange={(e) => setPasswordData({...passwordData, confirmPassword: e.target.value})}
className="w-full px-3 py-2 pr-10 border border-slate-300 rounded-md shadow-inner focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
/>
<button
type="button"
onClick={() => setShowConfirmPassword(!showConfirmPassword)}
className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600"
>
{showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
</button>
</div>
</div>
</div>

<div className="flex gap-3 mt-6">
<button
onClick={handlePasswordChange}
className="flex-1 bg-gradient-to-r from-purple-600 to-pink-500 text-white px-4 py-2 rounded-md font-medium hover:from-purple-700 hover:to-pink-600 transition-all duration-200"
>
Update Password
</button>
<button
onClick={() => setShowPasswordModal(false)}
className="px-4 py-2 border border-slate-300 rounded-md text-slate-600 hover:bg-slate-50 transition-colors"
>
Cancel
</button>
</div>
</div>
</div>
)}

{/* Delete Account Modal */}
{showDeleteModal && (
<div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
<div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
<div className="flex items-center gap-2 mb-4">
<AlertTriangle className="w-5 h-5 text-red-600" />
<h3 className="text-xl font-bold text-red-800">Delete Account</h3>
</div>

<p className="text-slate-600 mb-6">
Are you absolutely sure you want to delete your account? This action cannot be undone and will permanently remove all your data.
</p>

<div className="flex gap-3">
<button
onClick={handleDeleteAccount}
className="flex-1 bg-red-600 text-white px-4 py-2 rounded-md font-medium hover:bg-red-700 transition-colors"
>
Yes, Delete Account
</button>
<button
onClick={() => setShowDeleteModal(false)}
className="px-4 py-2 border border-slate-300 rounded-md text-slate-600 hover:bg-slate-50 transition-colors"
>
Cancel
</button>
</div>
</div>
</div>
)}
</div>
</ProtectedRoute>
);
}