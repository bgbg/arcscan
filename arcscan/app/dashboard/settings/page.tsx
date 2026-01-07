'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export default function Settings() {
  const [settings, setSettings] = useState({
    emailNotifications: true,
    darkMode: false,
    autoAnalysis: false,
  })

  const handleToggle = (setting: keyof typeof settings) => {
    setSettings(prevState => ({ ...prevState, [setting]: !prevState[setting] }))
  }

  const handleSave = () => {
    // Implement settings save logic here
    console.log('Saved settings:', settings)
  }

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Settings</h1>
      <Card>
        <CardHeader>
          <CardTitle>Application Settings</CardTitle>
          <CardDescription>Customize your ArcScan experience</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="email-notifications">Email Notifications</Label>
              <p className="text-sm text-gray-500">Receive email updates about your analyses</p>
            </div>
            <Switch
              id="email-notifications"
              checked={settings.emailNotifications}
              onCheckedChange={() => handleToggle('emailNotifications')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="dark-mode">Dark Mode</Label>
              <p className="text-sm text-gray-500">Use dark theme for the application</p>
            </div>
            <Switch
              id="dark-mode"
              checked={settings.darkMode}
              onCheckedChange={() => handleToggle('darkMode')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="auto-analysis">Auto Analysis</Label>
              <p className="text-sm text-gray-500">Automatically start analysis when video is uploaded</p>
            </div>
            <Switch
              id="auto-analysis"
              checked={settings.autoAnalysis}
              onCheckedChange={() => handleToggle('autoAnalysis')}
            />
          </div>
          <Button onClick={handleSave}>Save Settings</Button>
        </CardContent>
      </Card>
    </div>
  )
}

