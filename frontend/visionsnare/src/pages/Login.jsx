import { useState } from 'react'
import { login } from '../utils/api'
import styles from './Login.module.css'

export default function Login({ onAuth, goToRegister }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const handleSubmit = async e => {
    e.preventDefault()
    if (!username.trim() || !password) return
    setLoading(true)
    setError('')
    try {
      const { username: user } = await login(username.trim(), password)
      onAuth(user)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      {/* Animated background orbs */}
      <div className={styles.orb1} />
      <div className={styles.orb2} />
      <div className={styles.orb3} />

      <div className={styles.card}>
        {/* Logo / brand */}
        <div className={styles.brand}>
          <div className={styles.brandIcon}>
            <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="20" cy="20" r="18" stroke="url(#g1)" strokeWidth="2"/>
              <path d="M12 20 L18 14 L24 20 L18 26 Z" fill="url(#g1)" fillOpacity="0.9"/>
              <path d="M20 14 L28 20 L20 26" stroke="url(#g2)" strokeWidth="1.5" fill="none" strokeLinejoin="round"/>
              <defs>
                <linearGradient id="g1" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#6366f1"/>
                  <stop offset="1" stopColor="#3b82f6"/>
                </linearGradient>
                <linearGradient id="g2" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#a78bfa"/>
                  <stop offset="1" stopColor="#60a5fa"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h1 className={styles.brandName}>VisionSnare</h1>
          <p className={styles.brandTagline}>Deepfake Detection Platform</p>
        </div>

        <h2 className={styles.title}>Welcome back</h2>
        <p className={styles.subtitle}>Sign in to access your detection history</p>

        {error && (
          <div className={styles.errorBox}>
            <span className={styles.errorIcon}>⚠</span>
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className={styles.form} noValidate>
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor="login-username">Username</label>
            <div className={styles.inputWrap}>
              <span className={styles.inputIcon}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                </svg>
              </span>
              <input
                id="login-username"
                type="text"
                className={styles.input}
                placeholder="Enter your username"
                value={username}
                onChange={e => setUsername(e.target.value)}
                autoComplete="username"
                required
              />
            </div>
          </div>

          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor="login-password">Password</label>
            <div className={styles.inputWrap}>
              <span className={styles.inputIcon}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
              </span>
              <input
                id="login-password"
                type="password"
                className={styles.input}
                placeholder="Enter your password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                autoComplete="current-password"
                required
              />
            </div>
          </div>

          <button
            id="login-submit-btn"
            type="submit"
            className={styles.submitBtn}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className={styles.spinner} />
                Signing in…
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <p className={styles.switchText}>
          Don't have an account?{' '}
          <button className={styles.switchLink} onClick={goToRegister}>
            Create one
          </button>
        </p>
      </div>
    </div>
  )
}
