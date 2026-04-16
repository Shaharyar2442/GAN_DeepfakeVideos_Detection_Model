import { useState } from 'react'
import { register } from '../utils/api'
import styles from './Login.module.css'

export default function Register({ onAuth, goToLogin }) {
  const [username, setUsername]     = useState('')
  const [password, setPassword]     = useState('')
  const [confirm,  setConfirm]      = useState('')
  const [loading,  setLoading]      = useState(false)
  const [error,    setError]        = useState('')

  const pwMismatch    = confirm && password !== confirm
  const pwTooShort    = password && password.length < 6
  const usernameBad   = username && username.trim().length < 3
  const canSubmit     = username.trim().length >= 3 && password.length >= 6 && password === confirm && !loading

  const handleSubmit = async e => {
    e.preventDefault()
    if (!canSubmit) return
    setLoading(true)
    setError('')
    try {
      const { username: user } = await register(username.trim(), password)
      onAuth(user)
    } catch (err) {
      setError(err.message || 'Registration failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.orb1} />
      <div className={styles.orb2} />
      <div className={styles.orb3} />

      <div className={styles.card}>
        {/* Brand */}
        <div className={styles.brand}>
          <div className={styles.brandIcon}>
            <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="20" cy="20" r="18" stroke="url(#g1r)" strokeWidth="2"/>
              <path d="M12 20 L18 14 L24 20 L18 26 Z" fill="url(#g1r)" fillOpacity="0.9"/>
              <path d="M20 14 L28 20 L20 26" stroke="url(#g2r)" strokeWidth="1.5" fill="none" strokeLinejoin="round"/>
              <defs>
                <linearGradient id="g1r" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#6366f1"/>
                  <stop offset="1" stopColor="#3b82f6"/>
                </linearGradient>
                <linearGradient id="g2r" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#a78bfa"/>
                  <stop offset="1" stopColor="#60a5fa"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h1 className={styles.brandName}>VisionSnare</h1>
          <p className={styles.brandTagline}>Deepfake Detection Platform</p>
        </div>

        <h2 className={styles.title}>Create account</h2>
        <p className={styles.subtitle}>Start detecting deepfakes and track your history</p>

        {error && (
          <div className={styles.errorBox}>
            <span className={styles.errorIcon}>⚠</span>
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className={styles.form} noValidate>
          {/* Username */}
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor="reg-username">Username</label>
            <div className={styles.inputWrap}>
              <span className={styles.inputIcon}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                </svg>
              </span>
              <input
                id="reg-username"
                type="text"
                className={styles.input}
                placeholder="Choose a username (min 3 chars)"
                value={username}
                onChange={e => setUsername(e.target.value)}
                autoComplete="username"
                required
              />
            </div>
            {usernameBad && <span className={styles.hint}>Username must be at least 3 characters.</span>}
          </div>

          {/* Password */}
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor="reg-password">Password</label>
            <div className={styles.inputWrap}>
              <span className={styles.inputIcon}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
              </span>
              <input
                id="reg-password"
                type="password"
                className={styles.input}
                placeholder="Min 6 characters"
                value={password}
                onChange={e => setPassword(e.target.value)}
                autoComplete="new-password"
                required
              />
            </div>
            {pwTooShort && <span className={styles.hint}>Password must be at least 6 characters.</span>}
          </div>

          {/* Confirm password */}
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor="reg-confirm">Confirm Password</label>
            <div className={styles.inputWrap}>
              <span className={styles.inputIcon}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9 12l2 2 4-4"/><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
              </span>
              <input
                id="reg-confirm"
                type="password"
                className={styles.input}
                placeholder="Re-enter your password"
                value={confirm}
                onChange={e => setConfirm(e.target.value)}
                autoComplete="new-password"
                required
              />
            </div>
            {pwMismatch && <span className={styles.hint}>Passwords do not match.</span>}
          </div>

          <button
            id="register-submit-btn"
            type="submit"
            className={styles.submitBtn}
            disabled={!canSubmit}
          >
            {loading ? (
              <>
                <span className={styles.spinner} />
                Creating account…
              </>
            ) : (
              'Create Account'
            )}
          </button>
        </form>

        <p className={styles.switchText}>
          Already have an account?{' '}
          <button className={styles.switchLink} onClick={goToLogin}>
            Sign in
          </button>
        </p>
      </div>
    </div>
  )
}
