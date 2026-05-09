/**
 * api.js — VisionSnare auth + history API helpers
 *
 * All calls go through Vite's proxy to http://localhost:8000
 * The JWT token is stored in localStorage under "vs_token".
 */

const BASE = '/api'

// ─── Token helpers ────────────────────────────────────────────────────────

export function getToken() {
  return localStorage.getItem('vs_token')
}

export function getCurrentUser() {
  const raw = localStorage.getItem('vs_user')
  return raw ? JSON.parse(raw) : null
}

function saveSession(token, username) {
  localStorage.setItem('vs_token', token)
  localStorage.setItem('vs_user', JSON.stringify({ username }))
}

export function logout() {
  localStorage.removeItem('vs_token')
  localStorage.removeItem('vs_user')
}

// ─── Auth headers ─────────────────────────────────────────────────────────

function authHeaders() {
  const token = getToken()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

// ─── Auth routes ──────────────────────────────────────────────────────────

/**
 * Register a new user. Returns { username } on success.
 * Throws an Error with a user-readable message on failure.
 */
export async function register(username, password) {
  const res = await fetch(`${BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Registration failed.')
  saveSession(data.token, data.username)
  return { username: data.username }
}

/**
 * Login an existing user. Returns { username } on success.
 * Throws an Error with a user-readable message on failure.
 */
export async function login(username, password) {
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Login failed.')
  saveSession(data.token, data.username)
  return { username: data.username }
}

/**
 * Validate the stored JWT and return the current user.
 * Returns null if the token is missing or invalid.
 */
export async function validateSession() {
  const token = getToken()
  if (!token) return null
  try {
    const res = await fetch(`${BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!res.ok) { logout(); return null }
    const data = await res.json()
    return { username: data.username }
  } catch {
    return null
  }
}

// ─── History routes ───────────────────────────────────────────────────────

/**
 * Fetch the logged-in user's detection history from MongoDB Atlas.
 * Returns an array of history entry objects.
 */
export async function fetchHistory() {
  const res = await fetch(`${BASE}/history`, { headers: authHeaders() })
  if (!res.ok) return []
  const data = await res.json()
  return data.history || []
}

/**
 * Persist a new detection entry to MongoDB Atlas.
 * @param {object} entry — { filename, date, size, verdict, confidence, duration }
 */
export async function addHistory(entry) {
  try {
    await fetch(`${BASE}/history`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(entry),
    })
  } catch {
    // Non-critical — fail silently so the user still sees the result
  }
}
