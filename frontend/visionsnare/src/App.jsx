import { useState, useEffect } from 'react'
import Navbar    from './components/Navbar.jsx'
import Home      from './pages/Home.jsx'
import Detect    from './pages/Detect.jsx'
import HowItWorks from './pages/HowItWorks.jsx'
import Pricing   from './pages/Pricing.jsx'
import About     from './pages/About.jsx'
import Login     from './pages/Login.jsx'
import Register  from './pages/Register.jsx'
import { validateSession, logout } from './utils/api.js'

export default function App() {
  const [page,        setPage]        = useState('Home')
  const [authScreen,  setAuthScreen]  = useState('login')  // 'login' | 'register'
  const [currentUser, setCurrentUser] = useState(null)    // null = not logged in
  const [authLoading, setAuthLoading] = useState(true)    // shows nothing while checking token

  /* ── On mount: silently validate stored JWT ── */
  useEffect(() => {
    validateSession().then(user => {
      setCurrentUser(user)
      setAuthLoading(false)
    })
  }, [])

  /* ── Called after successful login or register ── */
  const handleAuth = username => {
    setCurrentUser({ username })
    setPage('Home')
  }

  /* ── Logout ── */
  const handleLogout = () => {
    logout()
    setCurrentUser(null)
    setPage('Home')
    setAuthScreen('login')
  }

  /* ── While checking stored token, show a minimal dark loader ── */
  if (authLoading) {
    return (
      <div style={{
        minHeight: '100vh', background: '#060b18',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div style={{
          width: 36, height: 36,
          border: '3px solid rgba(99,102,241,0.25)',
          borderTopColor: '#6366f1',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
        }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    )
  }

  /* ── Not logged in → show auth screens ── */
  if (!currentUser) {
    return authScreen === 'login'
      ? <Login onAuth={handleAuth} goToRegister={() => setAuthScreen('register')} />
      : <Register onAuth={handleAuth} goToLogin={() => setAuthScreen('login')} />
  }

  /* ── Logged in → full app ── */
  const render = () => {
    switch (page) {
      case 'Home':         return <Home setPage={setPage} />
      case 'Detect':       return <Detect currentUser={currentUser} />
      case 'How It Works': return <HowItWorks />
      case 'Pricing':      return <Pricing />
      case 'About':        return <About />
      default:             return <Home setPage={setPage} />
    }
  }

  return (
    <>
      <Navbar
        page={page}
        setPage={setPage}
        currentUser={currentUser}
        onLogout={handleLogout}
      />
      {render()}
    </>
  )
}
