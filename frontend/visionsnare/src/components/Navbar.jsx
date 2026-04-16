import logo from '/logo.png'
import styles from './Navbar.module.css'
import { logout } from '../utils/api.js'

const LINKS = ['Home', 'Detect', 'How It Works', 'Pricing', 'About']

export default function Navbar({ page, setPage, currentUser, onLogout }) {
  return (
    <nav className={styles.nav}>
      <div className={styles.logo} onClick={() => setPage('Home')}>
        <img src={logo} alt="VisionSnare" />
        <span>VisionSnare</span>
      </div>

      <ul className={styles.links}>
        {LINKS.map(l => (
          <li key={l}>
            <button
              className={`${styles.link} ${page === l ? styles.active : ''} ${l === 'Detect' ? styles.cta : ''}`}
              onClick={() => setPage(l)}
            >
              {l}
            </button>
          </li>
        ))}
      </ul>

      {/* User chip + logout */}
      {currentUser && (
        <div className={styles.userArea}>
          <div className={styles.userChip}>
            <div className={styles.avatar}>
              {currentUser.username.charAt(0).toUpperCase()}
            </div>
            <span className={styles.username}>{currentUser.username}</span>
          </div>
          <button
            id="logout-btn"
            className={styles.logoutBtn}
            onClick={onLogout}
            title="Sign out"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
              <polyline points="16 17 21 12 16 7"/>
              <line x1="21" y1="12" x2="9" y2="12"/>
            </svg>
            <span className={styles.logoutLabel}>Logout</span>
          </button>
        </div>
      )}
    </nav>
  )
}
