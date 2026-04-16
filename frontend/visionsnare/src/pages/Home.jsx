import styles from './Home.module.css'

export default function Home({ setPage }) {
  return (
    <div className="page">
      <div className={styles.hero}>
        <div className={styles.heroLeft}>
          <div className={styles.tag}>Spatio-Temporal NPR Detection</div>
          <h1 className={styles.h1}>
            Detect <em>deepfakes</em> using AI-powered video analysis
          </h1>
          <p className={styles.sub}>
            VisionSnare analyzes Neighboring Pixel Relationships (NPR) across spatial and
            temporal dimensions using a Hybrid CNN-LSTM architecture to identify synthetic
            media with high accuracy.
          </p>
          <div className={styles.btns}>
            <button className={styles.btnPrimary} onClick={() => setPage('Detect')}>
              Analyze a Video
            </button>
            <button className={styles.btnSecondary} onClick={() => setPage('How It Works')}>
              How It Works
            </button>
          </div>
        </div>

        <div className={styles.heroRight}>
          <div className={styles.visual}>
            <div className={styles.scanRing}>
              <div className={styles.scanInner}>🎭</div>
            </div>
            <div className={styles.visLabel}>Scanning NPR patterns…</div>
            <div className={styles.badge}>
              <span className={styles.badgeDot} />
              <span>In-domain accuracy: 99.46%</span>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.stats}>
        <div className={styles.stat}>
          <div className={styles.statNum}>99.46%</div>
          <div className={styles.statLabel}>In-domain accuracy on FakeAVCeleb dataset</div>
        </div>
        <div className={styles.stat}>
          <div className={styles.statNum}>70.64%</div>
          <div className={styles.statLabel}>Cross-domain accuracy on FaceForensics (unseen)</div>
        </div>
        <div className={styles.stat}>
          <div className={styles.statNum}>CNN-LSTM</div>
          <div className={styles.statLabel}>Hybrid architecture with attention mechanism</div>
        </div>
      </div>
    </div>
  )
}
