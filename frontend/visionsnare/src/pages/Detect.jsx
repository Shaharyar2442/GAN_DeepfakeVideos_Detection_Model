import { useState, useRef, useCallback, useEffect } from 'react'
import styles from './Detect.module.css'
import { fetchHistory, addHistory } from '../utils/api'

const STEPS = [
  'Smart frame sampling (optical flow)',
  'Face detection & alignment (MTCNN)',
  'Spatial NPR feature extraction (2×2)',
  'CNN-LSTM temporal aggregation',
  'Generating confidence score',
]

const API_URL = '/api/predict'

export default function Detect({ currentUser }) {
  const [file,     setFile]     = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [drag,     setDrag]     = useState(false)
  const [phase,    setPhase]    = useState('idle')
  const [stepIdx,  setStepIdx]  = useState(0)
  const [progress, setProgress] = useState(0)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)
  const [history,  setHistory]  = useState([])
  const [histLoading, setHistLoading] = useState(true)
  const inputRef    = useRef()
  const intervalRef = useRef(null)

  /* ── Load history from MongoDB on mount ── */
  useEffect(() => {
    fetchHistory().then(entries => {
      setHistory(entries)
      setHistLoading(false)
    })
  }, [currentUser])

  const handleFile = f => {
    if (!f || !f.type.startsWith('video/')) return
    setFile(f)
    setVideoUrl(URL.createObjectURL(f))
    setPhase('idle'); setResult(null); setProgress(0); setStepIdx(0); setError(null)
  }

  const onDrop = useCallback(e => {
    e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0])
  }, [])

  /* ── Start progress animation (runs while backend processes) ── */
  const startProgressAnimation = () => {
    let p = 0, s = 0
    intervalRef.current = setInterval(() => {
      // Slow down near 90% so it doesn't hit 100 before backend responds
      const increment = p < 70 ? (Math.random() * 5 + 1.5) : (Math.random() * 1.5 + 0.3)
      p = Math.min(p + increment, 92)
      setProgress(p)
      const ns = Math.min(Math.floor(p / 20), 4)
      if (ns !== s) { s = ns; setStepIdx(ns) }
    }, 300)
  }

  const stopProgressAnimation = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }

  /* ── Real API call to VisionSnare backend ── */
  const analyze = async () => {
    setPhase('analyzing'); setProgress(0); setStepIdx(0); setError(null)
    startProgressAnimation()

    try {
      const formData = new FormData()
      formData.append('video', file)

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      stopProgressAnimation()

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error (${response.status})`)
      }

      const data = await response.json()

      // Complete the progress bar
      setProgress(100); setStepIdx(4)

      // Short delay so the user sees 100% before the result card appears
      setTimeout(async () => {
        const res = {
          verdict:    data.verdict,
          confidence: data.confidence,
          raw_score:  data.raw_score,
        }
        setResult(res)
        setPhase('done')

        // Build new history entry
        const entry = {
          id:         Date.now(),
          filename:   file.name,
          date:       new Date().toISOString().split('T')[0],
          size:       (file.size / 1024 / 1024).toFixed(1) + ' MB',
          verdict:    res.verdict,
          confidence: res.confidence,
          duration:   '—',
        }

        // Optimistic update in UI
        setHistory(prev => [entry, ...prev])

        // Persist to MongoDB Atlas (non-blocking)
        await addHistory(entry)
      }, 600)

    } catch (err) {
      stopProgressAnimation()
      setPhase('idle')
      setError(err.message || 'An unexpected error occurred. Is the backend running?')
    }
  }

  const reset = () => {
    stopProgressAnimation()
    setFile(null); setVideoUrl(null); setPhase('idle'); setResult(null); setError(null)
  }

  return (
    <div className="page" style={{ padding: '60px 7% 60px' }}>
      <div className="page-header">
        <h2>Video Analysis</h2>
        <p>Upload a video to run VisionSnare's spatio-temporal NPR detection pipeline.</p>
      </div>

      {error && (
        <div className={styles.errorBanner}>
          <span>⚠️</span>
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className={styles.grid}>
        {/* LEFT */}
        <div>
          {!videoUrl ? (
            <div
              className={`${styles.uploadZone} ${drag ? styles.dragOver : ''}`}
              onDragOver={e => { e.preventDefault(); setDrag(true) }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current.click()}
            >
              <div className={styles.uploadIcon}>🎬</div>
              <h3>Drop your video here</h3>
              <p>or click to browse from your computer</p>
              <button className={styles.uploadBtn} onClick={e => { e.stopPropagation(); inputRef.current.click() }}>
                Choose File
              </button>
              <p className={styles.formats}>Supports MP4, MOV, AVI · Max 500 MB</p>
              <input className={styles.fileInput} type="file" accept="video/*" ref={inputRef}
                onChange={e => handleFile(e.target.files[0])} />
            </div>
          ) : (
            <div>
              <div className={styles.videoPreview}>
                <video src={videoUrl} controls />
                <span className={styles.previewLabel}>Preview</span>
              </div>
              <div className={styles.videoName}>
                <span>📄</span>
                <span><strong>{file.name}</strong> · {(file.size / 1024 / 1024).toFixed(1)} MB</span>
                <button className={styles.removeBtn} onClick={reset}>Remove ×</button>
              </div>
              <button className={styles.analyzeBtn} onClick={analyze} disabled={phase === 'analyzing'}>
                {phase === 'analyzing' ? 'Running pipeline…' : 'Run VisionSnare Detection'}
              </button>
            </div>
          )}
        </div>

        {/* RIGHT */}
        <div>
          {phase === 'idle' && !result && (
            <div className={styles.emptyState}>
              <div>🔍</div>
              <p>Upload a video on the left<br />to begin the detection pipeline.</p>
            </div>
          )}

          {phase === 'analyzing' && (
            <div className={styles.progressCard}>
              <h4>Running VisionSnare pipeline…</h4>
              <div className={styles.steps}>
                {STEPS.map((s, i) => (
                  <div className={styles.stepRow} key={s}>
                    <div className={`${styles.stepIcon} ${i < stepIdx ? styles.done : i === stepIdx ? styles.active : ''}`}>
                      {i < stepIdx ? '✓' : i + 1}
                    </div>
                    <span className={`${styles.stepLabel} ${i === stepIdx ? styles.stepActive : ''}`}>{s}</span>
                  </div>
                ))}
              </div>
              <div className={styles.progressBarWrap}>
                <div className={styles.progressBarFill} style={{ width: progress + '%' }} />
              </div>
              <div className={styles.progressPct}>{Math.round(progress)}%</div>
            </div>
          )}

          {result && (
            <div className={styles.resultCard}>
              <div className={`${styles.verdict} ${styles[result.verdict]}`}>
                <div className={styles.verdictIcon}>{result.verdict === 'fake' ? '⚠️' : '✅'}</div>
                <div>
                  <div className={styles.verdictLabel}>
                    {result.verdict === 'fake' ? 'Deepfake Detected' : 'Authentic Video'}
                  </div>
                  <div className={styles.verdictSub}>
                    {result.verdict === 'fake'
                      ? 'Spatio-temporal NPR artifacts detected — likely synthetic.'
                      : 'No significant NPR artifacts detected across frames.'}
                  </div>
                </div>
              </div>

              <div className={styles.confidenceSection}>
                <div className={styles.confidenceLabel}>
                  <span>Confidence Score</span><span>{result.confidence}%</span>
                </div>
                <div className={styles.gaugeBg}>
                  <div className={`${styles.gaugeFill} ${styles[result.verdict]}`}
                    style={{ width: result.confidence + '%' }} />
                </div>
              </div>

              <div className={styles.metaRow}>
                <div className={styles.metaItem}><label>File</label><span style={{ fontSize: '0.78rem', wordBreak: 'break-all' }}>{file?.name}</span></div>
                <div className={styles.metaItem}><label>Size</label><span>{file ? (file.size / 1024 / 1024).toFixed(1) + ' MB' : '—'}</span></div>
                <div className={styles.metaItem}><label>Model</label><span>CNN-LSTM + Attention</span></div>
                <div className={styles.metaItem}><label>NPR Grids</label><span>2×2</span></div>
              </div>

              <button className={styles.resetBtn} onClick={reset}>Analyze Another Video</button>
            </div>
          )}
        </div>
      </div>

      {/* ── Detection History ── */}
      <div style={{ marginTop: '3.5rem' }}>
        <h3 style={{ fontFamily: 'var(--serif)', fontSize: '1.5rem', color: 'var(--navy)', marginBottom: '0.5rem' }}>
          Detection History
        </h3>
        <p style={{ color: 'var(--muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
          Showing history for <strong>{currentUser?.username}</strong> — saved to the cloud.
        </p>

        {histLoading ? (
          <p style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>Loading history…</p>
        ) : history.length === 0 ? (
          <p style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>No detections yet. Run your first analysis above!</p>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Filename</th><th>Date</th><th>Size</th><th>Duration</th><th>Verdict</th><th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {history.map(r => (
                <tr key={r.id}>
                  <td style={{ fontWeight: 600, color: 'var(--navy)' }}>{r.filename}</td>
                  <td>{r.date}</td><td>{r.size}</td><td>{r.duration}</td>
                  <td>
                    <span className={`${styles.pill} ${styles[r.verdict]}`}>
                      {r.verdict === 'fake' ? '⚠ Deepfake' : '✓ Authentic'}
                    </span>
                  </td>
                  <td style={{ fontWeight: 600 }}>{r.confidence}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
