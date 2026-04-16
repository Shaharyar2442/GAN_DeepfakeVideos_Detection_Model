import { useState } from 'react'
import styles from './HowItWorks.module.css'

const STEPS = [
  {
    icon: '🎬', tag: 'Step 1 · Upload', title: 'Upload Your Video',
    simple: 'Just drag and drop any video file — or click to browse from your computer. We support MP4, MOV, and AVI files up to 500 MB. That is all you need to do.',
    model: 'Behind the scenes, the system validates the file format and prepares it for the detection pipeline. Supported formats: MP4, MOV, AVI (up to 500 MB).',
    tech: ['MP4', 'MOV', 'AVI', 'Max 500 MB'],
  },
  {
    icon: '🔎', tag: 'Step 2 · Frame Extraction', title: 'We Pick the Most Important Frames',
    simple: 'Not every frame in a video tells us something useful. VisionSnare automatically identifies the moments with the most movement — because that is where deepfake glitches are most likely to appear.',
    model: 'An optical flow algorithm computes motion scores between consecutive frames. Only frames exceeding a motion threshold are selected. MTCNN then detects and aligns facial landmarks using affine transformations, normalising face position and scale across all selected frames.',
    tech: ['Optical Flow', 'MTCNN', 'Face Alignment', 'OpenCV'],
  },
  {
    icon: '🧩', tag: 'Step 3 · Artifact Detection', title: 'Spotting the Traces AI Leaves Behind',
    simple: 'When AI generates a fake face, it leaves behind tiny invisible patterns in the pixels — almost like a digital fingerprint. Our model is trained to find these patterns, even when they are too subtle for the human eye.',
    model: 'Each face frame is processed through Neighboring Pixel Relationship (NPR) analysis using 2×2 non-overlapping grids — capturing upsampling artifacts left by deepfake generators. A temporal NPR map is computed as the absolute pixel-wise difference between consecutive spatial NPR frames, capturing inter-frame inconsistencies. Spatial and temporal NPR maps are concatenated into a 6-channel spatio-temporal tensor.',
    tech: ['NPR 2×2', 'Temporal NPR', '6-Channel Tensor'],
  },
  {
    icon: '🤖', tag: 'Step 4 · AI Analysis', title: 'Our Model Watches the Video Over Time',
    simple: 'A real face moves naturally and consistently from frame to frame. A deepfake often has tiny inconsistencies between frames — moments where the AI slips up. Our model watches the whole sequence to catch these slip-ups.',
    model: 'The 6-channel tensor is passed through a lightweight custom ResNet CNN (truncated at layer2) to extract a 512-dimensional feature vector per frame. An attention mechanism assigns softmax importance weights to each frame, focusing on artifact-rich moments. A sequence of 11 weighted feature vectors (12 frames minus 1 for temporal differencing) is fed through an LSTM network that models short and long-term temporal dependencies.',
    tech: ['Custom ResNet', 'Attention Mechanism', 'LSTM', '12-frame Sequences', 'PyTorch'],
  },
  {
    icon: '📊', tag: 'Step 5 · Result', title: 'You Get a Clear Verdict',
    simple: 'Within seconds you get a simple result — Real or Deepfake — along with a confidence score showing how certain the model is. No technical knowledge needed to understand it.',
    model: 'The LSTM output is passed to a fully connected binary classifier producing a probability score. Threshold 0.5 determines the Real/Fake label. The model achieved 99.46% in-domain accuracy (FakeAVCeleb) and 70.64% cross-domain (FaceForensics).',
    tech: ['Binary Classifier', 'Confidence Score', '99.46% In-Domain', '70.64% Cross-Domain'],
  },
]

export default function HowItWorks() {
  const [expanded, setExpanded] = useState({})
  const toggle = i => setExpanded(prev => ({ ...prev, [i]: !prev[i] }))

  return (
    <div className="page" style={{ padding: '60px 7% 60px' }}>
      <div className="page-header">
        <h2>How It Works</h2>
        <p>Simple for anyone to follow — click "Under the hood" on any step to see the technical details.</p>
      </div>

      {STEPS.map((s, i) => (
        <div className={styles.step} key={i}>
          <div className={styles.iconWrap}>
            <div className={styles.icon}>{s.icon}</div>
          </div>
          <div className={styles.content}>
            <div className={styles.tag}>{s.tag}</div>
            <h3 className={styles.title}>{s.title}</h3>
            <p className={styles.simple}>{s.simple}</p>
            <button className={styles.toggle} onClick={() => toggle(i)}>
              {expanded[i] ? '▲ Hide technical details' : '▼ Under the hood'}
            </button>
            {expanded[i] && (
              <div className={styles.modelBox}>
                <div className={styles.modelLabel}>Technical Detail</div>
                <p className={styles.modelText}>{s.model}</p>
                <div className={styles.techPills}>
                  {s.tech.map(t => <span key={t}>{t}</span>)}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
