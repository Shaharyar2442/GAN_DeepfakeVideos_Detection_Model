import { useState } from 'react'
import styles from './About.module.css'

const TEAM = [
  { name: 'Shaharyar Rizwan', id: '22I-0999', role: 'Data Pipeline & Evaluation Engineer', desc: 'Preprocessing pipeline, MTCNN alignment, cross-domain testing, results visualisation', emoji: '💻' },
  { name: 'Moazzam Hafeez', id: '22I-1093', role: 'Lead Model Developer & Experimental Setup', desc: 'Designed CNN-LSTM architecture, configured PyTorch/CUDA, executed training loops', emoji: '🧠' },
  { name: 'Fiza Jameel', id: '22I-0964', role: 'Research Lead & Feature Engineering', desc: 'Literature review (18 papers), multi-scale NPR algorithm, dataset curation', emoji: '🔬' },
]

const PIPELINE = [
  { icon: '🎞️', label: 'Frame Sampling', detail: 'Optical flow selects motion-rich frames' },
  { icon: '👤', label: 'Face Alignment', detail: 'MTCNN detects & aligns faces' },
  { icon: '🔬', label: 'Spatial NPR', detail: '2×2 non-overlapping pixel-diff grids' },
  { icon: '⏱️', label: 'Temporal NPR', detail: 'Inter-frame difference maps' },
  { icon: '🧠', label: 'CNN Backbone', detail: 'Custom ResNet feature extractor' },
  { icon: '🎯', label: 'Attention', detail: 'Softmax frame-importance weights' },
  { icon: '🔁', label: 'LSTM', detail: '11-frame temporal aggregation' },
  { icon: '✅', label: 'Verdict', detail: 'Real / Fake + confidence score' },
]

const PROBLEMS = [
  { icon: '👁️', title: 'Hard to see', body: 'Modern deepfakes are indistinguishable to the human eye. Even experts can be fooled by high-quality synthetic videos.' },
  { icon: '📱', title: 'Spreading fast', body: 'Social media platforms compress and share videos at scale. Misinformation travels faster than corrections.' },
  { icon: '🤖', title: 'AI keeps improving', body: 'GAN and diffusion generators get better every year, making artifacts smaller and harder for older detectors to catch.' },
  { icon: '⚠️', title: 'Existing tools fail', body: 'Most detectors only look at single frames and overfit to specific deepfake methods — failing on anything unseen.' },
]

export default function About() {
  const TECH = ['Python', 'PyTorch', 'OpenCV', 'MTCNN', 'Scikit-learn', 'CUDA', 'Ubuntu 24.04']

  return (
    <div className="page" style={{ padding: '60px 7% 80px' }}>

      {/* ── Top intro ── */}
      <div className={styles.top}>
        <div className={styles.topText}>
          <div className={styles.kicker}>Final Year Project · FAST-NUCES Islamabad · 2022–2026</div>
          <h2 className={styles.h2}>About VisionSnare</h2>
          <p>Deepfakes are AI-generated videos where a person's face is convincingly replaced or fabricated entirely. They are becoming harder to spot by eye, and they pose real dangers — spreading misinformation, damaging reputations, and undermining trust in digital media.</p>
          <p>VisionSnare is our answer to that problem. Built as our Final Year Project at FAST-NUCES Islamabad, the core idea is simple: when AI generates a face, it always leaves behind invisible pixel-level traces. Our model is trained to find those traces, even across multiple frames of a video.</p>
          <p>Unlike most existing tools that only inspect individual frames, VisionSnare watches how those traces change <em>over time</em> — catching inconsistencies that a single-frame check would miss entirely.</p>
          <div className={styles.nucesBadge}>
            <span>🎓</span>
            <div>
              <div className={styles.nucesBadgeName}>National University of Computer and Emerging Sciences</div>
              <div className={styles.nucesBadgeSub}>Department of Computer Science · Islamabad, Pakistan</div>
            </div>
          </div>
        </div>

        <div className={styles.topCard}>
          <div className={styles.cardLabel}>Experimental Results</div>
          <div className={styles.results}>
            {[
              ['In-domain Accuracy (FakeAVCeleb)', '97.68%', 'good'],
              ['Cross-domain (df40)', '93.77%', 'good'],
              ['Cross-domain (Celeb-DF-v1)', '83.43%', 'good'],
              ['Cross-domain (FaceForensics++)', '70.64%', 'warn'],
              ['In-domain Test Loss', '0.0595', 'good'],
              ['Training Epochs', '20', 'good'],
            ].map(([label, val, cls]) => (
              <div key={label} className={styles.resultRow}>
                <span className={styles.resultLabel}>{label}</span>
                <span className={`${styles.resultVal} ${styles[cls]}`}>{val}</span>
              </div>
            ))}
          </div>
          <div className={styles.cardLabel} style={{ marginTop: '1.4rem' }}>Tech Stack</div>
          <div className={styles.techPills}>
            {TECH.map(t => <span key={t} className={styles.techPill}>{t}</span>)}
          </div>
          <div className={styles.cardLabel} style={{ marginTop: '1.2rem' }}>Hardware</div>
          <p className={styles.hw}>Intel Core i7-10700K · NVIDIA RTX 3080 (10GB) · 32GB RAM</p>
        </div>
      </div>

      {/* ── Problem ── */}
      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>The Problem We Are Solving</h3>
        <div className={styles.problemGrid}>
          {PROBLEMS.map(p => (
            <div key={p.title} className={styles.problemCard}>
              <div className={styles.problemIcon}>{p.icon}</div>
              <h4>{p.title}</h4>
              <p>{p.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Approach ── */}
      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>Our Approach — Spatio-Temporal NPR</h3>
        <div className={styles.approachGrid}>
          <div className={styles.approachText}>
            <p>The foundation of VisionSnare is a concept called <strong>Neighboring Pixel Relationships (NPR)</strong>. Here is the key insight: whenever AI generates a face, it uses a process called <em>upsampling</em> to scale the image to full resolution. This always leaves tiny, repeating patterns in how nearby pixels relate to each other — patterns that do not appear in real photographs.</p>
            <p>We compute these patterns by looking at small 2×2 non-overlapping grids of pixels and measuring the differences between neighbouring pixel values. These difference maps are our <strong>"artifact fingerprint"</strong> for that frame.</p>
            <p>But we go further: we also compare these fingerprints <strong>across consecutive frames</strong>. A real face has naturally consistent pixel patterns over time. A deepfake flickers — the AI struggles to keep those patterns stable frame to frame. Our model catches that flickering.</p>
          </div>
          <div className={styles.approachVisual}>
            <div className={styles.visualTitle}>What NPR Sees</div>
            <div className={styles.nprDemo}>
              <div className={styles.nprCol}>
                <div className={styles.nprLabel}>Real frame</div>
                <div className={`${styles.nprGrid}`}>
                  {[...Array(9)].map((_, i) => <div key={i} className={styles.realCell} />)}
                </div>
                <div className={styles.nprCaption}>Consistent pixel patterns</div>
              </div>
              <div className={styles.nprArrow}>→</div>
              <div className={styles.nprCol}>
                <div className={styles.nprLabel}>Deepfake frame</div>
                <div className={styles.nprGrid}>
                  {[...Array(9)].map((_, i) => (
                    <div key={i} className={`${styles.fakeCell} ${i % 3 === 1 || i === 4 ? styles.anomaly : ''}`} />
                  ))}
                </div>
                <div className={styles.nprCaption}>Upsampling artifacts visible</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Architecture ── */}
      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>Model Architecture</h3>
        <p className={styles.archSub}>Our model is a <strong>Hybrid CNN-LSTM with Attention</strong> — a three-stage deep learning pipeline designed to catch both spatial artifacts within frames and temporal inconsistencies across frames.</p>

        <div className={styles.pipeline}>
          {PIPELINE.map((p, i) => (
            <div key={i} className={styles.pipelineItem}>
              <div className={styles.pipelineNode}>
                <div className={styles.pipelineIcon}>{p.icon}</div>
                <div className={styles.pipelineLabel}>{p.label}</div>
                <div className={styles.pipelineDetail}>{p.detail}</div>
              </div>
              {i < PIPELINE.length - 1 && <div className={styles.pipelineArrow}>→</div>}
            </div>
          ))}
        </div>

        <div className={styles.archCards}>
          <div className={styles.archCard}>
            <div className={styles.archCardTitle}>🔬 NPR Feature Maps</div>
            <p>2×2 non-overlapping grids produce spatial artifact maps by subtracting the top-left pixel from its neighbours. A temporal difference map between consecutive NPR frames captures inter-frame inconsistencies. Spatial and temporal maps are concatenated into a <strong>6-channel spatio-temporal tensor</strong> per frame.</p>
          </div>
          <div className={styles.archCard}>
            <div className={styles.archCardTitle}>🧠 CNN Backbone</div>
            <p>A lightweight custom ResNet processes each 6-channel tensor and outputs a compact feature vector per frame. The input layer is modified to accept multi-channel NPR input instead of standard RGB, keeping the model small and efficient.</p>
          </div>
          <div className={styles.archCard}>
            <div className={styles.archCardTitle}>🎯 Attention + LSTM</div>
            <p>An attention mechanism scores each frame by importance using softmax weights — focusing on the most artifact-rich moments. An 11-frame sequence of weighted vectors (12 input frames minus 1 for temporal differencing) is then passed through an LSTM that captures both short-term and long-term temporal patterns.</p>
          </div>
        </div>
      </section>

      {/* ── Datasets ── */}
      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>Training & Evaluation</h3>
        <div className={styles.datasetGrid}>
          <div className={styles.datasetCard}>
            <span className={styles.datasetBadge}>Primary Dataset</span>
            <div className={styles.datasetName}>FakeAVCeleb</div>
            <p>Used for training, validation, and in-domain testing. Contains videos generated by <strong>FSGAN</strong> and <strong>Wav2Lip</strong>, processed into 12-frame sequences. 1,377 test samples.</p>
            <div className={styles.datasetResult}>
              <span>Test Accuracy</span><strong>97.68%</strong>
            </div>
          </div>
          <div className={styles.datasetCard}>
            <span className={`${styles.datasetBadge} ${styles.cross}`}>Cross-Domain</span>
            <div className={styles.datasetName}>DFDC (df40)</div>
            <p>From the Deepfake Detection Challenge, used solely for evaluating generalization capabilities. 353 test samples.</p>
            <div className={styles.datasetResult}>
              <span>Test Accuracy</span><strong>93.77%</strong>
            </div>
          </div>
          <div className={styles.datasetCard}>
            <span className={`${styles.datasetBadge} ${styles.cross}`}>Cross-Domain</span>
            <div className={styles.datasetName}>Celeb-DF-v1</div>
            <p>High quality deepfake dataset used solely for generalization testing to evaluate robustness. 664 test samples.</p>
            <div className={styles.datasetResult}>
              <span>Test Accuracy</span><strong>83.43%</strong>
            </div>
          </div>
          <div className={styles.datasetCard}>
            <span className={`${styles.datasetBadge} ${styles.cross}`}>Cross-Domain</span>
            <div className={styles.datasetName}>FaceForensics++</div>
            <p>Used solely for generalization testing — the model had <strong>never seen this data</strong> during training.</p>
            <div className={styles.datasetResult}>
              <span>Test Accuracy</span><strong>70.64%</strong>
            </div>
          </div>
        </div>
        <div className={styles.genNote}>
          <span>💡</span>
          <p>Achieving <strong>70.64%</strong> on a completely unseen dataset confirms that VisionSnare has learned fundamental properties of synthetic generation — not just memorized training data. The temporal NPR signals generalize across different deepfake methods.</p>
        </div>
      </section>

      {/* ── Team ── */}
      <section className={styles.section}>
        <h3 className={styles.sectionTitle}>The Team</h3>
        <p className={styles.teamSub}>Supervised by <strong>Mr. Arslan Aslam</strong> · Department of Computer Science · FAST-NUCES Islamabad</p>
        <div className={styles.teamGrid}>
          {TEAM.map(m => (
            <div key={m.name} className={styles.teamCard}>
              <div className={styles.teamAvatar}>{m.emoji}</div>
              <div className={styles.teamName}>{m.name}</div>
              <div className={styles.teamId}>{m.id}</div>
              <div className={styles.teamRole}>{m.role}</div>
              <div className={styles.teamDesc}>{m.desc}</div>
            </div>
          ))}
        </div>
        <div className={styles.supervisorCard}>
          <div className={styles.supAvatar}>👨‍🏫</div>
          <div>
            <div className={styles.supLabel}>Project Supervisor</div>
            <div className={styles.supName}>Mr. Arslan Aslam</div>
            <div className={styles.supDept}>Department of Computer Science · FAST-NUCES Islamabad</div>
          </div>
        </div>
      </section>
    </div>
  )
}
