# VisionSnare — Deepfake Detection Frontend

A React + Vite frontend for the VisionSnare deepfake detection system.

## Project Structure

```
visionsnare/
├── public/
│   └── logo.png               ← VisionSnare logo
├── src/
│   ├── components/
│   │   ├── Navbar.jsx
│   │   └── Navbar.module.css
│   ├── pages/
│   │   ├── Home.jsx / Home.module.css
│   │   ├── Detect.jsx / Detect.module.css
│   │   ├── HowItWorks.jsx / HowItWorks.module.css
│   │   ├── Pricing.jsx / Pricing.module.css
│   │   └── About.jsx / About.module.css
│   ├── App.jsx                ← Page router & global state
│   ├── index.css              ← CSS variables & base styles
│   └── main.jsx               ← React entry point
├── index.html
├── vite.config.js
└── package.json
```

## Getting Started

### 1. Install dependencies
```bash
npm install
```

### 2. Run development server
```bash
npm run dev
```
Then open http://localhost:5173 in your browser.

### 3. Build for production
```bash
npm run build
```

## Connecting Your FastAPI Backend

In `src/pages/Detect.jsx`, find the `finalize()` function and replace the mock with a real API call:

```js
// Replace this mock:
const finalize = () => {
  const isFake = Math.random() > 0.45
  ...
}

// With this real call:
const finalize = async () => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData,
  })
  const data = await res.json()
  // data should be: { verdict: "fake"|"real", confidence: 94.2 }
  setResult(data)
  setPhase('done')
}
```

## Team
- Shaharyar Rizwan — 22I-0999
- Moazzam Hafeez — 22I-1093
- Fiza Jameel — 22I-0964

Supervised by Mr. Arslan Aslam
Department of Computer Science, FAST-NUCES Islamabad
Session 2022–2026
