import styles from './Pricing.module.css'

const PLANS = [
  {
    name: 'Free', price: '0', period: '/mo',
    desc: 'For students and personal use.',
    features: ['5 video analyses / month', 'Max 100 MB per video', 'Standard processing speed', 'Confidence score & verdict', 'Email support'],
    btn: 'Get Started', btnStyle: 'navy', featured: false,
  },
  {
    name: 'Pro', price: '19', period: '/mo',
    desc: 'For researchers and content creators.',
    features: ['100 analyses / month', 'Max 500 MB per video', 'Priority GPU processing', 'Frame-level NPR report', 'API access (1,000 calls/mo)', 'Priority support'],
    btn: 'Start Free Trial', btnStyle: 'white', featured: true, badge: 'Most Popular',
  },
  {
    name: 'Enterprise', price: '99', period: '/mo',
    desc: 'For organizations and media companies.',
    features: ['Unlimited analyses', 'Unlimited file size', 'Dedicated GPU inference', 'Custom model fine-tuning', 'Full API access', 'SLA + dedicated support'],
    btn: 'Contact Sales', btnStyle: 'outline', featured: false,
  },
]

export default function Pricing() {
  return (
    <div className="page" style={{ padding: '60px 7% 60px' }}>
      <div className="page-header">
        <h2>Simple Pricing</h2>
        <p>Start free, scale as you grow. No hidden fees.</p>
      </div>
      <div className={styles.grid}>
        {PLANS.map(p => (
          <div key={p.name} className={`${styles.card} ${p.featured ? styles.featured : ''}`}>
            <div>
              <div className={styles.name}>{p.name}</div>
              {p.badge && <span className={styles.badge}>{p.badge}</span>}
            </div>
            <div className={styles.price}>
              <sup>$</sup>{p.price}<small>{p.period}</small>
            </div>
            <div className={styles.desc}>{p.desc}</div>
            <ul className={styles.features}>
              {p.features.map(f => <li key={f}>{f}</li>)}
            </ul>
            <button className={`${styles.btn} ${styles[p.btnStyle]}`}>{p.btn}</button>
          </div>
        ))}
      </div>
    </div>
  )
}
