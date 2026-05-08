import './Header.css'

export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-logo">
          <span className="logo-icon">🫁</span>
          <span className="logo-text">LDSC<span className="logo-accent">-UNet-ViT</span></span>
        </div>
        <nav className="header-nav">
          <span className="nav-badge">Lung Disease Detection</span>
        </nav>
      </div>
    </header>
  )
}