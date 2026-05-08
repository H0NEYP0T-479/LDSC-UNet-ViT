import './Footer.css'

export default function Footer() {
  return (
    <footer className="footer">
      <p>LDSC-UNet-ViT &copy; {new Date().getFullYear()} — Lung Disease Segmentation & Classification</p>
    </footer>
  )
}