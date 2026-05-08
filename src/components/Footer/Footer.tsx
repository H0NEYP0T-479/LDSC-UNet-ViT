import React from 'react';
import './Footer.css';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-section">
          <h3>LDSC-UNet-ViT</h3>
          <p>Advanced lung disease detection and classification system</p>
        </div>
        <div className="footer-section">
          <h3>Links</h3>
          <ul>
            <li><a href="#documentation">Documentation</a></li>
            <li><a href="#github">GitHub</a></li>
            <li><a href="#license">License</a></li>
          </ul>
        </div>
        <div className="footer-section">
          <h3>Contact</h3>
          <p>Email: info@ldsc-unet-vit.com</p>
          <p>Support: support@ldsc-unet-vit.com</p>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; {currentYear} LDSC-UNet-ViT. All rights reserved.</p>
      </div>
    </footer>
  );
};
