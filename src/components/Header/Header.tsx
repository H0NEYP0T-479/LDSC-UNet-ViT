import React from 'react';
import './Header.css';

export const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <h1>LDSC-UNet-ViT</h1>
          <p>Lung Disease Segmentation & Classification</p>
        </div>
        <nav className="nav">
          <a href="#home">Home</a>
          <a href="#about">About</a>
          <a href="#documentation">Documentation</a>
          <a href="#contact">Contact</a>
        </nav>
      </div>
    </header>
  );
};
