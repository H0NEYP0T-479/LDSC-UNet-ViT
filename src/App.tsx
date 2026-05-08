import { useState } from 'react'
import Header from './components/Header/Header'
import Footer from './components/Footer/Footer'
import HomePage from './pages/HomePage'
import ErrorBoundary from './components/ErrorBoundary/ErrorBoundary'
import './App.css'

function App() {
  return (
    <ErrorBoundary>
      <div className="app">
        <Header />
        <main className="main-content">
          <HomePage />
        </main>
        <Footer />
      </div>
    </ErrorBoundary>
  )
}

export default App