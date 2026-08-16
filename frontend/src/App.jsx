import React, { useState, useRef } from 'react';
import './App.css';

const API_URL = 'http://127.0.0.1:5000/api/predict';

function App() {
  const [file, setFile] = useState(null);
  const [crop, setCrop] = useState('cassava');
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const onFileChange = (f) => {
    setError(null);
    setResult(null);
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileChange(e.dataTransfer.files[0]);
    }
  };

  const onAnalyze = async () => {
    if (!file) {
      setError('Please upload an image first.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append('image', file);
      form.append('crop', crop);

      const res = await fetch(API_URL, {
        method: 'POST',
        body: form,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `Server responded with ${res.status}`);
      }

      if (!data.success) {
        throw new Error(data.error || 'Prediction failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  const onReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = null;
  };

  const badgeColor = (className) => {
    if (!className) return 'gray';
    if (className.toLowerCase().includes('healthy')) return 'green';
    return 'red';
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>AgriScan: AI Leaf Disease Detector</h1>
        <p className="tagline">Instant crop health diagnosis for cassava and maize using AI.</p>
      </header>

      <main className="container">
        <div className="crop-picker">
          <label htmlFor="crop">Select crop</label>
          <select id="crop" value={crop} onChange={(e) => setCrop(e.target.value)}>
            <option value="cassava">Cassava</option>
            <option value="maize">Maize</option>
          </select>
        </div>

        <section
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => inputRef.current && inputRef.current.click()}
        >
          {preview ? (
            <img src={preview} alt="preview" className="preview-image" />
          ) : (
            <div className="drop-content">
              <strong>Drag & drop an image here</strong>
              <span>or click to upload (jpg, jpeg, png)</span>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/png, image/jpeg, image/jpg"
            className="file-input"
            onChange={(e) => e.target.files && onFileChange(e.target.files[0])}
          />
        </section>

        <div className="controls">
          <button className="btn primary" onClick={onAnalyze} disabled={loading || !file}>
            {loading ? 'Analyzing...' : 'Analyze Leaf Health'}
          </button>
          <button className="btn" onClick={onReset}>
            Analyze Another Image
          </button>
        </div>

        {error && <div className="alert error">{error}</div>}

        {result && (
          <section className="result-card">
            <div className="result-header">
              <span className={`badge ${badgeColor(result.class_name)}`}>{result.display_name}</span>
              <div className="confidence">{result.confidence}% Confidence</div>
            </div>

            <div className="confidence-bar">
              <div className="fill" style={{ width: `${result.confidence}%` }} />
            </div>

            <div className="recommendation">
              <h3>Recommendation</h3>
              <p>{result.recommendation}</p>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">Built for farmers • Keep fields healthy</footer>
    </div>
  );
}

export default App;
