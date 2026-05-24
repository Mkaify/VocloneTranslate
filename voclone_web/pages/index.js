import React, { useState, useRef } from 'react';

export default function CloudDashboard() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleProcessMedia = async () => {
    if (!file) return;
    setLoading(true);
    setAudioUrl(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Points securely to your local backend server on AWS
      const response = await fetch('http://18.212.22.217:8000/api/production-dub', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Pipeline error');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch (error) {
      alert('Operational Failure: The AI processing pipeline failed or memory bounds were exceeded.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#020617', color: '#f8fafc', fontFamily: 'sans-serif', padding: '40px' }}>
      <header style={{ borderBottom: '1px solid #1e293b', paddingBottom: '20px', marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold', tracking: 'wide', color: '#818cf8' }}>VOCLONETRANSLATE</h1>
        <span style={{ fontSize: '12px', color: '#34d399', backgroundColor: '#064e3b', padding: '6px 12px', borderRadius: '6px', fontWeight: 'bold' }}>● PRODUCTION INSTANCE ONLINE</span>
      </header>

      <main style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div style={{ border: '2px dashed #1e293b', borderRadius: '16px', padding: '60px', textAlign: 'center', backgroundColor: '#0f172a' }}>
          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="audio/*,video/*" style={{ display: 'none' }} />
          <button onClick={() => fileInputRef.current.click()} style={{ backgroundColor: 'transparent', border: 'none', color: '#6366f1', fontSize: '16px', fontWeight: '500', cursor: 'pointer' }}>
            Select Audio or Video File
          </button>
          <p style={{ color: '#64748b', fontSize: '12px', marginTop: '8px' }}>Supports MP4, MOV, WAV, MP3, M4A up to 50MB</p>
          
          {file && (
            <div style={{ marginTop: '20px', display: 'inline-block', backgroundColor: '#020617', padding: '8px 16px', borderRadius: '8px', border: '1px solid #334155', fontSize: '14px', color: '#a5b4fc' }}>
              📄 {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
            </div>
          )}
        </div>

        <button onClick={handleProcessMedia} disabled={loading || !file} style={{ width: '100%', padding: '16px', backgroundColor: '#4f46e5', color: '#ffffff', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', opacity: (loading || !file) ? 0.4 : 1 }}>
          {loading ? 'Processing Pipeline Active (Running Whisper + XTTS)...' : 'Upload and Dub to Urdu'}
        </button>

        {audioUrl && (
          <div style={{ backgroundColor: '#0f172a', border: '1px solid #10b981', borderRadius: '16px', padding: '24px', marginTop: '20px' }}>
            <h3 style={{ color: '#10b981', marginTop: 0, fontSize: '14px', uppercase: 'true' }}>⚡ Output Synchronization Complete</h3>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px' }}>
              <span style={{ fontSize: '14px', color: '#94a3b8' }}>Protocol: dubbed_media.wav</span>
              <audio src={audioUrl} controls />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
