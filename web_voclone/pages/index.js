import React, { useState, useRef } from 'react';

export default function CloudDashboard() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [translation, setTranslation] = useState('');
  
  // Updated state to handle generic media (audio or video)
  const [mediaUrl, setMediaUrl] = useState(null);
  const [isOutputVideo, setIsOutputVideo] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setTranscription('');
      setTranslation('');
      setMediaUrl(null);
      setIsOutputVideo(false);
    }
  };

  const handleProcessMedia = async () => {
    if (!file) return;
    setLoading(true);
    setMediaUrl(null);
    
    setTranscription('Whisper Processing Core Active... Parsing speech patterns...');
    setTranslation('Translation Layer Initializing...');

    const formData = new FormData();
    formData.append('audio_file', file);

    try {
      const response = await fetch('https://voclonetranslate.tech/dub', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Pipeline processing exception');

      const rawTranscript = response.headers.get('X-Transcription');
      const rawTranslation = response.headers.get('X-Translation');
      
      if (rawTranscript && rawTranslation) {
        setTranscription(JSON.parse(rawTranscript));
        setTranslation(JSON.parse(rawTranslation));
      } else {
        setTranscription('Speech analyzed and extracted successfully.');
        setTranslation('Linguistics translation rendering complete.');
      }

      // Dynamically detect the incoming blob type
      const mediaBlob = await response.blob();
      const localStreamUrl = URL.createObjectURL(mediaBlob);
      
      // Determine if the pipeline returned a video or if the source was a video
      const isVideoResponse = mediaBlob.type.includes('video') || file.type.startsWith('video/');
      
      setIsOutputVideo(isVideoResponse);
      setMediaUrl(localStreamUrl);

    } catch (error) {
      alert('Operational Failure: The AI processing pipeline hit an issue or memory bounds were exceeded.');
      setTranscription('');
      setTranslation('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#020617', color: '#f8fafc', fontFamily: 'sans-serif', padding: '40px' }}>
      <div style={{ maxWidth: '650px', margin: '0 auto' }}>
        <h1 style={{ fontSize: '32px', fontWeight: '800', marginBottom: '8px', color: '#ffffff' }}>VoCloneTranslate</h1>
        <p style={{ color: '#94a3b8', marginBottom: '40px' }}>AI-Powered Educational Video & Audio Translation Pipeline</p>

        <div 
          onClick={() => fileInputRef.current?.click()}
          style={{ border: '2px dashed #334155', borderRadius: '16px', padding: '40px', textAlign: 'center', cursor: 'pointer', marginBottom: '30px', backgroundColor: '#0f172a' }}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept="audio/*,video/*" 
            style={{ display: 'none' }} 
          />
          <span style={{ fontSize: '40px' }}>📁</span>
          <p style={{ fontSize: '16px', fontWeight: '500', margin: '12px 0 4px 0' }}>Select or drop your educational media here</p>
          <p style={{ fontSize: '12px', color: '#64748b', marginTop: '8px' }}>Supports MP4, MOV, WAV, MP3 up to 50MB</p>
          
          {file && (
            <div style={{ marginTop: '20px', display: 'inline-block', backgroundColor: '#020617', padding: '8px 16px', borderRadius: '8px', border: '1px solid #334155', fontSize: '14px', color: '#a5b4fc' }}>
              📄 {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
            </div>
          )}
        </div>

        <button 
          onClick={handleProcessMedia} 
          disabled={loading || !file} 
          style={{ width: '100%', padding: '16px', backgroundColor: '#4f46e5', color: '#ffffff', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', opacity: (loading || !file) ? 0.4 : 1, marginBottom: '30px', transition: 'all 0.2s ease' }}
        >
          {loading ? 'Processing Pipeline Active (Running Whisper + XTTS + Rendering)...' : 'Upload and Dub Lecture to Urdu'}
        </button>

        {(transcription || translation) && (
          <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '16px', padding: '24px', marginBottom: '24px' }}>
            {transcription && (
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ color: '#6366f1', margin: '0 0 6px 0', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>📝 Source Lecture Transcription (English)</h4>
                <p style={{ margin: 0, fontSize: '15px', color: '#e2e8f0', lineHeight: '1.5', backgroundColor: '#020617', padding: '12px', borderRadius: '8px', border: '1px solid #1e293b' }}>
                  {transcription}
                </p>
              </div>
            )}
            
            {translation && (
              <div>
                <h4 style={{ color: '#a855f7', margin: '0 0 6px 0', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>🔮 Translated Linguistics (Urdu)</h4>
                <p style={{ margin: 0, fontSize: '18px', color: '#f1f5f9', lineHeight: '1.6', direction: 'rtl', textAlign: 'right', backgroundColor: '#020617', padding: '12px', borderRadius: '8px', border: '1px solid #1e293b', fontFamily: 'serif' }}>
                  {translation}
                </p>
              </div>
            )}
          </div>
        )}

        {mediaUrl && (
          <div style={{ backgroundColor: '#0f172a', border: '1px solid #10b981', borderRadius: '16px', padding: '24px' }}>
            <h3 style={{ color: '#10b981', marginTop: 0, fontSize: '14px', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>⚡</span> Output Synchronization Complete
            </h3>
            <p style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '16px' }}>
              {isOutputVideo ? "Your dubbed Urdu lecture video has rendered successfully:" : "Your cloned Urdu speech track has synthesized successfully:"}
            </p>
            
            {/* Dynamic Rendering based on media type */}
            {isOutputVideo ? (
              <video 
                src={mediaUrl} 
                controls 
                autoPlay 
                style={{ width: '100%', borderRadius: '8px', border: '1px solid #1e293b', backgroundColor: '#000' }} 
              />
            ) : (
              <audio 
                src={mediaUrl} 
                controls 
                autoPlay 
                style={{ width: '100%' }} 
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
