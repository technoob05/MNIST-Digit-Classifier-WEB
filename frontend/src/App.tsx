import React, { useState } from 'react';
import { RefreshCw } from 'lucide-react';
import WebcamCapture from './components/WebcamCapture';
import ImageUpload from './components/ImageUpload';
import Prediction from './components/Prediction';
import ErrorMessage from './components/ErrorMessage';
import ModeSelector from './components/ModeSelector';
import { predictImage } from './utils/api';
import type { ImageSource } from './types';

function App() {
  const [mode, setMode] = useState<ImageSource>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSubmit = async (image: string | File) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictImage(image);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setMode(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
            MNIST Digit Classifier
          </h1>
          
          {error && (
            <ErrorMessage 
              message={error} 
              onDismiss={() => setError(null)} 
            />
          )}
          
          {!mode && <ModeSelector onSelectMode={setMode} />}

          {mode && (
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <button
                onClick={resetState}
                className="mb-4 flex items-center text-indigo-600 hover:text-indigo-800"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Start Over
              </button>

              {mode === 'upload' && (
                <ImageUpload onImageSelect={handleImageSubmit} />
              )}

              {mode === 'camera' && (
                <WebcamCapture onCapture={handleImageSubmit} />
              )}

              <Prediction prediction={prediction} loading={loading} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;