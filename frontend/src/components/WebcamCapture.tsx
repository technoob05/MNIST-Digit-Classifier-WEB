import React, { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';
import { Camera } from 'lucide-react';

interface WebcamCaptureProps {
  onCapture: (imageSrc: string) => void;
}

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onCapture }) => {
  const webcamRef = useRef<Webcam>(null);
  const [isReady, setIsReady] = useState(false);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      onCapture(imageSrc);
    }
  }, [onCapture]);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-full max-w-md mb-4">
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          onUserMedia={() => setIsReady(true)}
          className="rounded-lg w-full"
        />
      </div>
      
      <button
        onClick={capture}
        disabled={!isReady}
        className={`flex items-center px-6 py-3 rounded-lg ${
          isReady
            ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
        }`}
      >
        <Camera className="w-5 h-5 mr-2" />
        Capture Image
      </button>
    </div>
  );
};

export default WebcamCapture;