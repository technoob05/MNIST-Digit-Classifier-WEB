import React from 'react';
import { Camera, Upload } from 'lucide-react';

interface ModeSelectorProps {
  onSelectMode: (mode: 'upload' | 'camera') => void;
}

const ModeSelector: React.FC<ModeSelectorProps> = ({ onSelectMode }) => {
  return (
    <div className="grid grid-cols-2 gap-4 mb-8">
      <button
        onClick={() => onSelectMode('upload')}
        className="flex flex-col items-center justify-center p-6 bg-white rounded-lg shadow-lg hover:shadow-xl transition-shadow"
      >
        <Upload className="w-12 h-12 text-indigo-600 mb-2" />
        <span className="text-lg font-medium text-gray-700">Upload Image</span>
      </button>
      
      <button
        onClick={() => onSelectMode('camera')}
        className="flex flex-col items-center justify-center p-6 bg-white rounded-lg shadow-lg hover:shadow-xl transition-shadow"
      >
        <Camera className="w-12 h-12 text-indigo-600 mb-2" />
        <span className="text-lg font-medium text-gray-700">Use Camera</span>
      </button>
    </div>
  );
};

export default ModeSelector;