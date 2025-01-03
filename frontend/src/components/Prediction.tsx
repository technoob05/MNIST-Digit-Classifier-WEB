import React from 'react';
import { Loader2 } from 'lucide-react';

interface PredictionProps {
  prediction: string | null;
  loading: boolean;
}

const Prediction: React.FC<PredictionProps> = ({ prediction, loading }) => {
  if (!loading && !prediction) return null;

  return (
    <div className="mt-6 text-center">
      {loading ? (
        <div className="flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
          <span className="ml-2 text-gray-600">Processing image...</span>
        </div>
      ) : (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            Prediction Result
          </h3>
          <p className="text-3xl font-bold text-indigo-600">{prediction}</p>
        </div>
      )}
    </div>
  );
};

export default Prediction;