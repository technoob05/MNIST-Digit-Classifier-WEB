import React from 'react';
import { AlertCircle } from 'lucide-react';

interface ErrorMessageProps {
  message: string;
  onDismiss?: () => void;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, onDismiss }) => {
  return (
    <div className="bg-red-50 border-l-4 border-red-500 p-4 my-4">
      <div className="flex items-center">
        <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
        <p className="text-red-700">{message}</p>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="ml-auto text-red-500 hover:text-red-700"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;