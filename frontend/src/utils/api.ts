import { APIResponse } from '../types';
import { handleApiError } from './errorHandling';

const API_BASE_URL = 'http://localhost:5000';

async function createFormData(image: string | File): Promise<FormData> {
  const formData = new FormData();
  
  if (typeof image === 'string') {
    const response = await fetch(image);
    const blob = await response.blob();
    formData.append('image', blob, 'capture.jpg');
  } else {
    formData.append('image', image);
  }
  
  return formData;
}

export async function predictImage(image: string | File): Promise<string> {
  try {
    const formData = await createFormData(image);
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    
    const data: APIResponse = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || `HTTP error! status: ${response.status}`);
    }
    
    if (!data.success) {
      throw new Error(data.error || 'Prediction failed');
    }
    
    return data.prediction;
  } catch (error) {
    throw handleApiError(error);
  }
}