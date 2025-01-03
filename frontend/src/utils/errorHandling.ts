export class APIError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'APIError';
  }
}

export function handleApiError(error: unknown): Error {
  if (error instanceof APIError) {
    return error;
  }
  
  if (error instanceof TypeError && error.message === 'Failed to fetch') {
    return new APIError('Unable to connect to the server. Please ensure the backend is running.');
  }
  
  if (error instanceof Error) {
    return new APIError(error.message);
  }
  
  return new APIError('An unexpected error occurred');
}