export interface APIResponse {
  success: boolean;
  prediction?: string;
  error?: string;
}

export type ImageSource = 'upload' | 'camera' | null;