# MNIST Digit Classifier

A modern web application that performs real-time digit recognition using a custom neural network trained on the MNIST dataset. This application supports both webcam capture and image upload for digit classification.

![image](https://github.com/user-attachments/assets/954e5273-4365-4ac8-bbf3-3be9307cdae2)


## Features

- **Real-time Recognition**: Instantly recognize handwritten digits through your webcam
- **Image Upload**: Support for uploading images containing handwritten digits
- **Modern UI**: Clean and responsive interface built with React and Tailwind CSS
- **Advanced Preprocessing**: Enhanced image preprocessing pipeline for better accuracy
- **Confidence Scoring**: View confidence levels for each prediction
- **Error Handling**: Comprehensive error handling and user feedback
- **Logging System**: Detailed logging for monitoring and debugging

## Tech Stack

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- Vite
- Lucide Icons

### Backend
- Flask
- NumPy
- OpenCV
- PIL (Python Imaging Library)
- Custom Neural Network Implementation

## Prerequisites

Before you begin, ensure you have the following installed:
- Node.js (v16 or higher)
- Python 3.8+
- pip
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/technoob05/MNIST-Digit-Classifier-WEB.git
cd mnist-digit-classifier
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root:
```bash
VITE_API_URL=http://localhost:5000
```

2. Configure the backend (optional):
```python
# app.py
DEBUG = False  # Set to True for development
PORT = 5000    # Change if needed
```

## Running the Application

1. Start the Flask backend server:
```bash
python app.py
```

2. In a new terminal, start the frontend development server:
```bash
npm run dev
```

3. Open your browser and navigate to:
```
http://localhost:5173
```

## Development

### Frontend Structure
```
src/
├── components/
│   ├── WebcamCapture.tsx
│   ├── ImageUpload.tsx
│   ├── Prediction.tsx
│   └── ...
├── utils/
│   ├── api.ts
│   └── errorHandling.ts
└── App.tsx
```

### Backend Structure
```
├── app.py
├── models/
│   └── mnist_model.pkl
└── requirements.txt
```

### Adding New Features

1. Frontend Components:
```typescript
// Create new component
import React from 'react';

const NewFeature: React.FC = () => {
  return (
    <div className="p-4">
      // Component content
    </div>
  );
};

export default NewFeature;
```

2. Backend Endpoints:
```python
@app.route('/new-endpoint', methods=['POST'])
def new_endpoint():
    try:
        # Implementation
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

## API Documentation

### Endpoints

#### `POST /predict`
Upload an image for digit recognition.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: form-data with 'image' field

**Response:**
```json
{
  "success": true,
  "prediction": 5,
  "confidence": 0.98,
  "probabilities": [...],
  "timestamp": "2024-01-03T12:00:00Z"
}
```

#### `POST /predict_camera`
Process webcam capture for digit recognition.

**Request:**
- Method: POST
- Content-Type: application/json
- Body: 
```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:** Same as `/predict` endpoint

## Error Handling

The application includes comprehensive error handling for both frontend and backend:

- Network errors
- Image processing failures
- Model prediction errors
- Invalid input handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNIST Dataset creators
- Neural Network architecture inspired by modern deep learning practices
- UI/UX design inspired by modern web applications

## Contact

Technoob05 - [GitHub Profile](https://github.com/technoob05)

Project Link: [https://github.com/technoob05/MNIST-Digit-Classifier-WEB](https://github.com/technoob05/MNIST-Digit-Classifier-WEB)
