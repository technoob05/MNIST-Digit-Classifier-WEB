from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import cv2
import base64
import logging
from logging.handlers import RotatingFileHandler
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger = logging.getLogger(__name__)
logger.addHandler(handler)

app = Flask(__name__)
CORS(app)

class EnhancedMNISTNeuralNetwork:
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], output_size=10, 
                 dropout_rate=0.3, momentum=0.9):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        
        self.eta = 2e-3
        self.eta_min = 1e-4
        self.eta_decay = 0.999
        self.alpha = 1e-4
        self.gamma = 0.9
        self.eps = 1e-8
        
        self.initialize_parameters()
    
    def initialize_parameters(self):
        np.random.seed(42)
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.W = []
        self.b = []
        self.v_W = []
        self.v_b = []
        
        for i in range(len(sizes)-1):
            scale = np.sqrt(2.0 / sizes[i])
            w = np.random.randn(sizes[i], sizes[i+1]) * scale
            b = np.zeros(sizes[i+1])
            
            self.W.append(w)
            self.b.append(b)
            self.v_W.append(np.zeros_like(w))
            self.v_b.append(np.zeros_like(b))
    
    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)
    
    def forward(self, X, training=False):
        cache = {}
        cache['a0'] = X
        cache['masks'] = []
        
        for i in range(len(self.hidden_sizes)):
            cache[f'z{i+1}'] = np.matmul(cache[f'a{i}'], self.W[i]) + self.b[i]
            cache[f'a{i+1}'] = self.leaky_relu(cache[f'z{i+1}'])
            
            if training:
                mask = (np.random.rand(*cache[f'a{i+1}'].shape) > self.dropout_rate) / (1 - self.dropout_rate)
                cache[f'a{i+1}'] *= mask
                cache['masks'].append(mask)
        
        i = len(self.hidden_sizes)
        cache[f'z{i+1}'] = np.matmul(cache[f'a{i}'], self.W[i]) + self.b[i]
        
        exp_scores = np.exp(cache[f'z{i+1}'] - np.max(cache[f'z{i+1}'], axis=1, keepdims=True))
        cache['probs'] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return cache
    
    def predict(self, X):
        cache = self.forward(X, training=False)
        return np.argmax(cache['probs'], axis=1)

    def predict_proba(self, X):
        cache = self.forward(X, training=False)
        return cache['probs']

class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model with error handling"""
        try:
            logger.info("Loading model...")
            with open('mnist_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                
                # Create new model with saved configuration
                # Only pass the expected parameters
                config = {
                    'input_size': model_data['config'].get('input_size', 784),
                    'hidden_sizes': model_data['config'].get('hidden_sizes', [512, 256, 128]),
                    'output_size': model_data['config'].get('output_size', 10),
                    'dropout_rate': model_data['config'].get('dropout_rate', 0.3),
                    'momentum': model_data['config'].get('momentum', 0.9)
                }
                
                model = EnhancedMNISTNeuralNetwork(**config)
                
                # Load weights and biases
                model.W = [np.array(w) for w in model_data['weights']]
                model.b = [np.array(b) for b in model_data['biases']]
                
                self.model = model
                logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, image):
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.predict(image), self.model.predict_proba(image)[0]

def preprocess_image(image_data):
    """Enhanced preprocessing to match MNIST format with better noise handling and centering"""
    try:
        # Convert image to numpy array
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        else:
            img = Image.fromarray(image_data)
        
        # Convert to grayscale
        img = img.convert('L')
        img_array = np.array(img)

        # Add padding to avoid cutting off edges
        pad_size = 8
        img_array = np.pad(img_array, pad_size, mode='constant', constant_values=255)
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (should be the digit)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Extract the digit
            digit = binary[y:y+h, x:x+w]
            
            # Calculate aspect ratio
            aspect = h/w
            
            # Determine scaling to fit in 20x20 box while maintaining aspect ratio
            if aspect > 1:
                # Tall and narrow number
                new_h = 20
                new_w = int(20/aspect)
            else:
                # Short and wide number
                new_w = 20
                new_h = int(20*aspect)
            
            # Resize digit while maintaining aspect ratio
            digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create 28x28 blank image (black background)
            final_img = np.zeros((28, 28), dtype=np.uint8)
            
            # Calculate position to paste digit (center it)
            x_start = (28 - new_w) // 2
            y_start = (28 - new_h) // 2
            
            # Paste digit in center
            final_img[y_start:y_start+new_h, x_start:x_start+new_w] = digit
            
        else:
            # If no contours found, return a blank image
            final_img = np.zeros((28, 28), dtype=np.uint8)
        
        # Normalize to [0,1]
        final_img = final_img.astype('float32') / 255.0
        
        # Debug: Save intermediate images if needed
        # cv2.imwrite('debug_preprocessed.png', final_img * 255)
        
        # Reshape for model input (1, 784)
        return final_img.reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        raise
def decode_base64_image(base64_string):
    """Decode base64 image data"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        image_data = base64.b64decode(base64_string)
        return image_data
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_service.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        if 'image' in request.files:
            image_file = request.files['image']
            image_data = image_file.read()
        elif request.is_json and 'image' in request.json:
            image_data = decode_base64_image(request.json['image'])
        else:
            return jsonify({
                'error': 'No image provided',
                'success': False
            }), 400

        # Preprocess image
        processed_image = preprocess_image(image_data)

        # Make prediction
        prediction, probabilities = model_service.predict(processed_image)
        confidence = float(probabilities[prediction[0]])

        logger.info(f"Made prediction: {prediction[0]} with confidence {confidence:.4f}")

        return jsonify({
            'success': True,
            'prediction': int(prediction[0]),
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    try:
        if not request.is_json or 'image' not in request.json:
            return jsonify({
                'error': 'No image data provided',
                'success': False
            }), 400

        # Decode and process image
        image_data = decode_base64_image(request.json['image'])
        processed_image = preprocess_image(image_data)

        # Make prediction
        prediction, probabilities = model_service.predict(processed_image)
        confidence = float(probabilities[prediction[0]])

        logger.info(f"Made camera prediction: {prediction[0]} with confidence {confidence:.4f}")

        return jsonify({
            'success': True,
            'prediction': int(prediction[0]),
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in camera prediction: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {error}")
    return jsonify({
        'error': str(error),
        'success': False
    }), 500

# Create global model service instance
model_service = ModelService()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)