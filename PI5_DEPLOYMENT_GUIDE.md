# Raspberry Pi 5 Deployment Guide

This guide provides step-by-step instructions for deploying your trained train detection model to a Raspberry Pi 5.

## Prerequisites

- Raspberry Pi 5 with Raspberry Pi OS installed
- Python 3.9+ installed on the Pi
- SSH access or direct terminal access to the Pi
- Trained model file (`train_detector_model.h5`) from your local training

## Step 1: Convert Model to TensorFlow Lite

On your training machine, convert the TensorFlow model to TensorFlow Lite format for optimal Pi performance:

```bash
cd local-training/src
python3 -c "
import tensorflow as tf
model = tf.keras.models.load_model('../models/train_detector_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('../models/train_detector_model.tflite', 'wb') as f:
    f.write(tflite_model)
print('Model converted to TensorFlow Lite')
"
```

## Step 2: Create Pi Inference Script

Create the inference script that will run on the Pi:

```python
# Save this as pi_inference_script.py
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import cv2
import time

class TrainDetector:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0).astype(np.float32)
    
    def predict(self, image_path):
        """Run inference on a single image"""
        start_time = time.time()
        
        input_data = self.preprocess_image(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        inference_time = time.time() - start_time
        confidence = output_data[0][0]
        result = 'train' if confidence > 0.5 else 'no_train'
        
        return {
            'prediction': result,
            'confidence': float(confidence),
            'inference_time': inference_time
        }

def main():
    # Initialize detector
    detector = TrainDetector('train_detector_model.tflite')
    
    # Example usage
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = detector.predict(image_path)
        print(f"Image: {image_path}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Inference time: {result['inference_time']:.3f}s")
    else:
        print("Usage: python pi_inference_script.py <image_path>")

if __name__ == "__main__":
    main()
```

## Step 3: Transfer Files to Pi

Copy the model and inference script to your Raspberry Pi:

```bash
# Replace [PI_IP] with your Pi's IP address
scp local-training/models/train_detector_model.tflite pi@[PI_IP]:~/
scp pi_inference_script.py pi@[PI_IP]:~/

# Or if using direct file transfer (USB, etc.)
# Copy the files to a USB drive and transfer manually
```

## Step 4: Set Up Pi Environment

On your Raspberry Pi, install the required packages:

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install Python dependencies
sudo apt install python3-pip python3-opencv -y

# Install TensorFlow Lite runtime
pip3 install tflite-runtime numpy pillow

# Verify installation
python3 -c "import tflite_runtime.interpreter as tflite; print('TFLite runtime installed successfully')"
```

## Step 5: Test the Deployment

Test your model on the Pi:

```bash
# Test with a sample image
python3 pi_inference_script.py test_image.jpg

# Expected output:
# Model loaded: train_detector_model.tflite
# Input shape: [1, 224, 224, 3]
# Image: test_image.jpg
# Prediction: train
# Confidence: 0.873
# Inference time: 0.245s
```

## Step 6: Performance Optimization

### Memory Optimization
```python
# Add these optimizations to your inference script
import gc

class OptimizedTrainDetector(TrainDetector):
    def predict_batch(self, image_paths, batch_size=4):
        """Process multiple images in batches"""
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            for image_path in batch:
                results.append(self.predict(image_path))
            gc.collect()  # Free memory between batches
        return results
```

### CPU Optimization
```bash
# Enable ARM NEON optimizations
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4

# Run with optimized settings
python3 pi_inference_script.py test_image.jpg
```

## Step 7: Camera Integration (Optional)

For real-time camera detection:

```python
# Add to pi_inference_script.py
import cv2

def camera_detection(detector):
    """Real-time camera detection"""
    cap = cv2.VideoCapture(0)  # Use default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame temporarily
        cv2.imwrite('temp_frame.jpg', frame)
        
        # Run detection
        result = detector.predict('temp_frame.jpg')
        
        # Display result on frame
        label = f"{result['prediction']} ({result['confidence']:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Train Detection', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Add to main() function
if len(sys.argv) > 1 and sys.argv[1] == '--camera':
    detector = TrainDetector('train_detector_model.tflite')
    camera_detection(detector)
```

## Troubleshooting

### Common Issues

1. **TensorFlow Lite not found**
   ```bash
   pip3 install --upgrade pip
   pip3 install tflite-runtime
   ```

2. **OpenCV import errors**
   ```bash
   sudo apt install python3-opencv libopencv-dev
   ```

3. **Memory errors**
   - Reduce image batch size
   - Use lower resolution images
   - Add `gc.collect()` calls

4. **Slow inference**
   - Verify TensorFlow Lite conversion
   - Check CPU temperature (`vcgencmd measure_temp`)
   - Ensure adequate power supply

### Performance Benchmarks

Expected performance on Raspberry Pi 5:
- **Inference time**: 200-500ms per image
- **Memory usage**: ~100MB
- **Model size**: ~10-20MB (TFLite)
- **Accuracy**: Should match training accuracy (85.7%+)

## Next Steps

1. **Optimize further**: Experiment with quantization for faster inference
2. **Add logging**: Implement detection logging and statistics
3. **Create service**: Set up as a systemd service for automatic startup
4. **Remote monitoring**: Add web interface or API for remote access

---

Your train detection model is now ready to run on Raspberry Pi 5! The TensorFlow Lite format provides optimal performance for edge deployment while maintaining the accuracy of your trained model.