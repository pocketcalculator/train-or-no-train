#!/usr/bin/env python3
"""
Train Detection Service - Integrates with Azure Function
This service processes images to detect trains and can be called from the Azure Function.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

# Azure imports
from azure.storage.blob import BlobServiceClient
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Model imports (will be installed separately)
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("Warning: OpenCV/TensorFlow not available. Install for full functionality.")

class TrainDetectionService:
    """Service for detecting trains in images using custom or Azure AI models."""
    
    def __init__(self, 
                 storage_connection_string: str,
                 model_path: Optional[str] = None,
                 azure_ai_endpoint: Optional[str] = None,
                 azure_ai_key: Optional[str] = None,
                 use_custom_model: bool = True):
        """
        Initialize the train detection service.
        
        Args:
            storage_connection_string: Azure Storage connection string
            model_path: Path to custom TensorFlow model file
            azure_ai_endpoint: Azure AI endpoint for fallback
            azure_ai_key: Azure AI key for fallback
            use_custom_model: Whether to use custom model first
        """
        self.storage_client = BlobServiceClient.from_connection_string(storage_connection_string)
        self.model_path = model_path
        self.use_custom_model = use_custom_model and CV_AVAILABLE
        self.custom_model = None
        self.azure_ai_client = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure AI client if credentials provided
        if azure_ai_endpoint and azure_ai_key:
            self.azure_ai_client = ChatCompletionsClient(
                endpoint=azure_ai_endpoint,
                credential=AzureKeyCredential(azure_ai_key)
            )
        
        # Load custom model if available
        if self.use_custom_model and model_path and os.path.exists(model_path):
            self._load_custom_model()
    
    def _load_custom_model(self):
        """Load the custom TensorFlow model."""
        try:
            self.custom_model = keras.models.load_model(self.model_path)
            self.logger.info(f"Custom model loaded from: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load custom model: {str(e)}")
            self.custom_model = None
    
    def _preprocess_image_for_custom_model(self, image_data: bytes, 
                                         img_height: int = 224, 
                                         img_width: int = 224) -> Optional[np.ndarray]:
        """Preprocess image data for the custom model."""
        if not CV_AVAILABLE:
            return None
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (img_width, img_height))
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)
            
            return img_batch
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def _predict_with_custom_model(self, image_data: bytes, 
                                 threshold: float = 0.5) -> Optional[Dict]:
        """Use custom model to predict train presence."""
        if not self.custom_model or not CV_AVAILABLE:
            return None
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image_for_custom_model(image_data)
            if processed_image is None:
                return None
            
            # Make prediction
            prediction_proba = self.custom_model.predict(processed_image, verbose=0)[0][0]
            prediction = int(prediction_proba > threshold)
            
            result = {
                "method": "custom_model",
                "train_detected": prediction == 1,
                "confidence": float(max(prediction_proba, 1 - prediction_proba)),
                "train_probability": float(prediction_proba),
                "threshold": threshold
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Custom model prediction failed: {str(e)}")
            return None
    
    def _predict_with_azure_ai(self, image_analysis: str) -> Optional[Dict]:
        """Use Azure AI to analyze image description for train detection."""
        if not self.azure_ai_client:
            return None
        
        try:
            # Create prompt for train detection
            system_prompt = """You are an expert at analyzing images for train detection. 
            Based on the image description provided, determine if there is a train, locomotive, 
            or rail cars visible in the image. Consider terms like: train, locomotive, railway, 
            railroad, rail car, subway, metro, freight, passenger car, engine.
            
            Respond with a JSON object containing:
            - train_detected: boolean (true if train is present)
            - confidence: float (0.0 to 1.0 confidence score)
            - reasoning: string (brief explanation)"""
            
            user_prompt = f"Image description: {image_analysis}\n\nIs there a train in this image?"
            
            response = self.azure_ai_client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt)
                ],
                model="gpt-4o-mini",  # Use cost-effective model
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    
                    result = {
                        "method": "azure_ai_analysis",
                        "train_detected": result_data.get("train_detected", False),
                        "confidence": result_data.get("confidence", 0.5),
                        "reasoning": result_data.get("reasoning", ""),
                        "raw_response": response_text
                    }
                    
                    return result
                else:
                    # Fallback: keyword-based analysis
                    return self._keyword_based_analysis(response_text)
                    
            except json.JSONDecodeError:
                # Fallback: keyword-based analysis
                return self._keyword_based_analysis(response_text)
                
        except Exception as e:
            self.logger.error(f"Azure AI prediction failed: {str(e)}")
            return None
    
    def _keyword_based_analysis(self, text: str) -> Dict:
        """Fallback keyword-based train detection."""
        train_keywords = [
            "train", "locomotive", "railway", "railroad", "rail", "subway", "metro",
            "cargo", "freight", "passenger", "engine", "railcar", "boxcar", "coach"
        ]
        
        text_lower = text.lower()
        detected_keywords = [keyword for keyword in train_keywords if keyword in text_lower]
        
        train_detected = len(detected_keywords) > 0
        confidence = min(len(detected_keywords) * 0.3, 1.0) if train_detected else 0.8
        
        return {
            "method": "keyword_analysis",
            "train_detected": train_detected,
            "confidence": confidence,
            "detected_keywords": detected_keywords,
            "analysis_text": text
        }
    
    async def analyze_image_blob(self, container_name: str, blob_name: str) -> Dict:
        """
        Analyze an image blob for train detection.
        
        Args:
            container_name: Name of the blob container
            blob_name: Name of the blob file
            
        Returns:
            Dictionary containing detection results
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Get blob client
            blob_client = self.storage_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            # Check if blob exists
            if not await blob_client.exists():
                return {
                    "error": f"Blob {blob_name} not found in container {container_name}",
                    "timestamp": timestamp
                }
            
            # Download blob data
            blob_data = await blob_client.download_blob()
            image_data = await blob_data.readall()
            
            results = {
                "blob_name": blob_name,
                "container_name": container_name,
                "timestamp": timestamp,
                "file_size": len(image_data),
                "detection_results": []
            }
            
            # Try custom model first if available
            if self.use_custom_model and self.custom_model:
                self.logger.info(f"Analyzing {blob_name} with custom model...")
                custom_result = self._predict_with_custom_model(image_data)
                if custom_result:
                    results["detection_results"].append(custom_result)
                    results["primary_method"] = "custom_model"
                    results["train_detected"] = custom_result["train_detected"]
                    results["confidence"] = custom_result["confidence"]
            
            # If custom model failed or not available, try Azure AI
            if not results.get("detection_results") and self.azure_ai_client:
                self.logger.info(f"Analyzing {blob_name} with Azure AI...")
                # For this example, we'll use a simple description
                # In practice, you'd use Azure Computer Vision to get image description
                azure_result = self._predict_with_azure_ai(f"Railway track image: {blob_name}")
                if azure_result:
                    results["detection_results"].append(azure_result)
                    results["primary_method"] = "azure_ai"
                    results["train_detected"] = azure_result["train_detected"]
                    results["confidence"] = azure_result["confidence"]
            
            # Fallback: basic filename analysis
            if not results.get("detection_results"):
                filename_result = self._keyword_based_analysis(blob_name)
                results["detection_results"].append(filename_result)
                results["primary_method"] = "filename_analysis"
                results["train_detected"] = filename_result["train_detected"]
                results["confidence"] = filename_result["confidence"]
            
            self.logger.info(f"Analysis complete for {blob_name}: "
                           f"Train detected: {results.get('train_detected', False)}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error analyzing {blob_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "error": error_msg,
                "blob_name": blob_name,
                "container_name": container_name,
                "timestamp": timestamp
            }
    
    def analyze_image_sync(self, container_name: str, blob_name: str) -> Dict:
        """Synchronous wrapper for image analysis."""
        return asyncio.run(self.analyze_image_blob(container_name, blob_name))
    
    async def batch_analyze_container(self, container_name: str, 
                                    file_extensions: List[str] = None) -> List[Dict]:
        """
        Analyze all images in a container for train detection.
        
        Args:
            container_name: Name of the container to analyze
            file_extensions: List of file extensions to process (default: common image formats)
            
        Returns:
            List of analysis results for each image
        """
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        results = []
        
        try:
            # Get container client
            container_client = self.storage_client.get_container_client(container_name)
            
            # List all blobs in container
            blob_list = container_client.list_blobs()
            
            for blob in blob_list:
                # Check if file has valid image extension
                if any(blob.name.lower().endswith(ext.lower()) for ext in file_extensions):
                    self.logger.info(f"Processing blob: {blob.name}")
                    
                    # Analyze the image
                    result = await self.analyze_image_blob(container_name, blob.name)
                    results.append(result)
                else:
                    self.logger.info(f"Skipping non-image file: {blob.name}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error batch analyzing container {container_name}: {str(e)}"
            self.logger.error(error_msg)
            return [{"error": error_msg, "container_name": container_name}]


def create_train_detection_service() -> TrainDetectionService:
    """Factory function to create TrainDetectionService with environment configuration."""
    
    # Get configuration from environment variables
    storage_connection = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    model_path = os.getenv("TRAIN_MODEL_PATH", "train_detection_model.h5")
    azure_ai_endpoint = os.getenv("AZURE_AI_ENDPOINT")
    azure_ai_key = os.getenv("AZURE_AI_KEY")
    use_custom_model = os.getenv("USE_CUSTOM_MODEL", "true").lower() == "true"
    
    if not storage_connection:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
    
    return TrainDetectionService(
        storage_connection_string=storage_connection,
        model_path=model_path if os.path.exists(model_path) else None,
        azure_ai_endpoint=azure_ai_endpoint,
        azure_ai_key=azure_ai_key,
        use_custom_model=use_custom_model
    )


# Example usage
def main():
    """Example usage of the TrainDetectionService."""
    print("🚂 Train Detection Service Test")
    print("=" * 35)
    
    try:
        # Create service
        service = create_train_detection_service()
        
        # Test single image
        container_name = input("Enter container name (default: incoming): ").strip() or "incoming"
        blob_name = input("Enter blob name to test: ").strip()
        
        if blob_name:
            print(f"🔍 Analyzing {blob_name} in {container_name}...")
            result = service.analyze_image_sync(container_name, blob_name)
            
            print("\n📊 DETECTION RESULT")
            print("=" * 20)
            print(json.dumps(result, indent=2))
        
        # Test batch analysis
        batch_test = input("\nRun batch analysis? (y/n): ").strip().lower()
        if batch_test == 'y':
            print(f"🔍 Batch analyzing container: {container_name}")
            results = asyncio.run(service.batch_analyze_container(container_name))
            
            print(f"\n📊 BATCH RESULTS ({len(results)} images)")
            print("=" * 30)
            for result in results:
                if "error" not in result:
                    train_status = "🚂 TRAIN" if result.get("train_detected") else "❌ NO TRAIN"
                    confidence = result.get("confidence", 0)
                    print(f"{result['blob_name']}: {train_status} (confidence: {confidence:.2f})")
                else:
                    print(f"{result.get('blob_name', 'unknown')}: ERROR - {result['error']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure to set AZURE_STORAGE_CONNECTION_STRING environment variable")


if __name__ == "__main__":
    main()