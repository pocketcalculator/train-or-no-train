#!/usr/bin/env python3
"""
Azure AI Vision Custom Model Prototype for Train Detection
This script helps evaluate if Azure AI Vision Custom Model is suitable for your train detection needs.
"""

import os
import json
from typing import List, Dict, Tuple
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import cv2
import numpy as np
from pathlib import Path

class TrainDetectionPrototype:
    def __init__(self, vision_endpoint: str, vision_key: str, storage_connection: str):
        """Initialize Azure AI Vision and Storage clients."""
        self.vision_client = ImageAnalysisClient(
            endpoint=vision_endpoint,
            credential=AzureKeyCredential(vision_key)
        )
        self.storage_client = BlobServiceClient.from_connection_string(storage_connection)
        
    def analyze_image_with_builtin_models(self, image_path: str) -> Dict:
        """
        Use Azure AI Vision's built-in models to analyze train images.
        This helps understand what the service can detect without custom training.
        """
        try:
            with open(image_path, "rb") as image_data:
                # Use built-in object detection and general analysis
                result = self.vision_client.analyze(
                    image_data=image_data.read(),
                    visual_features=[
                        VisualFeatures.OBJECTS,
                        VisualFeatures.TAGS,
                        VisualFeatures.CAPTION,
                        VisualFeatures.DENSE_CAPTIONS
                    ]
                )
                
                analysis = {
                    "image_path": image_path,
                    "caption": result.caption.text if result.caption else "No caption",
                    "objects": [],
                    "tags": [],
                    "dense_captions": []
                }
                
                # Extract detected objects
                if result.objects:
                    for obj in result.objects.list:
                        analysis["objects"].append({
                            "name": obj.tags[0].name if obj.tags else "unknown",
                            "confidence": obj.tags[0].confidence if obj.tags else 0,
                            "bounding_box": {
                                "x": obj.bounding_box.x,
                                "y": obj.bounding_box.y,
                                "w": obj.bounding_box.w,
                                "h": obj.bounding_box.h
                            }
                        })
                
                # Extract tags
                if result.tags:
                    for tag in result.tags.list:
                        analysis["tags"].append({
                            "name": tag.name,
                            "confidence": tag.confidence
                        })
                
                # Extract dense captions
                if result.dense_captions:
                    for caption in result.dense_captions.list:
                        analysis["dense_captions"].append({
                            "text": caption.text,
                            "confidence": caption.confidence
                        })
                
                return analysis
                
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return {"error": str(e)}
    
    def batch_analyze_dataset(self, dataset_path: str) -> List[Dict]:
        """Analyze all images in the dataset to understand detection patterns."""
        results = []
        dataset_dir = Path(dataset_path)
        
        # Process images in both categories
        for category in ["train_present", "no_train"]:
            category_path = dataset_dir / category
            if category_path.exists():
                print(f"Analyzing {category} images...")
                for image_file in category_path.glob("*.{jpg,jpeg,png,PNG,JPG,JPEG}"):
                    print(f"Processing: {image_file.name}")
                    result = self.analyze_image_with_builtin_models(str(image_file))
                    result["category"] = category
                    result["filename"] = image_file.name
                    results.append(result)
        
        return results
    
    def evaluate_builtin_performance(self, analysis_results: List[Dict]) -> Dict:
        """
        Evaluate how well Azure AI Vision's built-in models detect trains.
        This helps decide if we need a custom model.
        """
        train_keywords = [
            "train", "locomotive", "railway", "railroad", "rail", "subway", "metro",
            "cargo", "freight", "passenger", "engine", "railcar", "boxcar"
        ]
        
        evaluation = {
            "total_images": len(analysis_results),
            "train_present_images": len([r for r in analysis_results if r.get("category") == "train_present"]),
            "no_train_images": len([r for r in analysis_results if r.get("category") == "no_train"]),
            "correct_detections": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "detection_patterns": {}
        }
        
        for result in analysis_results:
            if "error" in result:
                continue
                
            # Check if any train-related keywords appear in tags or captions
            has_train_keyword = False
            all_text = []
            
            # Collect all text from analysis
            if result.get("caption"):
                all_text.append(result["caption"].lower())
            
            for tag in result.get("tags", []):
                all_text.append(tag["name"].lower())
            
            for caption in result.get("dense_captions", []):
                all_text.append(caption["text"].lower())
            
            # Check for train keywords
            full_text = " ".join(all_text)
            has_train_keyword = any(keyword in full_text for keyword in train_keywords)
            
            # Evaluate detection accuracy
            actual_category = result.get("category")
            if actual_category == "train_present" and has_train_keyword:
                evaluation["correct_detections"] += 1
            elif actual_category == "no_train" and not has_train_keyword:
                evaluation["correct_detections"] += 1
            elif actual_category == "train_present" and not has_train_keyword:
                evaluation["false_negatives"] += 1
            elif actual_category == "no_train" and has_train_keyword:
                evaluation["false_positives"] += 1
        
        # Calculate accuracy
        total_classified = evaluation["correct_detections"] + evaluation["false_positives"] + evaluation["false_negatives"]
        if total_classified > 0:
            evaluation["accuracy"] = evaluation["correct_detections"] / total_classified
        else:
            evaluation["accuracy"] = 0
        
        return evaluation
    
    def generate_recommendations(self, evaluation: Dict) -> List[str]:
        """Generate recommendations based on built-in model performance."""
        recommendations = []
        
        accuracy = evaluation.get("accuracy", 0)
        
        if accuracy >= 0.9:
            recommendations.append("✅ Azure AI Vision built-in models work well for your use case!")
            recommendations.append("✅ Consider using the built-in object detection directly")
            recommendations.append("💡 You might only need keyword filtering on the results")
        elif accuracy >= 0.7:
            recommendations.append("⚠️ Azure AI Vision shows promise but needs improvement")
            recommendations.append("🎯 Consider training a Custom Vision model")
            recommendations.append("💡 Combine built-in detection with custom classification")
        else:
            recommendations.append("❌ Built-in models insufficient for your specific use case")
            recommendations.append("🎯 Custom model training is recommended")
            recommendations.append("💡 Consider TensorFlow/PyTorch custom solution")
        
        # Add specific recommendations based on false rates
        false_positive_rate = evaluation.get("false_positives", 0) / max(evaluation.get("no_train_images", 1), 1)
        false_negative_rate = evaluation.get("false_negatives", 0) / max(evaluation.get("train_present_images", 1), 1)
        
        if false_positive_rate > 0.3:
            recommendations.append("⚠️ High false positive rate - background objects being confused with trains")
        
        if false_negative_rate > 0.3:
            recommendations.append("⚠️ High false negative rate - trains not being detected consistently")
        
        return recommendations

def main():
    """Main function to run the prototype evaluation."""
    print("🚂 Azure AI Vision Train Detection Prototype")
    print("=" * 50)
    
    # Configuration (you'll need to set these)
    VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "")
    VISION_KEY = os.getenv("AZURE_VISION_KEY", "")
    STORAGE_CONNECTION = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    
    if not all([VISION_ENDPOINT, VISION_KEY]):
        print("❌ Please set Azure AI Vision credentials:")
        print("   export AZURE_VISION_ENDPOINT='your-endpoint'")
        print("   export AZURE_VISION_KEY='your-key'")
        return
    
    # Initialize prototype
    prototype = TrainDetectionPrototype(VISION_ENDPOINT, VISION_KEY, STORAGE_CONNECTION)
    
    # Analyze dataset
    dataset_path = input("Enter path to your dataset folder: ").strip()
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    print(f"🔍 Analyzing dataset at: {dataset_path}")
    results = prototype.batch_analyze_dataset(dataset_path)
    
    # Save detailed results
    with open("vision_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Detailed results saved to: vision_analysis_results.json")
    
    # Evaluate performance
    evaluation = prototype.evaluate_builtin_performance(results)
    
    # Generate recommendations
    recommendations = prototype.generate_recommendations(evaluation)
    
    # Print results
    print("\n📊 EVALUATION RESULTS")
    print("=" * 30)
    print(f"Total Images: {evaluation['total_images']}")
    print(f"Train Present: {evaluation['train_present_images']}")
    print(f"No Train: {evaluation['no_train_images']}")
    print(f"Correct Detections: {evaluation['correct_detections']}")
    print(f"False Positives: {evaluation['false_positives']}")
    print(f"False Negatives: {evaluation['false_negatives']}")
    print(f"Accuracy: {evaluation['accuracy']:.2%}")
    
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 20)
    for rec in recommendations:
        print(rec)
    
    # Save evaluation
    with open("evaluation_results.json", "w") as f:
        json.dump({"evaluation": evaluation, "recommendations": recommendations}, f, indent=2)
    
    print(f"\n💾 Evaluation saved to: evaluation_results.json")

if __name__ == "__main__":
    main()