#!/usr/bin/env python3
"""
Dataset Preparation Helper
Helps organize and validate your train detection dataset.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import cv2
from collections import defaultdict

class DatasetHelper:
    def __init__(self, dataset_path: str = "../dataset"):
        """Initialize dataset helper."""
        self.dataset_path = Path(dataset_path)
        self.train_present_dir = self.dataset_path / "train_present"
        self.no_train_dir = self.dataset_path / "no_train"
        
    def create_dataset_structure(self):
        """Create the required dataset directory structure."""
        print("📁 Creating dataset structure...")
        
        directories = [
            self.train_present_dir,
            self.no_train_dir,
            self.dataset_path / "validation" / "test_train",
            self.dataset_path / "validation" / "test_no_train"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        
        # Create .gitkeep files to preserve empty directories
        for directory in directories:
            gitkeep_file = directory / ".gitkeep"
            if not gitkeep_file.exists():
                gitkeep_file.touch()
        
        print("✅ Dataset structure created successfully!")
        self.print_dataset_structure()
    
    def print_dataset_structure(self):
        """Print the expected dataset structure."""
        print("\n📋 Expected dataset structure:")
        print("dataset/")
        print("├── train_present/     # Images WITH trains")
        print("├── no_train/          # Images WITHOUT trains")
        print("└── validation/")
        print("    ├── test_train/    # Test images with trains")
        print("    └── test_no_train/ # Test images without trains")
        print()
    
    def validate_dataset(self) -> Dict:
        """Validate the dataset and return statistics."""
        print("🔍 Validating dataset...")
        
        stats = {
            "train_present": 0,
            "no_train": 0,
            "validation_train": 0,
            "validation_no_train": 0,
            "total": 0,
            "issues": []
        }
        
        # Check each directory
        directories = {
            "train_present": self.train_present_dir,
            "no_train": self.no_train_dir,
            "validation_train": self.dataset_path / "validation" / "test_train",
            "validation_no_train": self.dataset_path / "validation" / "test_no_train"
        }
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for category, directory in directories.items():
            if not directory.exists():
                stats["issues"].append(f"Directory missing: {directory}")
                continue
            
            # Count valid image files
            image_files = []
            for file in directory.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    # Try to load the image to validate it
                    try:
                        img = cv2.imread(str(file))
                        if img is not None:
                            image_files.append(file)
                        else:
                            stats["issues"].append(f"Corrupted image: {file}")
                    except Exception as e:
                        stats["issues"].append(f"Error reading {file}: {str(e)}")
            
            stats[category] = len(image_files)
            stats["total"] += len(image_files)
        
        # Print results
        print(f"📊 Dataset Statistics:")
        print(f"   - Train present: {stats['train_present']} images")
        print(f"   - No train: {stats['no_train']} images")
        print(f"   - Validation (train): {stats['validation_train']} images")
        print(f"   - Validation (no train): {stats['validation_no_train']} images")
        print(f"   - Total: {stats['total']} images")
        
        # Check for issues
        if stats["issues"]:
            print(f"\n⚠️ Issues found:")
            for issue in stats["issues"]:
                print(f"   - {issue}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        min_per_category = 100
        recommended_per_category = 200
        
        if stats['train_present'] < min_per_category:
            print(f"   - Add more 'train present' images (current: {stats['train_present']}, min: {min_per_category})")
        
        if stats['no_train'] < min_per_category:
            print(f"   - Add more 'no train' images (current: {stats['no_train']}, min: {min_per_category})")
        
        # Check balance
        if stats['train_present'] > 0 and stats['no_train'] > 0:
            ratio = max(stats['train_present'], stats['no_train']) / min(stats['train_present'], stats['no_train'])
            if ratio > 2:
                print(f"   - Dataset is imbalanced (ratio: {ratio:.1f}:1). Consider balancing the classes.")
        
        if stats['train_present'] >= recommended_per_category and stats['no_train'] >= recommended_per_category:
            print(f"   ✅ Dataset size looks good for training!")
        else:
            print(f"   - For best results, aim for {recommended_per_category}+ images per category")
        
        return stats
    
    def organize_images_by_keyword(self, source_directory: str):
        """Help organize images based on filename keywords."""
        print(f"🔧 Organizing images from: {source_directory}")
        
        source_path = Path(source_directory)
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_directory}")
            return
        
        # Keywords that suggest train presence
        train_keywords = ['train', 'locomotive', 'rail', 'subway', 'metro', 'cargo', 'freight']
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in source_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"📁 Found {len(image_files)} image files")
        
        # Categorize by filename
        train_files = []
        no_train_files = []
        uncertain_files = []
        
        for file in image_files:
            filename_lower = file.name.lower()
            
            if any(keyword in filename_lower for keyword in train_keywords):
                train_files.append(file)
            elif any(word in filename_lower for word in ['empty', 'clear', 'vacant']):
                no_train_files.append(file)
            else:
                uncertain_files.append(file)
        
        print(f"📊 Automatic categorization:")
        print(f"   - Likely train images: {len(train_files)}")
        print(f"   - Likely no-train images: {len(no_train_files)}")
        print(f"   - Uncertain: {len(uncertain_files)}")
        
        # Ask user for confirmation before moving
        if train_files or no_train_files:
            proceed = input(f"\nMove automatically categorized files? (y/n): ").strip().lower()
            if proceed == 'y':
                self.create_dataset_structure()
                
                # Move train files
                for file in train_files:
                    dest = self.train_present_dir / file.name
                    shutil.copy2(file, dest)
                    print(f"   📋 Moved to train_present: {file.name}")
                
                # Move no-train files
                for file in no_train_files:
                    dest = self.no_train_dir / file.name
                    shutil.copy2(file, dest)
                    print(f"   📋 Moved to no_train: {file.name}")
                
                print(f"✅ Moved {len(train_files + no_train_files)} files")
        
        if uncertain_files:
            print(f"\n🤔 Manual review needed for {len(uncertain_files)} files:")
            for file in uncertain_files[:10]:  # Show first 10
                print(f"   - {file.name}")
            if len(uncertain_files) > 10:
                print(f"   ... and {len(uncertain_files) - 10} more")
    
    def create_sample_dataset(self, num_samples: int = 10):
        """Create sample placeholder files for testing."""
        print(f"🧪 Creating sample dataset with {num_samples} files per category...")
        
        self.create_dataset_structure()
        
        # Create sample files (just empty files for structure testing)
        for i in range(num_samples):
            train_file = self.train_present_dir / f"sample_train_{i+1:03d}.jpg"
            no_train_file = self.no_train_dir / f"sample_no_train_{i+1:03d}.jpg"
            
            train_file.touch()
            no_train_file.touch()
        
        print(f"✅ Created {num_samples * 2} sample files")
        print(f"📝 Note: These are empty placeholder files. Replace with real images before training.")


def main():
    """Main function for dataset preparation."""
    print("🚂 Train Detection Dataset Helper")
    print("=" * 35)
    
    helper = DatasetHelper()
    
    while True:
        print("\n🔧 Available actions:")
        print("1. Create dataset structure")
        print("2. Validate existing dataset")
        print("3. Organize images by filename keywords")
        print("4. Create sample dataset (for testing)")
        print("5. Show expected structure")
        print("6. Exit")
        
        choice = input("\nSelect an action (1-6): ").strip()
        
        if choice == "1":
            helper.create_dataset_structure()
        
        elif choice == "2":
            stats = helper.validate_dataset()
            
        elif choice == "3":
            source_dir = input("Enter path to source images directory: ").strip()
            if source_dir:
                helper.organize_images_by_keyword(source_dir)
        
        elif choice == "4":
            num_samples = input("Number of sample files per category (default: 10): ").strip()
            try:
                num_samples = int(num_samples) if num_samples else 10
                helper.create_sample_dataset(num_samples)
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == "5":
            helper.print_dataset_structure()
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()