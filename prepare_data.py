"""
Prepare dataset for fire detection binary classification.
Converts YOLO detection format to fire/non_fire classification format.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Source dataset path
DATASET_PATH = Path.home() / ".cache/kagglehub/datasets/pengbo00/home-fire-dataset/versions/1"

# Destination paths
DATA_DIR = Path("data")
FIRE_DIR = DATA_DIR / "fire"
NON_FIRE_DIR = DATA_DIR / "non_fire"

def setup_directories():
    """Create fire and non_fire directories."""
    FIRE_DIR.mkdir(parents=True, exist_ok=True)
    NON_FIRE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directories: {FIRE_DIR} and {NON_FIRE_DIR}")

def get_image_class(label_file):
    """
    Determine if image contains fire based on label file.
    Returns: 'fire' if label exists (contains fire), 'non_fire' if no label
    """
    if not label_file.exists():
        return 'non_fire'
    
    # Check if file has content
    content = label_file.read_text().strip()
    if not content:
        return 'non_fire'
    
    # If there are labels, it contains fire
    return 'fire'

def copy_images_from_split(split_name, fire_count, non_fire_count):
    """Copy images from a dataset split (train/val/test) to fire/non_fire folders."""
    
    images_dir = DATASET_PATH / split_name / "images"
    labels_dir = DATASET_PATH / split_name / "labels"
    
    if not images_dir.exists():
        print(f"⚠️  {split_name} images directory not found, skipping...")
        return fire_count, non_fire_count
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    print(f"\n📂 Processing {split_name} split ({len(image_files)} images)...")
    
    for img_path in tqdm(image_files, desc=f"  {split_name}"):
        # Get corresponding label file
        label_path = labels_dir / (img_path.stem + ".txt")
        
        # Determine class
        img_class = get_image_class(label_path)
        
        # Copy to appropriate directory
        if img_class == 'fire':
            dest = FIRE_DIR / f"{split_name}_{img_path.name}"
            shutil.copy2(img_path, dest)
            fire_count += 1
        else:
            dest = NON_FIRE_DIR / f"{split_name}_{img_path.name}"
            shutil.copy2(img_path, dest)
            non_fire_count += 1
    
    return fire_count, non_fire_count

def main():
    """Main function to prepare the dataset."""
    print("🔥 FIRE DETECTION - DATASET PREPARATION")
    print("=" * 60)
    
    # Check if source dataset exists
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("Please run test.py first to download the dataset.")
        return
    
    print(f"📦 Source dataset: {DATASET_PATH}")
    print(f"📁 Destination: {DATA_DIR}")
    print()
    
    # Setup directories
    setup_directories()
    
    # Process all splits
    fire_count = 0
    non_fire_count = 0
    
    for split in ['train', 'val', 'test']:
        fire_count, non_fire_count = copy_images_from_split(split, fire_count, non_fire_count)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"🔥 Fire images:     {fire_count:,}")
    print(f"❄️  Non-fire images: {non_fire_count:,}")
    print(f"📊 Total images:    {fire_count + non_fire_count:,}")
    print(f"⚖️  Class balance:   {fire_count / (fire_count + non_fire_count) * 100:.1f}% fire")
    print()
    print("🚀 Ready to train! Run: python train.py")
    print()

if __name__ == "__main__":
    main()

