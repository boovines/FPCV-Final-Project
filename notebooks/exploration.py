"""
Fire Detection - Exploratory Data Analysis

This script explores the fire detection dataset and evaluates the trained model.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

IMG_SIZE = 224
BATCH_SIZE = 32


def check_dataset():
    """Check dataset structure and class distribution."""
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    
    data_dir = Path("../data")
    fire_dir = data_dir / "fire"
    non_fire_dir = data_dir / "non_fire"

    num_fire = len(list(fire_dir.glob("*"))) if fire_dir.exists() else 0
    num_non_fire = len(list(non_fire_dir.glob("*"))) if non_fire_dir.exists() else 0
    total = num_fire + num_non_fire

    print(f"Fire images: {num_fire}")
    print(f"Non-fire images: {num_non_fire}")
    print(f"Total images: {total}")
    
    if total > 0:
        print(f"Class balance: {num_fire / total * 100:.1f}% fire, {num_non_fire / total * 100:.1f}% non-fire")
    else:
        print("⚠️  No images found! Please add images to data/fire/ and data/non_fire/")
    print()


def visualize_samples():
    """Visualize sample images from the dataset."""
    print("=" * 60)
    print("2. SAMPLE IMAGES")
    print("=" * 60)
    
    try:
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "../data",
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
        class_names = dataset.class_names
        print(f"Classes detected: {class_names}")
        
        # Display sample images
        plt.figure(figsize=(15, 10))
        for images, labels in dataset.take(1):
            for i in range(min(9, len(images))):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]], fontsize=12)
                plt.axis("off")
        
        plt.suptitle("Sample Images from Dataset", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig("sample_images.png", dpi=100, bbox_inches='tight')
        print("✅ Sample images saved to: sample_images.png")
        plt.close()
        print()
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure images are in data/fire/ and data/non_fire/\n")


def load_and_evaluate_model():
    """Load the trained model and evaluate on validation set."""
    print("=" * 60)
    print("3. MODEL EVALUATION")
    print("=" * 60)
    
    model_path = "../fire_classifier_final.keras"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at '{model_path}'")
        print("Train the model first using: python train.py\n")
        return None, None
    
    print("📦 Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!\n")
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    print()
    
    # Load validation dataset
    try:
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "../data",
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE
        )
        
        class_names = val_ds.class_names
        
        # Evaluate model
        print("Evaluating model on validation set...")
        loss, accuracy, auc = model.evaluate(val_ds)
        
        print(f"\n📊 Validation Metrics:")
        print(f"  Loss:     {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  AUC:      {auc:.4f}")
        print()
        
        return model, val_ds
        
    except Exception as e:
        print(f"❌ Error loading validation data: {e}\n")
        return model, None


def visualize_predictions(model, val_ds):
    """Visualize model predictions on sample images."""
    if model is None or val_ds is None:
        return
    
    print("=" * 60)
    print("4. PREDICTION VISUALIZATION")
    print("=" * 60)
    
    class_names = val_ds.class_names
    
    plt.figure(figsize=(15, 10))
    for images, labels in val_ds.take(1):
        predictions = model.predict(images, verbose=0)
        
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            true_label = class_names[labels[i]]
            pred_prob = predictions[i][0]
            pred_label = "fire" if pred_prob > 0.5 else "non_fire"
            
            # Color based on correctness
            color = "green" if true_label == pred_label else "red"
            
            plt.title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.3f})", 
                     color=color, fontsize=10)
            plt.axis("off")
    
    plt.suptitle("Predictions on Validation Set (Green=Correct, Red=Wrong)", 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=100, bbox_inches='tight')
    print("✅ Predictions saved to: predictions.png")
    plt.close()
    print()


def generate_confusion_matrix(model, val_ds):
    """Generate and plot confusion matrix."""
    if model is None or val_ds is None:
        return
    
    print("=" * 60)
    print("5. CONFUSION MATRIX & CLASSIFICATION REPORT")
    print("=" * 60)
    
    class_names = val_ds.class_names
    
    # Collect all predictions and labels
    y_true = []
    y_pred = []
    
    print("Generating predictions on full validation set...")
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend((predictions > 0.5).astype(int).flatten())
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches='tight')
    print("✅ Confusion matrix saved to: confusion_matrix.png\n")
    plt.close()
    
    # Classification report
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print()


def main():
    """Run all exploratory data analysis steps."""
    print("\n🔥 FIRE DETECTION - EXPLORATORY DATA ANALYSIS 🔥\n")
    
    # Step 1: Check dataset
    check_dataset()
    
    # Step 2: Visualize samples
    visualize_samples()
    
    # Step 3: Load and evaluate model
    model, val_ds = load_and_evaluate_model()
    
    # Step 4: Visualize predictions
    visualize_predictions(model, val_ds)
    
    # Step 5: Generate confusion matrix
    generate_confusion_matrix(model, val_ds)
    
    print("=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\n📊 Generated files:")
    print("  - sample_images.png")
    print("  - predictions.png")
    print("  - confusion_matrix.png")
    print("\n💡 Experiment Ideas:")
    print("  - Try different architectures (ResNet, MobileNet, etc.)")
    print("  - Add data augmentation")
    print("  - Fine-tune the EfficientNet base layers")
    print("  - Adjust hyperparameters (learning rate, dropout, etc.)")
    print("  - Collect more diverse data")
    print()


if __name__ == "__main__":
    main()

