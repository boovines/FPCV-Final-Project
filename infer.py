import tensorflow as tf
import cv2
import numpy as np
import sys
import os

IMG_SIZE = 224

def predict_frame(model, frame_path):
    """
    Predict fire probability for a single image.
    
    Args:
        model: Trained Keras model
        frame_path: Path to the image file
    
    Returns:
        Fire probability (0-1)
    """
    if not os.path.exists(frame_path):
        print(f"❌ Error: File '{frame_path}' not found!")
        return None
    
    img = cv2.imread(frame_path)
    if img is None:
        print(f"❌ Error: Unable to read image '{frame_path}'")
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img, verbose=0)[0][0]
    
    print(f"🔥 Fire probability: {pred:.3f}")
    
    if pred > 0.5:
        print(f"⚠️  FIRE DETECTED! (confidence: {pred:.1%})")
    else:
        print(f"✅ No fire detected (confidence: {(1-pred):.1%})")
    
    return pred

def main():
    """
    Load the trained model and run inference on a test image.
    """
    model_path = "fire_classifier_final.keras"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at '{model_path}'")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
    
    print("📦 Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Check if an image path was provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n🖼️  Analyzing: {image_path}")
        predict_frame(model, image_path)
    else:
        print("\n💡 Usage: python infer.py <path_to_image>")
        print("Example: python infer.py sample_fire.jpg")
        print("\nPlease provide an image path as argument.")

if __name__ == "__main__":
    main()

