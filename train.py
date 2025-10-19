import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from models.efficientnet_fire_classifier import build_fire_classifier
from tqdm import tqdm

IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data"
EPOCHS = 10

def main():
    """
    Train the EfficientNet-B0 fire classifier on fire/non_fire image dataset.
    """
    print("🔥 Loading datasets...")
    
    # Load training dataset
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Load validation dataset
    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    # Improve performance with caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("📊 Building model...")
    model = build_fire_classifier(img_size=IMG_SIZE)
    
    print(f"🚀 Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "fire_model.h5", 
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=3, 
                restore_best_weights=True,
                monitor="val_loss",
                verbose=1
            )
        ]
    )

    print("💾 Saving final model...")
    model.save("fire_classifier_final.keras")
    
    print("✅ Training complete!")
    print(f"📈 Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"📈 Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"📈 Final validation AUC: {history.history['val_auc'][-1]:.4f}")

if __name__ == "__main__":
    main()

