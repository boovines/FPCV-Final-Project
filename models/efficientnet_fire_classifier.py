import tensorflow as tf
from tensorflow.keras import layers, models

def build_fire_classifier(img_size=224, dropout=0.3):
    """
    Build an EfficientNet-B0 based binary classifier for fire detection.
    
    Args:
        img_size: Input image size (default: 224)
        dropout: Dropout rate for regularization (default: 0.3)
    
    Returns:
        Compiled Keras model for fire vs. not-fire classification
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Normalize inputs to [0, 1]
    x = layers.Rescaling(1./255)(inputs)
    
    # Note: Using weights=None due to Keras 3 compatibility issues
    # For production, consider using TensorFlow 2.15 or earlier
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights=None,  # Training from scratch
        input_tensor=None
    )
    base.trainable = True  # train all layers
    
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

