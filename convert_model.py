import tensorflow as tf


def load_trained_model():
    # Load original model, ignoring custom objects
    original_model = tf.keras.models.load_model('app/models/ml/final_model.keras', compile=False,
                                                custom_objects={'RandomRotation': None, 'RandomFlip': None})

    # Get trained weights
    weights = original_model.get_weights()

    # Create clean model architecture
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Rescaling(1. / 255)(inputs)
    base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights=None)
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Set trained weights
    model.set_weights(weights)
    model.save('app/models/ml/inference_model.keras')
    return model


# Load and save model
model = load_trained_model()