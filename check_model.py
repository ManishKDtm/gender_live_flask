import tensorflow as tf

# Load model
model = tf.keras.models.load_model('gender_classification_model.h5')
print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("\nModel summary:")
model.summary()