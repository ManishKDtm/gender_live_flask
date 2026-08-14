import tensorflow as tf

# Load model
model = tf.keras.models.load_model('gender_classification_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite!")
print("File: model.tflite")