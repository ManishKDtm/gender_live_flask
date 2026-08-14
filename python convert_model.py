import tensorflow as tf
import json
import numpy as np
import os

def convert_keras_to_tfjs(model_path, output_dir):
    """Convert Keras model to TensorFlow.js format manually"""
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model architecture
    model_json = model.to_json()
    
    # Create weights manifest
    weights = model.get_weights()
    
    # Save all weights to a single binary file
    weight_data = []
    total_size = 0
    
    for i, w in enumerate(weights):
        flat_w = w.flatten().astype('float32')
        weight_data.append(flat_w)
        total_size += flat_w.size * 4  # 4 bytes per float32
    
    # Concatenate all weights
    all_weights = np.concatenate(weight_data)
    all_weights.tofile(os.path.join(output_dir, 'group1-shard1of1.bin'))
    
    # Create model.json
    model_config = {
        "modelTopology": json.loads(model_json),
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": [{
                "name": "dense/kernel",
                "shape": list(weights[0].shape),
                "dtype": "float32"
            }]
        }],
        "format": "layers-model",
        "generatedBy": "manual-converter",
        "convertedBy": "custom-script",
        "version": "1.0"
    }
    
    # Save model.json
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model converted successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {output_dir}/model.json")
    print(f"  - {output_dir}/group1-shard1of1.bin")
    print(f"Total model size: {total_size / (1024*1024):.2f} MB")
    
    return True

if __name__ == "__main__":
    # Convert your model
    convert_keras_to_tfjs('gender_classification_model.h5', 'tfjs_model')