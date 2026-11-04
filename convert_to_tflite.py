"""Convert the trained Keras model to TFLite for the Flutter app.

Run: .venv\\Scripts\\python convert_to_tflite.py
Writes fruit_model.tflite + labels.txt into ../fruit_recognition_app/assets/
and checks the TFLite output against Keras on one test image.
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'fruit_recognition_app', 'assets')
MODEL_H5 = os.path.join(HERE, 'trained_model', 'efficientnet_b0_fruit_veg.h5')

os.makedirs(ASSETS, exist_ok=True)

labels = sorted(
    d for d in os.listdir(os.path.join(HERE, 'dataset', 'train'))
    if os.path.isdir(os.path.join(HERE, 'dataset', 'train', d))
)
assert len(labels) == 36, f'expected 36 classes, got {len(labels)}'
with open(os.path.join(ASSETS, 'labels.txt'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(labels))

model = tf.keras.models.load_model(MODEL_H5)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# float16 weights halve-ish the file size; activations stay float32.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_path = os.path.join(ASSETS, 'fruit_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(converter.convert())
print(f'[OK] {tflite_path} ({os.path.getsize(tflite_path) / 1e6:.1f} MB)')
print(f'[OK] {len(labels)} labels')

# Sanity check: Keras vs TFLite on the first test image of every class.
# Raw 0-255 pixels: Keras 3 EfficientNetB0 rescales/normalizes internally.
interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

max_diff = 0.0
mismatches = 0
for label in labels:
    sample_dir = os.path.join(HERE, 'dataset', 'test', label)
    sample = os.path.join(sample_dir, os.listdir(sample_dir)[0])
    x = img_to_array(load_img(sample, target_size=(224, 224)))[None]

    keras_out = model.predict(x, verbose=0)[0]
    interp.set_tensor(inp['index'], x.astype(inp['dtype']))
    interp.invoke()
    tflite_out = interp.get_tensor(out['index'])[0]

    max_diff = max(max_diff, float(np.abs(keras_out - tflite_out).max()))
    if keras_out.argmax() != tflite_out.argmax():
        mismatches += 1
        print(f'  MISMATCH {sample}: keras {labels[keras_out.argmax()]} '
              f'vs tflite {labels[tflite_out.argmax()]}')

print(f'[CHECK] {len(labels)} test images, {mismatches} argmax mismatches, '
      f'max abs diff {max_diff:.5f}')
