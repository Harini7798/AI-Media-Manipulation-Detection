import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils import get_filename_only

dataset_path = os.path.join('.', 'split_dataset')

tmp_debug_path = os.path.join('.', 'tmp_debug')
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0  # Uses TF's EfficientNet implementation/weights
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    rescale = 1/255,    #rescale the tensor values to [0,1]
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",  #"categorical", "binary", "sparse", "input"
    batch_size = batch_size_num,
    shuffle = True
    #save_to_dir = tmp_debug_path
)

val_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0,1]
)

val_generator = val_datagen.flow_from_directory(
    directory = val_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",  #"categorical", "binary", "sparse", "input"
    batch_size = batch_size_num,
    shuffle = True
    #save_to_dir = tmp_debug_path
)

test_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0,1]
)

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = None,
    batch_size = 1,
    shuffle = False
)

# Train a CNN classifier
# Loading ImageNet weights is optional: if the weights download fails (offline or 404),
# fall back to random initialization so training can still run.
try:
    efficient_net = EfficientNetB0(
        weights='imagenet',
        input_shape=(input_size, input_size, 3),
        include_top=False,
        pooling='max',
    )
except Exception as e:
    print(f"Warning: failed to load EfficientNetB0 imagenet weights ({e}). Training from scratch (weights=None).")
    efficient_net = EfficientNetB0(
        weights=None,
        input_shape=(input_size, input_size, 3),
        include_top=False,
        pooling='max',
    )

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = os.path.join('.', 'tmp_checkpoint')
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model.h5'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )
]

# Train network
num_epochs = 20
history = model.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = custom_callbacks
)
print(history.history)

'''
# Plot results
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
'''

# load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

# ── Evaluation with metrics ──────────────────────────────────────────────────

# Use a dedicated test generator WITH labels for evaluation
eval_test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = 1,
    shuffle = False
)

# Generate predictions
eval_test_generator.reset()
preds = best_model.predict(eval_test_generator, verbose=1)
pred_probs = preds.flatten()
pred_labels = (pred_probs >= 0.5).astype(int)

# Ground-truth labels from the generator
true_labels = eval_test_generator.classes
class_names = list(eval_test_generator.class_indices.keys())  # e.g. ['fake', 'real']

# ── Metrics ──────────────────────────────────────────────────────────────────
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
cm = confusion_matrix(true_labels, pred_labels)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion Matrix (rows=true, cols=predicted):")
print(f"  Classes: {class_names}")
print(cm)
print(f"\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0))

# Per-file results
test_results = pd.DataFrame({
    "Filename": eval_test_generator.filenames,
    "True Label": true_labels,
    "Prediction": pred_probs,
    "Predicted Label": pred_labels,
})
print(test_results)

# Save metrics to JSON
metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "confusion_matrix": cm.tolist(),
    "class_names": class_names,
}
metrics_path = os.path.join(checkpoint_filepath, "eval_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to {metrics_path}")