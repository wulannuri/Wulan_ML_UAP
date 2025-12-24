import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================================
# SET ROOT PROJECT PATH
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ['disgust', 'happy', 'sad']

DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

# =====================================================
# LOAD TEST DATA
# =====================================================
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=CLASS_NAMES
)

# =====================================================
# EVALUATION FUNCTION
# =====================================================
def evaluate_model(model_name):
    print(f"\n🔍 Evaluating {model_name}")

    model_path = os.path.join(MODEL_DIR, model_name)
    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    print("\n📊 Classification Report")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# =====================================================
# RUN EVALUATION
# =====================================================
if __name__ == "__main__":
    evaluate_model("base_cnn.h5")
    evaluate_model("vgg19.h5")
    evaluate_model("mobilenet.h5")
