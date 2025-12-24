import os
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# SET ROOT PROJECT PATH
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# =====================================================
# IMPORT MODEL
# =====================================================
from models.base_cnn import build_base_cnn
from models.vgg19_model import build_vgg19
from models.mobilenet_model import build_mobilenet

from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# =====================================================
# CONFIG
# =====================================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3
CLASS_NAMES = ['disgust', 'happy', 'sad']

DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR   = os.path.join(DATA_DIR, 'valid')

SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# VALIDASI DATASET
# =====================================================
assert os.path.exists(TRAIN_DIR), "Folder data/train tidak ditemukan"
assert os.path.exists(VAL_DIR), "Folder data/valid tidak ditemukan"

print("✅ Dataset ditemukan")

# =====================================================
# PREPROCESSING FUNCTION
# =====================================================
def create_data_generators(model_name="base"):
    if model_name == "vgg19":
        preprocess_fn = vgg19_preprocess
        rescale = None
    elif model_name == "mobilenet":
        preprocess_fn = mobilenet_preprocess
        rescale = None
    else:
        preprocess_fn = None
        rescale = 1./255

    train_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocess_fn
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    return train_gen, val_gen

# =====================================================
# CALLBACK
# =====================================================
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# =====================================================
# PLOT FUNCTION
# =====================================================
def plot_history(history, title):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{title} - Loss')
    plt.legend()

    plt.show()

# =====================================================
# 1️⃣ BASE CNN
# =====================================================
print("\n=== TRAINING BASE CNN ===")
train_gen, val_gen = create_data_generators("base")

base_model = build_base_cnn(IMG_SIZE + (3,), NUM_CLASSES)
history_base = base_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

base_model.save(os.path.join(SAVE_DIR, 'base_cnn.h5'))
plot_history(history_base, "Base CNN")

# =====================================================
# 2️⃣ VGG19
# =====================================================
print("\n=== TRAINING VGG19 ===")
train_gen, val_gen = create_data_generators("vgg19")

vgg_model = build_vgg19(IMG_SIZE + (3,), NUM_CLASSES)
history_vgg = vgg_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

vgg_model.save(os.path.join(SAVE_DIR, 'vgg19.h5'))
plot_history(history_vgg, "VGG19")

# =====================================================
# 3️⃣ MOBILENETV2
# =====================================================
print("\n=== TRAINING MOBILENETV2 ===")
train_gen, val_gen = create_data_generators("mobilenet")

mobilenet_model = build_mobilenet(IMG_SIZE + (3,), NUM_CLASSES)
history_mobile = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen, 
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

mobilenet_model.save(os.path.join(SAVE_DIR, 'mobilenet.h5'))
plot_history(history_mobile, "MobileNetV2")

print("\n🎉 SEMUA MODEL BERHASIL DILATIH")
print(f"📁 Model tersimpan di: {SAVE_DIR}")