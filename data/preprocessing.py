import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess


def get_preprocessing_function(model_name):
    """
    Mengembalikan preprocessing function sesuai model
    """
    if model_name.lower() == "mobilenet":
        return mobilenet_preprocess
    elif model_name.lower() == "vgg19":
        return vgg19_preprocess
    else:
        return None  # untuk Base CNN


def create_data_generators(
    train_dir,
    val_dir,
    img_size=(224, 224),
    batch_size=32,
    model_name="base"
):
    """
    Membuat data generator untuk training dan validation
    """

    preprocess_fn = get_preprocessing_function(model_name)

    # ==========================
    # TRAINING DATA (augmentasi)
    # ==========================
    train_datagen = ImageDataGenerator(
        rescale=None if preprocess_fn else 1./255,
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # ==========================
    # VALIDATION DATA (tanpa augmentasi)
    # ==========================
    val_datagen = ImageDataGenerator(
        rescale=None if preprocess_fn else 1./255,
        preprocessing_function=preprocess_fn
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator
