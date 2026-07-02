import numpy as np
import zipfile
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def read_zip(zip_path, start_name, classes, common_size):
    """
    Read images and their labels from a zip file and resize them.
    
    Parameters:
    - zip_path: Path to the zip file containing the images.
    - start_name: Base directory name within the zip file.
    - classes: List of class names corresponding to each folder in the zip file.
    - common_size: Target size for resizing the images (e.g., (224, 224)).
    
    Returns:
    - images: A NumPy array of image data.
    - labels: A NumPy array of corresponding image labels.
    """

    images = []
    labels = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Loop through each class directory and read the images
        for label, class_name in enumerate(classes):
            # Filter for files in this directory
            class_files = [f for f in zip_ref.namelist() if f.startswith(start_name+'/'+ class_name + '/') and (f.endswith('.png') or f.endswith('.jpg'))]
    
            for image_name in class_files:
                image_data = zip_ref.read(image_name)
                image = Image.open(BytesIO(image_data))
                # Convert to RGB and resize
                image = image.convert('RGB').resize(common_size)
                image_array = np.array(image)
                images.append(image_array)
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def load_and_preprocess_data_augment(zip_path, start_name, classes, common_size, batch_size):
    """
    Load, normalize, and augment image data from a zip file.
    
    Parameters:
    - zip_path: Path to the zip file containing the images.
    - start_name: Base directory name within the zip file.
    - classes: List of class names corresponding to each folder in the zip file.
    - common_size: Target size for resizing the images (e.g., (224, 224)).
    - batch_size: Batch size for training and evaluation.
    
    Returns:
    - augmented_train_ds: Augmented training dataset.
    - val_ds: Validation dataset.
    - test_ds: Test dataset.
    """

    images, _ = read_zip(zip_path, start_name, classes, common_size)
    images_norm = images.astype('float32') / 255.0

    X_tmp, X_test = train_test_split(images_norm, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_tmp, test_size=0.25, random_state=42)
    
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2)
        ])
    
    def augment(x, y):
        x = data_augmentation(x, training=True)
        return x, y
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, X_test))
    
    augmented_train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    augmented_train_ds = augmented_train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return augmented_train_ds, val_ds, test_ds

def load_and_preprocess_data(zip_path, start_name, classes, common_size, batch_size):
    """
    Load and normalize image data from a zip file (without augmentation).
    
    Parameters:
    - zip_path: Path to the zip file containing the images.
    - start_name: Base directory name within the zip file.
    - classes: List of class names corresponding to each folder in the zip file.
    - common_size: Target size for resizing the images (e.g., (224, 224)).
    - batch_size: Batch size for training and evaluation.
    
    Returns:
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    - test_ds: Test dataset.
    """

    images, _ = read_zip(zip_path, start_name, classes, common_size)
    images_norm = images.astype('float32') / 255.0

    X_tmp, X_test = train_test_split(images_norm, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_tmp, test_size=0.25, random_state=42)


    train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, X_test))
    
    return train_ds, val_ds, test_ds

def datasets_to_arrays(augmented_train_ds, val_ds, test_ds):
    """
    Convert TensorFlow datasets into NumPy arrays.
    
    Parameters:
    - augmented_train_ds: Augmented training dataset in TensorFlow Dataset format.
    - val_ds: Validation dataset in TensorFlow Dataset format.
    - test_ds: Test dataset in TensorFlow Dataset format.
    
    Returns:
    - X_train_augmented: NumPy array of augmented training data.
    - X_val: NumPy array of validation data.
    - X_test: NumPy array of test data.
    """

    X_train_augmented = []
    for data in augmented_train_ds:
        x = data[0]
        X_train_augmented.append(x.numpy())
    X_train_augmented = np.array(X_train_augmented)

    X_val = []
    for data in val_ds:
        x = data[0]
        X_val.append(x.numpy())
    X_val = np.array(X_val)

    X_test = []
    for data in test_ds:
        x = data[0]
        X_test.append(x.numpy())
    X_test = np.array(X_test)

    print(X_train_augmented.shape)
    print(X_val.shape)
    print(X_test.shape)

    return X_train_augmented, X_val, X_test
