import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.optimize import curve_fit

from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from image_data_processing import read_zip

class BalancedAccuracy(Metric):
    """
    Custom TensorFlow metric class to calculate balanced accuracy.

    Balanced accuracy takes into account both true positives and true negatives,
    and averages the accuracy for each class.
    
    """

    def __init__(self, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        
        self.balanced_accuracy = self.add_weight(name='bacc', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric for each batch.
        
        Parameters:
        - y_true: True labels for the batch.
        - y_pred: Predicted labels for the batch.
        - sample_weight: Optional weights for each sample.
        """

        y_true = tf.reshape(tf.argmax(y_true, axis=1), [-1])
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), [-1])
        
        correct_predictions = tf.equal(y_true, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        self.balanced_accuracy.assign_add(accuracy)
        self.count.assign_add(1)
        
    def result(self):
        """
        Compute and return the final balanced accuracy.
        """
        return self.balanced_accuracy / self.count
    
    def reset_state(self):
        """
        Reset the state of the metric. This is called between epochs or steps.
        """

        self.balanced_accuracy.assign(0.0)
        self.count.assign(0.0)

def build_model(input_shape, num_classes):
    """
    Build a simple Convolutional Neural Network (CNN) model using TensorFlow/Keras.
    
    Parameters:
    - input_shape: Shape of the input data (e.g., (64, 64, 3) for 64x64 RGB images).
    - num_classes: Number of output classes for classification.
    
    Returns:
    - model: Compiled CNN model ready for training.
    """

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[BalancedAccuracy()])
    
    return model

def bootstrap_iteration(args):
    """
    Perform a single bootstrap iteration for a given classifier type.
    
    Parameters:
    - X_train: Training data features
    - y_train: Training data labels
    - X_test: Test data features
    - y_test: Test data labels
    - num_classes: Number of classes for classification (used in CNN case)
    - Classtype: Type of classifier ('SVM', 'RF', 'CNN')
    - sample_size_ratio: Ratio of training data to use for bootstrap resampling
    - iteration: Current iteration number (for potential logging/debugging)
    
    Returns:
    - score: Balanced accuracy score of the model
    - importances/saliency_map: Feature importances for RF, saliency map for CNN
    
    """

    X_train, y_train, X_test, y_test, num_classes, Classtype, sample_size_ratio, iteration = args
    n_size = int(len(X_train) * sample_size_ratio)
    
    X_train_resample, y_train_resample = resample(X_train, y_train, n_samples=n_size) 

    if Classtype == 'SVM':
        model = SVC(kernel='rbf')
        X_train_reshape = np.reshape(X_train_resample, (X_train_resample.shape[0], X_train_resample.shape[1]*X_train_resample.shape[2]*X_train_resample.shape[3]))
        model.fit(X_train_reshape, y_train_resample)
        X_test_reshape = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))
        y_pred = model.predict(X_test_reshape)
        score = balanced_accuracy_score(y_test, y_pred)
        return score
    
    elif Classtype == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train_reshape = np.reshape(X_train_resample, (X_train_resample.shape[0], X_train_resample.shape[1]*X_train_resample.shape[2]*X_train_resample.shape[3]))
        model.fit(X_train_reshape, y_train_resample)
        importances = model.feature_importances_
        X_test_reshape = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))
        y_pred = model.predict(X_test_reshape)
        score = balanced_accuracy_score(y_test, y_pred)
        return score, importances
    
    elif Classtype == 'CNN':
        model = build_model(X_train_resample.shape[1:], num_classes)  # Build and compile the model
        model.fit(X_train_resample, y_train_resample, epochs=10, batch_size=8, verbose=0)
        _, score = model.evaluate(X_test, y_test, verbose=0)

        with tf.GradientTape() as tape:
            X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
            tape.watch(X_test_tensor)
            predictions = model(X_test_tensor)
            loss = tf.reduce_max(predictions, axis=-1)
        grads = tape.gradient(loss, X_test_tensor)
        saliency_map = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
        return score, saliency_map

def bootstrap_sample_size_planning_sequential(X_train, y_train, X_test, y_test, num_classes, Classtype='CNN', n_iterations=100, sample_size_ratio=0.8):
    """
    Perform sequential bootstrap sampling for a given classifier type and calculate performance scores.
    
    Parameters:
    - X_train: Training data features.
    - y_train: Training data labels.
    - X_test: Test data features.
    - y_test: Test data labels.
    - num_classes: Number of classes (used for CNN case).
    - Classtype: Type of classifier ('RF', 'CNN', etc.). Default is 'CNN'.
    - n_iterations: Number of bootstrap iterations to perform. Default is 100.
    - sample_size_ratio: Ratio of training data to use for resampling. Default is 0.8.
    
    Returns:
    - For RF: Tuple of (scores, feature_importances).
    - For CNN: Tuple of (scores, saliency_maps).
    - For other classifiers: List of scores.
    """
    
    scores = []
    
    if Classtype == 'RF':
        feature_importances = []  # For RF only
    elif Classtype == 'CNN':
        saliency_maps = []  # For CNN only

    start_time = time.time()
    
    for i in range(n_iterations):
        # Create the argument tuple for each iteration
        args = (X_train, y_train, X_test, y_test, num_classes, Classtype, sample_size_ratio, i)
        
        # Perform the iteration
        result = bootstrap_iteration(args)
        
        # Collect the results
        scores.append(result[0])
        
        if Classtype == 'RF':
            feature_importances.append(result[1])
        elif Classtype == 'CNN':
            saliency_maps.append(result[1])
    
    end_time = time.time()
    print(f"Sequential execution time: {end_time - start_time} seconds")
    
    # Return based on Classtype
    if Classtype == 'RF':
        return scores, feature_importances
    elif Classtype == 'CNN':
        return scores, saliency_maps
    else:
        return scores

def plot_train_val_loss(history, title='Training and Validation Loss', save_path='training_and_validation_loss.png'):
    """
    Plots the training and validation loss.

    Parameters:
    - history: A history object returned from a Keras model's fit method.
    - title: (optional) Title of the plot.
    - save_path: (optional) File path to save the plot image.
    """
    
    train_loss = history['loss']
    val_loss = history.get('val_loss', None)  # Handle cases where validation loss might not be available
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    
    # Plot training loss
    plt.plot(epochs_range, train_loss, label='Training Loss')
    
    # Plot validation loss only if it's available
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    
    # Save the plot and show
    plt.savefig(save_path, dpi=300)
    plt.show()
    
def visualize_evaluation_metrics(mse_scores, mae_scores, ssim_scores, psnr_scores, save_path):
    """
    Visualizes evaluation metrics (MSE, MAE, SSIM, PSNR) using both histograms and violin plots.
    
    Parameters:
    - mse_scores: Array of Mean Squared Error (MSE) scores for individual images.
    - mae_scores: Array of Mean Absolute Error (MAE) scores for individual images.
    - ssim_scores: Array of Structural Similarity Index (SSIM) scores for individual images.
    - psnr_scores: Array of Peak Signal-to-Noise Ratio (PSNR) scores for individual images.
    """
    
    scores = [mse_scores, mae_scores, ssim_scores, psnr_scores]
    labels = ['MSE', 'MAE', 'SSIM', 'PSNR']
    
    # Create subplots
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))

    # Loop through each metric to create histograms and violin plots
    for i, (score, label) in enumerate(zip(scores, labels)):
        # Histogram
        axes[i, 0].hist(score, bins=20)
        axes[i, 0].set_xlabel(label)
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].set_title(f'Distribution of {label}')
        
        # Violin plot
        sns.violinplot(y=score, ax=axes[i, 1])
        axes[i, 1].set_ylabel(label)
        axes[i, 1].set_title(f'Distribution of {label}')
    
    plt.tight_layout()
    
    # Save the visualization
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confidence_intervals(positions, scores, color, label, alpha=0.3):
    """
    Plot confidence intervals for each score at specified positions.
    
    Parameters:
    - positions: List of x-axis positions for each score.
    - scores: List of score arrays for which confidence intervals are calculated.
    - color: Color for the confidence interval plot.
    - label: Label for the plot, applied only to the first interval to avoid duplicate labels.
    - alpha: Transparency level for the confidence interval shading. Default is 0.3.
    """
    
    for i, scr in enumerate(scores):
        confidence_interval = np.percentile(scr, [2.5, 97.5])
        plt.fill_betweenx(y=[confidence_interval[0], confidence_interval[1]],
                          x1=positions[i]-0.1, x2=positions[i]+0.1, color=color, alpha=alpha,
                          label=label if i == 0 else "")

def plt_bxplt(scores, scores_vae, ratio_s, save_path='/content/gdrive/MyDrive/Colab Notebooks/box_plot_both.png'):
    """
    Plots boxplots of scores for two datasets (with and without VAE) along with confidence intervals.
    
    Parameters:
    - scores: List of score arrays for models without VAE.
    - scores_vae: List of score arrays for models with VAE.
    - ratio_s: List of sample size ratios corresponding to the score arrays.
    - save_path: (optional) Path to save the plot image. Default is '/content/gdrive/MyDrive/Colab Notebooks/box_plot_both.png'.
    """

    plt.figure(figsize=(12, 8))
    
    positions = np.arange(len(ratio_s)) * 2.0
    offset = 0.3

    # Box plots for scores without VAE and with VAE
    plt.boxplot(scores, positions=positions - offset, widths=0.4, patch_artist=True, boxprops=dict(facecolor="skyblue"))
    plt.boxplot(scores_vae, positions=positions + offset, widths=0.4, patch_artist=True, boxprops=dict(facecolor="lightgreen"))

    # Plot confidence intervals
    plot_confidence_intervals(positions - offset, scores, 'orange', '95% CI', alpha=0.3)
    plot_confidence_intervals(positions + offset, scores_vae, 'orange', '', alpha=0.3)

    # Customize the plot
    plt.title('Boxplot of Accuracy Scores by Sample Size Ratio with CI')
    plt.xlabel('Sample Size Ratio')
    plt.ylabel('Accuracy Score')
    plt.xticks(positions, [str(r) for r in ratio_s], rotation=45)
    plt.grid(True)

    # Legend creation
    legend_elements = [
        plt.Line2D([0], [0], color='skyblue', marker='s', markersize=10, label='w/o VAE'),
        plt.Line2D([0], [0], color='lightgreen', marker='s', markersize=10, label='VAE'),
        plt.Line2D([0], [0], color='orange', marker='_', markersize=10, linewidth=4, label='95% CI')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    
def normalize_images(images):
    """
    Normalize image pixel values to the range [0, 1].
    
    Parameters:
    - images: NumPy array of image pixel values. Typically, image pixel values range from 0 to 255.
    
    Returns:
    - Normalized image array where pixel values are scaled to the range [0, 1].
    """
    return images.astype('float32') / 255.0

def reconstruct_images(vae_model, images, img_shape=(64, 64, 3)):
    """
    Reconstruct images using a Variational Autoencoder (VAE) model.
    
    Parameters:
    - vae_model: The trained VAE model used for reconstructing the images.
    - images: Input images to be reconstructed, typically preprocessed (e.g., normalized).
    - img_shape: The target shape of the reconstructed images. Default is (64, 64, 3).
    
    Returns:
    - Reconstructed images reshaped to the specified img_shape.
    """

    reconstr_images = vae_model.predict(images)
    return np.reshape(reconstr_images, (images.shape[0], *img_shape))

def save_scores(score_lists, file_names, base_path='/content/gdrive/MyDrive/Colab Notebooks/'):
    """Save score lists to CSV files."""
    save_data(score_lists, file_names, base_path)
        
def save_data(data_lists, file_names, base_path='/content/gdrive/MyDrive/Colab Notebooks/'):
    """
    Save score lists to CSV files by delegating the task to the save_data function.
    
    Parameters:
    - score_lists: A list of score arrays to be saved. Each element in the list represents a different set of scores.
    - file_names: A list of filenames corresponding to each score array, to be used for saving the files.
    - base_path: (optional) The directory where the CSV files will be saved. Default is '/content/gdrive/MyDrive/Colab Notebooks/'.
    """

    for data, file_name in zip(data_lists, file_names):
        if isinstance(data, tuple):
            # Save tuple data (e.g., scores and feature importances) as NPZ
            np.savez(f"{base_path}{file_name}.npz", scores=data[0], feature_importances_or_saliency_maps=data[1])
        else:
            # Save single list data as CSV
            np.savetxt(f"{base_path}{file_name}.csv", data, delimiter=",")
            
def extract_feature_resnet50(test_path, common_size, batch_size):
    """
    Extract features from images using a pre-trained ResNet50 model without the top classification layers.
    
    Parameters:
    - test_path: Path to the directory containing the images for feature extraction.
    - common_size: The target size for resizing images (e.g., (224, 224) for ResNet50).
    - batch_size: Number of images to process in each batch.
    
    Returns:
    - A NumPy array containing the mean of the extracted features for each image.
    """

    base_model = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Load images and preprocess using ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    dataset = datagen.flow_from_directory(test_path, target_size=common_size, batch_size=batch_size, class_mode=None, shuffle=False)
    
    # Extract features and flatten the output
    features_dataset = model.predict(dataset, verbose=1)
    return np.mean(features_dataset.reshape(features_dataset.shape[0], -1), axis=1)

def extract_mean_per_img(dataset):
    """
    Extract the mean value for each image across all pixel channels.
    
    Parameters:
    - dataset: A NumPy array or list of images where each image has shape (height, width, channels).
    
    Returns:
    - A NumPy array containing the mean value for each image across all pixels and channels.
    """

    return np.array([np.mean(img, axis=(0, 1, 2)) for img in dataset])

def hist_KS(data1, data2, bins, title, save_path):
    """
     Generate a histogram for two datasets and perform the Kolmogorov-Smirnov (K-S) test.
     
     Parameters:
     - data1: First dataset (e.g., scores or features).
     - data2: Second dataset for comparison.
     - bins: Number of bins to use in the histogram.
     - title: Title for the histogram plot.
     - save_path: File path where the histogram image will be saved.
     """
 
    ks_statistic, p_value = ks_2samp(data1, data2)
    print(f"K-S statistic: {ks_statistic}, P-value: {p_value}")

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(data1, bins=bins, alpha=0.5, label='Testing set - VAE trained', density=True)
    plt.hist(data2, bins=bins, alpha=0.5, label='Testing set - # source', density=True)
    plt.legend()
    plt.title(f"{title}\nK-S Statistic: {ks_statistic:.4f}, P-value: {p_value:.4f}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def extract_and_compare(test_path1, test_path2, size, type_analys, save_path, bins=50):
    """
    Helper function to extract features or image data and run the Kolmogorov-Smirnov (K-S) test.
    
    Parameters:
    - test_path1: Path to the first dataset directory (either features or images).
    - test_path2: Path to the second dataset directory (either features or images).
    - size: The target size for resizing images or features extraction (e.g., (224, 224)).
    - type_analys: Type of analysis to perform ('features' or 'images'). 
                   If 'features', it extracts using ResNet50; otherwise, it uses raw images.
    - save_path: Directory where the resulting K-S test histogram image will be saved.
    - bins: (optional) Number of bins to use for the histograms. Default is 50.
    """

    if type_analys == 'features':
        features_1 = extract_feature_resnet50(test_path1, size, 32)
        features_2 = extract_feature_resnet50(test_path2, size, 32)
    else:
        images_1, _ = read_zip(test_path1, None, None, size)
        _, X_test_1 = train_test_split(images_1, test_size=0.1, random_state=42)
        features_1 = extract_mean_per_img(X_test_1)
        
        images_2, _ = read_zip(test_path2, None, None, size)
        _, X_test_2 = train_test_split(images_2, test_size=0.3, random_state=42)
        features_2 = extract_mean_per_img(X_test_2)

    # Plot histogram and perform K-S test
    title = f'Two-Sided K-S Test on {"Image features (ResNet50)" if type_analys == "features" else "Images"}'
    hist_KS(features_1, features_2, bins, title, save_path + f'KS_hist_{type_analys}.png')

def twoS_KS(test_path1, classes1, start_name1, test_path2, classes2, start_name2, type_analys, save_path):
    """
    Main function to handle two-sample Kolmogorov-Smirnov (K-S) tests for feature or image analysis.
    
    Parameters:
    - test_path1: Path to the first dataset directory (features or images).
    - classes1: Not used in this function but could represent class labels for dataset 1.
    - start_name1: Not used in this function but could represent a prefix for images in dataset 1.
    - test_path2: Path to the second dataset directory (features or images).
    - classes2: Not used in this function but could represent class labels for dataset 2.
    - start_name2: Not used in this function but could represent a prefix for images in dataset 2.
    - type_analys: Type of analysis ('features' for ResNet50 feature extraction or 'images' for raw image analysis).
    - save_path: Path to save the resulting K-S test histogram image.
    
    Returns:
    None. The function performs the two-sample K-S test and saves the histogram result.
    """

    common_size = (224, 224) if type_analys == 'features' else (64, 64)
    extract_and_compare(test_path1, test_path2, common_size, type_analys, save_path)

def inverse_power_law(x, a, b, c):
    """
     Inverse power law model function.
     
     This function models a power law relationship in the form:
     y = a * x^(-b) + c
    
     Parameters:
     - x: Input value or array of values for which the model is applied.
     - a: Scaling factor for the power law.
     - b: Exponent that controls the rate of decay or growth.
     - c: Constant offset added to the result.
     
     Returns:
     - The computed value of the inverse power law model for input x.
     """
 
    return a * x ** (-b) + c

def prediction_uncertainty(x, params, std_devs):
    """
    Simplified uncertainty propagation for the inverse power law model.
    
    This function calculates the uncertainty in the predicted value of the
    inverse power law model, given uncertainties in the parameters (a, b, and c).
    
    Parameters:
    - x: Input value or array of values for which the uncertainty is computed.
    - params: A list or array of model parameters [a, b, c] from the inverse power law model.
    - std_devs: Standard deviations (uncertainties) for each parameter [std_dev_a, std_dev_b, std_dev_c].
    
    Returns:
    - The propagated uncertainty in the prediction based on the uncertainties in the model parameters.
    """

    a, b, _ = params
    partial_a = x ** (-b)
    partial_b = -a * x ** (-b) * np.log(x)
    partial_c = 1
    return np.sqrt((partial_a * std_devs[0]) ** 2 + (partial_b * std_devs[1]) ** 2 + (partial_c * std_devs[2]) ** 2)

def fit_learning_curve(data_list, ratio_s, new_size, save_path, name_file):
    """
    Fit an inverse power law to learning curve data, compute uncertainty, and predict future values.
    
    Parameters:
    - data_list: List of accuracy scores for different sample sizes.
    - ratio_s: List of sample sizes.
    - new_size: The sample size(s) for prediction.
    - save_path: Directory to save the plot.
    - name_file: File name for saving the plot.
    
    Returns:
    - a, b, c: Fitted parameters of the inverse power law.
    - predicted_value: Predicted accuracy for the given new sample size(s).
    """
    
    # Calculate accuracies and errors
    scores = np.array(data_list)
    accuracies = np.mean(scores, axis=1)
    errors = 1 - accuracies
    
    # Fit inverse power law model to the data
    x_range = np.linspace(min(ratio_s), max(ratio_s), len(ratio_s))
    params, covariance = curve_fit(inverse_power_law, x_range, errors)
    std_devs = np.sqrt(np.diag(covariance))
    a, b, c = params
    
    # Compute fitted values and uncertainties
    fitted_values = inverse_power_law(x_range, *params)
    uncertainty = prediction_uncertainty(x_range, params, std_devs)
    
    # Predict values for new sample size(s)
    if np.isscalar(new_size):
        predicted_value = 1 - inverse_power_law(new_size, *params)
    else:
        predicted_value = 1 - inverse_power_law(np.array(new_size), *params)
    
    # Plot learning curve and fitted curve
    plt.figure(figsize=(8, 6))
    plt.scatter(ratio_s, errors, color='blue', label='Learning Curve')
    plt.plot(x_range, fitted_values, color='red', label='Fitted Curve')
    plt.axhline(c, color='green', linestyle='--', label='Bayes Error')
    plt.fill_between(x_range, fitted_values - uncertainty, fitted_values + uncertainty, 
                     color='red', alpha=0.2, label='Uncertainty')
    
    # Annotate plot with parameter values
    plt.text(max(x_range) * 0.6, max(fitted_values) * 0.9, f'a = {a:.4f}\nb = {b:.4f}\nc = {c:.4f}', 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    
    # Plot configuration
    plt.xlabel('Sample Size')
    plt.ylabel('Error Rate')
    plt.title('Learning Curve and Inverse Power Law Fit')
    plt.legend()
    
    # Save and display the plot
    plt.savefig(f'{save_path}/{name_file}_LearningCurve.png', dpi=300)
    plt.show()
    
    return a, b, c, predicted_value