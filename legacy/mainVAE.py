from image_data_processing import load_and_preprocess_data, load_and_preprocess_data_augment, datasets_to_arrays
from training import train_vae, train_vae_augment
from utilsVAE import plot_train_val_loss, visualize_evaluation_metrics
import numpy as np
import tensorflow as tf
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def predict_metrcis_save(model, data, augment = False):
    """
    Predicts test data using the VAE model, computes evaluation metrics (MSE, MAE, SSIM, PSNR), 
    and saves the metrics to a file.
    
    Parameters:
    - model: The trained VAE model.
    - data: The test dataset (TensorFlow Dataset if augment=True, NumPy array if augment=False).
    - augment: Boolean flag indicating whether the dataset is augmented (True) or not (False).
    
    Returns:
    - mse_scores: Mean Squared Error scores for the test dataset.
    - mae_scores: Mean Absolute Error scores for the test dataset.
    - ssim_scores: Structural Similarity Index scores for the test dataset.
    - psnr_scores: Peak Signal-to-Noise Ratio scores for the test dataset.
    """

    if augment == False:
        reconstr_test = model.predict(data)
        mse_scores = np.mean(np.square(data - reconstr_test), axis=(1, 2, 3))
        mae_scores = np.mean(np.abs(data - reconstr_test), axis=(1, 2, 3))
        ssim_scores = np.array([ssim(data[i], reconstr_test[i], data_range=1, channel_axis=-1) for i in range(len(data))])
        psnr_scores = np.array([psnr(data[i], reconstr_test[i], data_range=1) for i in range(len(data))]) 
    else:
        reconstr_test_batches = []
        orig_test_batches = []
        for x_batch, _ in data:
            reconstr_batch = model.predict(x_batch)
            reconstr_test_batches.append(reconstr_batch)
            orig_test_batches.append(x_batch.numpy() if isinstance(x_batch, tf.Tensor) else x_batch)
        # Concatenate all batches together
        reconstr_test = np.concatenate(reconstr_test_batches, axis=0)
        orig_test = np.concatenate(orig_test_batches, axis=0)
        
        mse_scores = np.mean(np.square(orig_test - reconstr_test), axis=(1, 2, 3))
        mae_scores = np.mean(np.abs(orig_test - reconstr_test), axis=(1, 2, 3))
        ssim_scores = np.array([ssim(orig_test[i], reconstr_test[i], data_range=1, channel_axis=-1) for i in range(len(orig_test))])
        psnr_scores = np.array([psnr(orig_test[i], reconstr_test[i], data_range=1) for i in range(len(orig_test))])
    
    print("Mean Absolute Error (MSE) on Test Set:", np.mean(mse_scores))
    print("Mean Absolute Error (MAE) on Test Set:", np.mean(mae_scores))
    print("Mean Structural Similarity Index (SSIM) on Test Set:", np.mean(ssim_scores))
    print("Mean Peak Signal-to-Noise Ratio (PSNR) on Test Set:", np.mean(psnr_scores))
    
    metrics = {
        'mse_scores': mse_scores,
        'mae_scores': mae_scores,
        'ssim_scores': ssim_scores,
        'psnr_scores': psnr_scores
    }
    
    with open('evaluation_metrics.pkl', 'wb') as file:
        pickle.dump(metrics, file)
        
    return mse_scores, mae_scores, ssim_scores, psnr_scores

def main():
    """
    Main function to load the dataset, train the VAE model, evaluate it, and save the results.
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and preprocess data
    zip_path = '/content/gdrive/MyDrive/Colab Notebooks/Dataset.zip'
    classes = ['COVID', 'NORMAL', 'PNEUMONIA']
    start_name = 'Dataset'
    common_size = (64, 64)
    latent_dim = 64 * 64
    batch_size = 64
    augment = False
    
    if augment == False:
        train_ds, val_ds, test_ds = load_and_preprocess_data(zip_path, start_name, classes, common_size, batch_size)
        X_train, X_val, X_test = datasets_to_arrays(train_ds, val_ds, test_ds)
        encoder, decoder, vae, history = train_vae(X_train, X_val, latent_dim=latent_dim, batch_size=64, epochs=200)
        mse_scores, mae_scores, ssim_scores, psnr_scores = predict_metrcis_save(vae, X_test, augment = False)
        
    else:
        augmented_train_ds, val_ds, test_ds = load_and_preprocess_data_augment(zip_path, start_name, classes, common_size, batch_size)
        encoder, decoder, vae, history = train_vae_augment(augmented_train_ds, val_ds, latent_dim=latent_dim, epochs=200)
        mse_scores, mae_scores, ssim_scores, psnr_scores = predict_metrcis_save(vae, test_ds, augment = True)
    
    # Save History
    with open('/content/gdrive/MyDrive/Colab Notebooks/history_vae', 'wb') as file_pi:
              pickle.dump(history.history, file_pi)
    # Save the model
    vae.save('/content/gdrive/MyDrive/Colab Notebooks/vae_model', save_format='tf')
    
    # Save encoder, decoder
    encoder.save('/content/gdrive/MyDrive/Colab Notebooks/encoder', save_format='tf')
    decoder.save('/content/gdrive/MyDrive/Colab Notebooks/decoder', save_format='tf')
    
    visualize_evaluation_metrics(mse_scores, mae_scores, ssim_scores, psnr_scores, 'evaluation_visualization.png')
    plot_train_val_loss(history.history)
    
    return mse_scores, mae_scores, ssim_scores, psnr_scores

if __name__ == "__main__":
    main()
    