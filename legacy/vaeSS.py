from sklearn.model_selection import train_test_split
from image_data_processing import read_zip
from utilsVAE import bootstrap_sample_size_planning_sequential
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from utilsVAE import plt_bxplt, normalize_images, reconstruct_images, save_scores, save_data, fit_learning_curve 
import time

def main():
    """
    Main function to load the dataset, preprocess the images, train models using bootstrapping, 
    and save the results. Both CNN and RF models are trained on original and VAE-reconstructed data.
    """

    # Load and preprocess data
    zip_path = '/content/gdrive/MyDrive/Colab Notebooks/test.zip'
    classes = ['COVID19', 'NORMAL', 'PNEUMONIA']
    start_name = 'test'
    common_size = (64, 64)
    
    images, labels = read_zip(zip_path, start_name, classes, common_size)
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
    
    Xtrain_norm = normalize_images(X_train)
    Xtest_norm = normalize_images(X_test)

    # Load model and reconstruct images
    vaeModel = tf.keras.models.load_model('/content/gdrive/MyDrive/Colab Notebooks/vae_model')
    Xtrain_vae = reconstruct_images(vaeModel, Xtrain_norm)
    Xtest_vae = reconstruct_images(vaeModel, Xtest_norm)

    # Define ratios
    ratio_s = np.round(np.arange(0.05,0.81,0.05),2) #np.round(np.linspace(0.05, 0.9, 15), 2)
    scores_list, saliency_map_list, scores_list_RF, feature_importances_list, scores_list_vae, saliency_map_list_vae, feature_importances_list_vae, scores_list_vae_RF = [], [], [], [], [], [], [], []

    # Perform bootstrapping for each ratio and collect scores
    for ratio in ratio_s:
        print(f"Processing ratio: {ratio}")
        
        print(f"Start of CNN on original training for processing ratio: {ratio}")
        start_time = time.time()
        scores, saliency_map = bootstrap_sample_size_planning_sequential(Xtrain_norm, to_categorical(y_train, len(classes)), Xtest_norm, to_categorical(y_test, len(classes)), 3, Classtype = 'CNN', n_iterations=20, sample_size_ratio=ratio)
        end_time = time.time()
        print(f"Bootstrap for CNN -Originaltraining- execution time for ratio {ratio}: {end_time - start_time} seconds")
        scores_list.append(scores)
        saliency_map_list.append(saliency_map)
        
        print(f"Start of RF on original training for processing ratio: {ratio}")
        start_time = time.time()
        scores_RF, feature_importances_RF = bootstrap_sample_size_planning_sequential(Xtrain_norm, y_train, Xtest_norm, y_test, 3, Classtype = 'RF', n_iterations=20, sample_size_ratio=ratio)
        end_time = time.time()
        print(f"Bootstrap for RF -Originaltraining- execution time for ratio {ratio}: {end_time - start_time} seconds")
        scores_list_RF.append(scores_RF)
        feature_importances_list.append(feature_importances_RF)
        
        print(f"Start of CNN on VAE reconstructed for processing ratio: {ratio}")
        start_time = time.time()
        scores_vae, saliency_map_vae = bootstrap_sample_size_planning_sequential(Xtrain_vae, to_categorical(y_train, len(classes)), Xtest_vae, to_categorical(y_test, len(classes)), 3, Classtype = 'CNN', n_iterations=20, sample_size_ratio=ratio)
        end_time = time.time()
        print(f"Bootstrap for CNN -VAEreconstructed- execution time for ratio {ratio}: {end_time - start_time} seconds")
        scores_list_vae.append(scores_vae)
        saliency_map_list_vae.append(saliency_map_vae)
        
        print(f"Start of RF on VAE reconstructed for processing ratio: {ratio}")
        start_time = time.time()
        scores_vae_RF, feature_importances_vae = bootstrap_sample_size_planning_sequential(Xtrain_vae, y_train, Xtest_vae, y_test, 3, Classtype = 'RF', n_iterations=20, sample_size_ratio=ratio)
        end_time = time.time()
        print(f"Bootstrap for RF -VAEreconstructed- execution time for ratio {ratio}: {end_time - start_time} seconds\n")
        scores_list_vae_RF.append(scores_vae_RF)
        feature_importances_list_vae.append(feature_importances_vae)


    # Save scores to files
    save_scores([scores_list, scores_list_RF, scores_list_vae, scores_list_vae_RF],
                ['scores_list', 'scores_list_RF', 'scores_list_vae', 'scores_list_vae_RF'])
    # Adjust the call to save_data in your main function to include the additional data
    save_data([(scores_list, saliency_map_list), (scores_list_RF, feature_importances_list), 
               (scores_list_vae, saliency_map_list_vae), (scores_list_vae_RF, feature_importances_list_vae)],
              ['CNN_scores_saliency', 'RF_scores_importances', 'CNN_VAE_scores_saliency', 'RF_VAE_scores_importances'])
    
    _, _, _, predicted_value_CNN = fit_learning_curve(scores_list, ratio_s, [0.9], '/content/gdrive/MyDrive/Colab Notebooks/', 'CNN')
    _, _, _, predicted_value_CNN_VAE = fit_learning_curve(scores_list_vae, ratio_s, [0.9], '/content/gdrive/MyDrive/Colab Notebooks/', 'CNN_VAE')
    _, _, _, predicted_value_RF = fit_learning_curve(scores_list_RF, ratio_s, [0.9], '/content/gdrive/MyDrive/Colab Notebooks/', 'RF')
    _, _, _, predicted_value_RF_VAE = fit_learning_curve(scores_list_vae_RF, ratio_s, [0.9], '/content/gdrive/MyDrive/Colab Notebooks/', 'RF_VAE')
    
    
    # Visualize
    plt_bxplt(scores_list, scores_list_vae, ratio_s, save_path='/content/gdrive/MyDrive/Colab Notebooks/box_plot_both.png')
    plt_bxplt(scores_list_RF, scores_list_vae_RF, ratio_s, save_path='/content/gdrive/MyDrive/Colab Notebooks/box_plot_both_RF.png')

if __name__ == "__main__":
    main()
