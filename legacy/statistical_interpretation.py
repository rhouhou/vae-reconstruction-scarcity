import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import seaborn as sns

def load_and_reshape_features(filepath, len_cases=16, img_size=64, type_method='RF'):
    """
    Loads feature importances or saliency maps from a file and reshapes the data based on the method used (RF or CNN).
    
    Parameters:
    - filepath: Path to the .npz file containing the feature importances or saliency maps.
    - len_cases: Number of cases (images) to process.
    - img_size: Size of the images (assumed to be square, e.g., 64x64).
    - type_method: Type of model ('RF' for Random Forest, 'CNN' for Convolutional Neural Network).
    
    Returns:
    - mean_bootstrap: Averaged feature importance or saliency maps over the bootstrapped iterations.
    - reshape_data: The reshaped feature importance or saliency maps, ready for visualization.
    """
    data = np.load(filepath)
    feature_importances_or_saliency_maps = data['feature_importances_or_saliency_maps']
    mean_bootstrap = np.mean(feature_importances_or_saliency_maps, axis=1) * 255

    # Reshape the data based on the type of model used (Random Forest or CNN)
    if type_method == 'RF':
        reshape_data = np.reshape(mean_bootstrap, (len_cases, img_size, img_size, 3))
    elif type_method == 'CNN':
        reshape_data = np.reshape(mean_bootstrap, (len_cases, img_size * img_size))

    return mean_bootstrap, reshape_data

def compute_p_values(data1, data2, num_features, compare_across_samples=False):
    """
    Computes p-values for feature importance comparison using the Wilcoxon signed-rank test.
    
    Parameters:
    - data1, data2: Input data for comparison.
    - num_features: Number of features (e.g., 16 or 64*64 for CNN).
    - compare_across_samples: Boolean flag to determine if data is compared across samples (True) or features (False).
    
    Returns:
    - p_values: Array of p-values for each feature or comparison.
    """
    p_values = np.full(num_features, np.nan)
    
    for i in range(num_features):
        if compare_across_samples:
            feature_importances1 = data1[:, i]
            feature_importances2 = data2[:, i]
        else:
            feature_importances1 = data1[i, :]
            feature_importances2 = data2[i, :]

        if np.all(feature_importances1 - feature_importances2 == 0):
            print(f"All differences between the paired samples for feature {i} are zero.")
        else:
            stat, p = wilcoxon(feature_importances1, feature_importances2)
            p_values[i] = p

    return p_values


def plot_boxplot_with_pvalues(data1, data2, ratio_s, p_values, adjust_p=1.5, save_path='box_plot_with_pvalues.png'):
    """
    Creates a boxplot to visualize feature importances for two models and annotates the plot with p-values for 
    significant differences.
    
    Parameters:
    - data1: Feature importance or saliency map data for the first model (e.g., original model).
    - data2: Feature importance or saliency map data for the second model (e.g., VAE-reconstructed model).
    - ratio_s: Array of sample size ratios.
    - p_values: Array of p-values calculated from statistical tests to indicate significant differences.
    - adjust_p: Adjustment value for placing the significance markers (stars) above the boxplots.
    - save_path: File path where the generated plot will be saved.

    Returns:
    - None. The plot is saved to the specified file path.
    """
    plt.figure(figsize=(12, 8))
    positions_without_vae = np.arange(len(ratio_s)) * 2.0 - 0.3
    positions_vae = np.arange(len(ratio_s)) * 2.0 + 0.3

    # Plotting box plots
    plt.boxplot(data1.T, positions=positions_without_vae, widths=0.4, patch_artist=True, boxprops=dict(facecolor="skyblue"))
    plt.boxplot(data2.T, positions=positions_vae, widths=0.4, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    
    # Annotating significant differences with '*' on the boxplot
    for i, p_value in enumerate(p_values):
        if p_value < 0.05:
            y = max(np.max(data1[:, i]), np.max(data2[:, i])) + adjust_p
            plt.text((positions_without_vae[i] + positions_vae[i]) / 2, y, '*', ha='center', fontsize=32, color='red')

    plt.title('Boxplot of Feature Importance by Sample Size Ratio')
    plt.xlabel('Sample Size Ratio')
    plt.ylabel('Feature Importance')
    middle_positions = (positions_without_vae + positions_vae) / 2
    plt.xticks(middle_positions, [str(r) for r in ratio_s], rotation=45)
    plt.grid(True)

    # Legend
    boxplot_legend = [plt.Line2D([0], [0], color='skyblue', marker='s', markersize=10, label='w/o VAE'),
                      plt.Line2D([0], [0], color='lightgreen', marker='s', markersize=10, label='VAE')]
    plt.legend(handles=boxplot_legend, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(data, title="Heatmap", cmap='hot', save_path=None):
    """
    Plots a heatmap for a given dataset.
    
    Parameters:
    - data: 2D array of data to plot in the heatmap.
    - title: Title of the plot.
    - cmap: Colormap for the heatmap.
    - save_path: Optional file path to save the heatmap image.
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(data, cmap=cmap, annot=False)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Example of loading and reshaping data
mean_bootstrap_RF, reshape_data_RF = load_and_reshape_features("path_to_RF_file", len_cases=16, img_size=64, type_method='RF')
mean_bootstrap_RF_VAE, reshape_data_RF_VAE = load_and_reshape_features("path_to_RF_VAE_file", len_cases=16, img_size=64, type_method='RF')
mean_bootstrap_CNN, reshape_data_CNN = load_and_reshape_features("path_to_CNN_file", len_cases=16, img_size=64, type_method='CNN')
mean_bootstrap_CNN_VAE, reshape_data_CNN_VAE = load_and_reshape_features("path_to_CNN_VAE_file", len_cases=16, img_size=64, type_method='CNN')

# Compute p-values for RF and CNN
p_values_RF = compute_p_values(mean_bootstrap_RF, mean_bootstrap_RF_VAE, 16)
p_values_CNN = compute_p_values(reshape_data_CNN, reshape_data_CNN_VAE, 16)

# Define sample size ratios and create boxplots with p-values
ratio_s = np.round(np.arange(0.05, 0.81, 0.05), 2)
plot_boxplot_with_pvalues(mean_bootstrap_RF, mean_bootstrap_RF_VAE, ratio_s, p_values_RF, adjust_p=1.5, save_path='path_to_file_RF.png')
plot_boxplot_with_pvalues(reshape_data_CNN, reshape_data_CNN_VAE, ratio_s, p_values_CNN, adjust_p=0, save_path='path_to_file_CNN.png')

# Compute p-values across sample sizes and plot heatmap
p_values_RF = compute_p_values(mean_bootstrap_RF, mean_bootstrap_RF_VAE, 64 * 64 * 3, compare_across_samples=True)
p_values_CNN = compute_p_values(reshape_data_CNN, reshape_data_CNN_VAE, 64 * 64, compare_across_samples=True)

# Reshape p-values and plot heatmap for CNN
data_p_resh_CNN = np.reshape(p_values_CNN, (64, 64))
plot_heatmap(data_p_resh_CNN, title="CNN-based Method Heatmap")

# Highlight significant p-values in the heatmap
significant_mask = np.where(data_p_resh_CNN < 0.05, data_p_resh_CNN, np.nan)
plot_heatmap(significant_mask, title="Significant P-values (< 0.05) CNN", cmap="viridis")