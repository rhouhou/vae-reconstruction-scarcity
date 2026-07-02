# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:04:50 2024

@author: Rola
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from utilsVAE import plot_confidence_intervals

scores_list_RF = (pd.read_csv('C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/scores_list_RF.csv', header=None)).to_numpy()
scores_list_RF_vae = (pd.read_csv('C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/scores_list_vae_RF.csv', header=None)).to_numpy()

_, p_value_RF = mannwhitneyu(scores_list_RF, scores_list_RF_vae)
print(f"RF model between RF & RF-VAE: {p_value_RF}")
 
ratio_s = np.round(np.arange(0.05,0.81,0.05),2)
p_value_heatmap_both = []
# Mann-Whitney U Test
for i in range(16):
    for j in range(16):
        _, p_value_RF = mannwhitneyu(scores_list_RF[i,:], scores_list_RF[j,:])
        p_value_heatmap_both.append(p_value_RF)
        print(f"RF model between {ratio_s[i]} and {ratio_s[j]}: {p_value_RF}")
p_value_matrix_both = np.array(p_value_heatmap_both).reshape(16, 16)

significant_matrix_RF = np.where(p_value_matrix_RF < 0.05, p_value_matrix_RF, np.nan)
significant_matrix_RF_VAE = np.where(p_value_matrix_RF_VAE < 0.05, p_value_matrix_RF_VAE, np.nan)

plt.figure(figsize=(10, 8))
# Use 'mask' parameter in seaborn.heatmap or handle NaN values directly in matplotlib.pyplot.imshow
plt.imshow(significant_matrix_RF, cmap='viridis', interpolation='nearest', vmin=min(p_value_matrix_RF.min(), p_value_matrix_RF_VAE.min())
               , vmax=max(p_value_matrix_RF.max(), p_value_matrix_RF_VAE.max()))

# Add a colorbar with label
cbar = plt.colorbar(label='P-value')
cbar.set_label('P-value', rotation=270, labelpad=18)

# Set tick labels
plt.xticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, rotation=45, fontsize=14)
plt.yticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, fontsize=14)

# Labels
plt.xlabel('Sample Size Ratio', fontsize=18)
plt.ylabel('Sample Size Ratio', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('C:/Users/rolah/OneDrive/Desktop/significant_p_both.png')
plt.show()




scores_list_RF = (pd.read_csv('C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/scores_list_RF.csv', header=None)).to_numpy()
scores_list_RF_vae = (pd.read_csv('C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/scores_list_vae_RF.csv', header=None)).to_numpy()

ratio_s = np.round(np.arange(0.05,0.81,0.05),2)
p_value_heatmap_RF = []
# Mann-Whitney U Test
for i in range(16):
    for j in range(16):
        _, p_value_RF = mannwhitneyu(scores_list_RF[i,:], scores_list_RF[j,:])
        p_value_heatmap_RF.append(p_value_RF)
        print(f"RF model between {ratio_s[i]} and {ratio_s[j]}: {p_value_RF}")
        
p_value_heatmap_RF_VAE = []
# Mann-Whitney U Test
for i in range(16):
    for j in range(16):
        _, p_value_RF = mannwhitneyu(scores_list_RF_vae[i,:], scores_list_RF_vae[j,:])
        p_value_heatmap_RF_VAE.append(p_value_RF)
        print(f"RF model between {ratio_s[i]} and {ratio_s[j]}: {p_value_RF}")

# Reshape your flat list into a 15x15 array
p_value_matrix_RF = np.array(p_value_heatmap_RF).reshape(16, 16)
p_value_matrix_RF_VAE = np.array(p_value_heatmap_RF_VAE).reshape(16, 16)

significant_matrix_RF = np.where(p_value_matrix_RF < 0.05, p_value_matrix_RF, np.nan)
significant_matrix_RF_VAE = np.where(p_value_matrix_RF_VAE < 0.05, p_value_matrix_RF_VAE, np.nan)

plt.figure(figsize=(10, 8))
# Use 'mask' parameter in seaborn.heatmap or handle NaN values directly in matplotlib.pyplot.imshow
plt.imshow(significant_matrix_RF, cmap='viridis', interpolation='nearest', vmin=min(p_value_matrix_RF.min(), p_value_matrix_RF_VAE.min())
               , vmax=max(p_value_matrix_RF.max(), p_value_matrix_RF_VAE.max()))

# Add a colorbar with label
cbar = plt.colorbar(label='P-value')
cbar.set_label('P-value', rotation=270, labelpad=18)

# Set tick labels
plt.xticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, rotation=45, fontsize=14)
plt.yticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, fontsize=14)

# Labels
plt.xlabel('Sample Size Ratio', fontsize=18)
plt.ylabel('Sample Size Ratio', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('C:/Users/rolah/OneDrive/Desktop/significant_p_RF.png')
plt.show()


###############################


# Example load (you should load your actual data)
# scores_list_RF = pd.read_csv('path_to_scores_list_RF.csv', header=None).to_numpy()
# scores_list_RF_vae = pd.read_csv('path_to_scores_list_vae_RF.csv', header=None).to_numpy()

# Verify both arrays have the same number of columns for comparison

num_comparisons = scores_list_RF_vae.shape[0]
p_between_more = []
for i in range(num_comparisons):
    for j in range(i + 1, num_comparisons + 1):  # Adjusted to compare up to row 16 (zero-indexed)
        # Perform the t-test for the specified rows
        stat, p = mannwhitneyu(scores_list_RF_vae[i, :], scores_list_RF[j - 1, :])
        p_between_more.append(p)
        print(f'Comparing scores_list_RF_vae row {i} with scores_list_RF row {j - 1}: t-statistic = {stat}, p-value = {p}')

p_value_matrix_RF = np.array(p_between_more).reshape(16, 16)
significant_matrix = np.where(np.array(p_between_more < 0.05), p_between_more, np.nan)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Placeholder for loading your data
# scores_list_RF = pd.read_csv('path_to_scores_list_RF.csv', header=None).to_numpy()
# scores_list_RF_vae = pd.read_csv('path_to_scores_list_vae_RF.csv', header=None).to_numpy()

# Initialize the results matrix with NaN values
results_p_values = np.full((16, 16), np.nan)

# Perform comparisons, filling the results matrix with p-values
for i in range(16):
    for j in range(16):  # Adjust as necessary for your comparison logic
        if i<j:
            stat, p = mannwhitneyu(scores_list_RF_vae[i, :], scores_list_RF[j, :])
            if p<0.05:
                results_p_values[i, j] = p
    

# Generating the heatmap for the results
plt.figure(figsize=(12, 10))
# Use mask to avoid plotting NaN values
mask = np.isnan(results_p_values)
sns.heatmap(results_p_values.T, annot=True, fmt=".2g", cmap="viridis", cbar=True)
plt.title('Heatmap of p-values for Comparisons')
plt.xlabel('scores_list_RF_vae Index')
plt.ylabel('scores_list_RF Index')
plt.show()

plt.figure(figsize=(10, 8))
# Use 'mask' parameter in seaborn.heatmap or handle NaN values directly in matplotlib.pyplot.imshow
plt.imshow(results_p_values.T, cmap='viridis',aspect='auto', edgecolors='black')
cbar = plt.colorbar(label='P-value')
cbar.set_label('P-value', rotation=270, labelpad=18)
plt.xticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, rotation=45, fontsize=14)
plt.yticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, fontsize=14)
plt.xlabel('Sample Size Ratio VAE', fontsize=18)
plt.ylabel('Sample Size Ratio', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('C:/Users/rolah/OneDrive/Desktop/significant_p_VAE_no_significant.png')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'results_p_values.T' is your transposed results array
# Replace this with your actual transposed data
data = results_p_values.T

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap
ax = sns.heatmap(data, annot=True, fmt=".2g", cmap='viridis', cbar=True, 
            cbar_kws={"label": "P-value"}, linewidths=0.5, linecolor='black')

# Rotate the x and y axis labels if necessary
tick_labels = np.round(np.arange(0.05, 0.85, 0.05),2)  # Replace with your actual tick labels if different
tick_positions = np.arange(0, len(tick_labels)) + 0.5  # Offsetting by 0.5 to center the ticks

# Apply the ticks to the heatmap
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels)

plt.xticks(rotation=45, fontsize=14)
plt.yticks(rotation=45, fontsize=14)

#plt.xticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, rotation=45, fontsize=14)
#plt.yticks(ticks=np.arange(len(ratio_s)), labels=ratio_s, rotation=45, fontsize=14)

# Set the labels and title using fontsize parameters
plt.xlabel('Sample Size Ratio VAE', fontsize=18)
plt.ylabel('Sample Size Ratio', fontsize=18)

# Adjust layout for tight fit and appropriate space for the colorbar label
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('C:/Users/rolah/OneDrive/Desktop/overall_no_significant.png')
# Display the plot
plt.show()









###

scores_list_RF = pd.read_csv('C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/scores_list_RF.csv', header=None)
scores_list_RF_vae = pd.read_csv('C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/scores_list_vae_RF.csv', header=None)

plt.figure(figsize=(12, 8))
positions_without_vae = np.arange(len(ratio_s))

# Plotting box plots
plt.boxplot(scores_list_RF, widths=0.4, patch_artist=True,
                             boxprops=dict(facecolor="skyblue"))

# Plotting confidence intervals
plot_confidence_intervals(positions_without_vae, scores_list_RF, 'orange', '95% CI', alpha=0.3)

# Customizing the plots
plt.title('Boxplot of Accuracy Scores by Sample Size Ratio with CI')
plt.xlabel('Sample Size Ratio')
plt.ylabel('Accuracy Score')
#middle_positions = (positions_without_vae + positions_vae) / 2
#plt.xticks(middle_positions, [str(r) for r in ratio_s], rotation=45)
plt.grid(True)

# Simplifying legend creation
boxplot_legend = [plt.Line2D([0], [0], color='skyblue', marker='s', markersize=10, label='w/o VAE'),
                  plt.Line2D([0], [0], color='lightgreen', marker='s', markersize=10, label='VAE'),
                  plt.Line2D([0], [0], color='orange', marker='_', markersize=10, linewidth=4, label='95% CI')]
plt.legend(handles=boxplot_legend, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(save_path)
plt.close()


#####

data = np.load("C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/RF_scores_importances.npz")
feature_RF = data['feature_importances_or_saliency_maps']
mean_bootstrap = np.mean(feature_RF, axis=1)*255
std_boostrap = np.round(np.std(feature_RF, axis=(1,2)),3)
reshape_data = np.reshape(mean_bootstrap, (16, 64,64,3))

fig, axs = plt.subplots(4, 4, figsize=(10, 10))  # Create a 4x4 grid of subplots
for i, ax in enumerate(axs.flat):
    # Assuming your data is in RGB format
    #grayscale_image = 0.2989 * reshape_data[i, :, :, 0] + 0.5870 * reshape_data[i, :, :, 1] + 0.1140 * reshape_data[i, :, :, 2]
    #ax.imshow(grayscale_image)
    ax.imshow(reshape_data[i,:,:,:])
    ax.set_title(f'Sample ratio: {ratio_s[i]}\n StD: {std_boostrap[i]}')
    ax.axis('off')  # Hide axis for clarity

plt.tight_layout()
fig.savefig('C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/RF_feature_importance.png', format='png', dpi=300, bbox_inches='tight')

data_VAE = np.load("C:/Users/rolah/OneDrive/Desktop/To check/my Work DKFZ/Results-VAE-17 March 2024/RF_VAE_scores_importances.npz")
feature_RF_VAE = data_VAE['feature_importances_or_saliency_maps']
mean_bootstrap = np.mean(feature_RF_VAE, axis=1)*255
std_boostrap = np.round(np.std(feature_RF_VAE, axis=(1,2)),3)
reshape_data = np.reshape(mean_bootstrap, (16, 64,64,3))

fig, axs = plt.subplots(4, 4, figsize=(10, 10))  # Create a 4x4 grid of subplots
for i, ax in enumerate(axs.flat):
    # Assuming your data is in RGB format
    #grayscale_image = 0.2989 * reshape_data[i, :, :, 0] + 0.5870 * reshape_data[i, :, :, 1] + 0.1140 * reshape_data[i, :, :, 2]
    #ax.imshow(grayscale_image)
    ax.imshow(reshape_data[i,:,:,:])
    ax.set_title(f'Sample ratio: {ratio_s[i]}\n StD: {std_boostrap[i]}')
    ax.axis('off')  # Hide axis for clarity

plt.tight_layout()
fig.savefig('C:/Users/rolah/OneDrive/Desktop/RF_feature_importance_VAE.png', format='png', dpi=300, bbox_inches='tight')

_, p_value_RF = mannwhitneyu(feature_RF, feature_RF_VAE)
print(f"RF modelfeature:  {p_value_RF}")


mean_bootstrap = np.reshape(feature_RF, (16,20*64*64*3))
mean_bootstrap_VAE = np.reshape(feature_RF_VAE, (16,20*64*64*3))
p_value_heatmap_both = []
# Mann-Whitney U Test
for i in range(16):
        _, p_value_both = mannwhitneyu(mean_bootstrap[i,:], mean_bootstrap_VAE[i,:])
        p_value_heatmap_both.append(p_value_RF)
        print(f"RF model feature {ratio_s[i]}: {p_value_RF}")



