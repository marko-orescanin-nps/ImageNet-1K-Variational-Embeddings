import matplotlib.pyplot as plt

# Aleatoric uncertainty and epistemic uncertainty lists
aleatoric_uncertainty = [0.26281697, 0.26278505, 0.26280913, 0.2628203, 0.26279855, 0.26280656, 0.2628163, 0.26281357, 0.26280856, 0.2628128]
epistemic_uncertainty = [0.00038297847, 0.00038020127, 0.00038138242, 0.00038224595, 0.0003822455, 0.00038149953, 0.00038347038, 0.0003837157, 0.00038205535, 0.00038288653]

# Calculate mean values
mean_aleatoric = sum(aleatoric_uncertainty) / len(aleatoric_uncertainty)
mean_epistemic = sum(epistemic_uncertainty) / len(epistemic_uncertainty)

# Plotting the bar graph
plt.figure(figsize=(6, 6))
plt.bar(['Aleatoric', 'Epistemic'], [mean_aleatoric, mean_epistemic], color=['skyblue', 'lightgreen'])

# Y-axis label
plt.ylabel('Mean Uncertainty Values')

# Title
plt.title('Mean Aleatoric vs Epistemic Uncertainty (VI FLIPOUT IMAGENET)')

# Show the plot
plt.tight_layout()
plt.savefig(f"/data/kraken/coastal_project/coastal_proj_code/resnet_VI_flipout/imagenet/plots/flipout_imagenet_bar_graph.png")
plt.close()