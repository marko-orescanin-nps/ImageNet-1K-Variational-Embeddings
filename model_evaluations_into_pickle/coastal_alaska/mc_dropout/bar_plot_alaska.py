import matplotlib.pyplot as plt

aleatoric_uncertainty = [0.4683967, 0.46769574, 0.47017705, 0.46834755, 0.4670713, 0.46624568, 0.46849293, 0.46902564, 0.4669774, 0.46512944]

# Epistemic Uncertainties
epistemic_uncertainty = [0.3573778, 0.36184263, 0.36281076, 0.36395922, 0.3622134, 0.36303827, 0.36265448, 0.3644279, 0.3636898, 0.36244446]

# Calculate mean values
mean_aleatoric = sum(aleatoric_uncertainty) / len(aleatoric_uncertainty)
mean_epistemic = sum(epistemic_uncertainty) / len(epistemic_uncertainty)

# Plotting the bar graph
plt.figure(figsize=(6, 6))
plt.bar(['Aleatoric', 'Epistemic'], [mean_aleatoric, mean_epistemic], color=['skyblue', 'lightgreen'])

# Y-axis label
plt.ylabel('Mean Uncertainty Values')

# Title
plt.title('Mean Aleatoric vs Epistemic Uncertainty MC DROPOUT Alaska')

# Show the plot
plt.tight_layout()
plt.savefig(f"/data/kraken/coastal_project/coastal_alaska/mcdropout_alaska_bar_graph.png")
plt.close()