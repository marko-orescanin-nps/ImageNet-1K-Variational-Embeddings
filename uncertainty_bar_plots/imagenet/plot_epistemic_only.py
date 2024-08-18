import matplotlib.pyplot as plt
import numpy as np


#IMAGENET

#MCDROPOUT

epistemic_uncertainty_imagenet_mcdrop = [0.10942321, 0.10923628, 0.10905153, 0.10922606, 0.109203406, 0.10915725, 0.10897642, 0.10893499, 0.10912325, 0.10910049]
total_entropy_imagenet_mcdrop  = [0.92369014, 0.9231403, 0.92315423, 0.92357653, 0.92373437, 0.9232123, 0.9226884, 0.9233933, 0.9235088, 0.92255384]
mean_epistemic_imagenet_mcdrop = sum(epistemic_uncertainty_imagenet_mcdrop) / len(epistemic_uncertainty_imagenet_mcdrop)
mean_total_entropy_imagenet_mcdrop = sum(total_entropy_imagenet_mcdrop) / len(total_entropy_imagenet_mcdrop)


#FLIPOUT

epistemic_uncertainty_imagenet_flipout =  [0.0010805706, 0.0010756943, 0.0010781296, 0.0010795287, 0.0010797279, 0.0010770863, 0.0010807514, 0.001081081, 0.001078288, 0.00107947]
total_entropy_imagenet_flipout  = [0.66912717, 0.6690642, 0.6691036, 0.6691325, 0.66906875, 0.66908187, 0.66909665, 0.66912323, 0.66909814, 0.6691106]
mean_epistemic_imagenet_flipout = sum(epistemic_uncertainty_imagenet_flipout) / len(epistemic_uncertainty_imagenet_flipout)
mean_total_entropy_imagenet_flipout = sum(total_entropy_imagenet_flipout) / len(total_entropy_imagenet_flipout)


#ORANGE CABLE


#MCDROPOUT
epistemic_uncertainty_cable_mcdrop = [0.22426707, 0.23463705, 0.22347222, 0.23229133, 0.227141, 0.22783186, 0.22858413, 0.22937955, 0.21903639, 0.22377148]
total_entropy_cable_mcdrop = [2.4290419, 2.4849634, 2.4677105, 2.4738586, 2.4560483, 2.460987, 2.4618266, 2.4587805, 2.433287, 2.457631]
mean_epistemic_cable_mcdrop = sum(epistemic_uncertainty_cable_mcdrop) / len(epistemic_uncertainty_cable_mcdrop)
mean_total_entropy_cable_mcdrop = sum(total_entropy_cable_mcdrop) / len(total_entropy_cable_mcdrop)


#FLIPOUT


epistemic_uncertainty_cable_flipout = [0.05151533, 0.050560184, 0.055708643, 0.04001056, 0.059229482, 0.03970462, 0.04476236, 0.04810731, 0.03531187, 0.03647526]
total_entropy_cable_flipout =  [0.057225607, 0.059014227, 0.06374602, 0.04872875, 0.06804271, 0.04859013, 0.05269801, 0.05708306, 0.043112148, 0.0435184]
mean_epistemic_cable_flipout = sum(epistemic_uncertainty_cable_flipout) / len(epistemic_uncertainty_cable_flipout)
mean_total_entropy_cable_flipout = sum(total_entropy_cable_flipout) / len(total_entropy_cable_flipout)




species = ("MC Dropout", "VI Flipout")
batch_1 = {
    'ImageNet': (mean_epistemic_imagenet_mcdrop, mean_epistemic_imagenet_flipout),
    'Orange Cables': (mean_epistemic_cable_mcdrop, mean_epistemic_cable_flipout),
}


x = np.arange(len(species))  
width = 0.25  
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in batch_1.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel("Mean Epistemic Uncertainty",fontsize=18)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(species, fontsize=18)
ax.tick_params(axis='y', which='both', labelsize=18)
ax.legend(loc='upper right',fontsize=14)

plt.savefig(f"/data/kraken/coastal_project/coastal_proj_code/paper_bar_plots/imagenet/epistemic_only.png")



