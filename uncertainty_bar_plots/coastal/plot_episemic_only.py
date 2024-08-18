import matplotlib.pyplot as plt
import numpy as np
#ALASKA

#These values are gathered from the files of plot_alaska_mcdrop.py, plot_cable.py, etc. 
#These files mentioned above calculate the epispemic and aleatoric values from the picke files and output a number. Those numbers 
#are compiled into the lists below and then plotted on a bar plot.
#MCDROPOUT
epistemic_uncertainty_alaska_mcdrop = [0.3573778, 0.36184263, 0.36281076, 0.36395922, 0.3622134, 0.36303827, 0.36265448, 0.3644279, 0.3636898, 0.36244446]
total_entropy_alaska_mcdrop = [0.82577455, 0.8295384, 0.8329878, 0.8323067, 0.8292846, 0.82928395, 0.83114743, 0.8334535, 0.83066726, 0.82757396]
mean_epistemic_alaska_mcdrop = sum(epistemic_uncertainty_alaska_mcdrop) / len(epistemic_uncertainty_alaska_mcdrop)
mean_total_entropy_alaska_mcdrop = sum(total_entropy_alaska_mcdrop) / len(total_entropy_alaska_mcdrop)

#FLIPOUT
epistemic_uncertainty_alaska_flipout = [0.03372672, 0.033458598, 0.0336816, 0.03354431, 0.03366582, 0.03399759, 0.03390709, 0.032748595, 0.03404665, 0.03309805]
total_entropy_alaska_flipout = [0.3601864, 0.35973454, 0.35965672, 0.36090434, 0.36068457, 0.36075446, 0.36026412, 0.35914505, 0.36003205, 0.35994902]
mean_epistemic_alaska_flipout = sum(epistemic_uncertainty_alaska_flipout) / len(epistemic_uncertainty_alaska_flipout)
mean_total_entropy_alaska_flipout = sum(total_entropy_alaska_flipout) / len(total_entropy_alaska_flipout)

#FLORIDA

#MCDROPOUT 
epistemic_uncertainty_florida_mcdrop = [0.06536288, 0.061322864, 0.06791166, 0.065155216, 0.06936892, 0.0665648, 0.071382955, 0.06769857, 0.067792736, 0.06967855]
total_entropy_florida_mcdrop = [0.1366969, 0.129521, 0.14322387, 0.14034219, 0.14599614, 0.14132203, 0.1439424, 0.14106354, 0.14399938, 0.14595948]
mean_epistemic_florida_mcdrop = sum(epistemic_uncertainty_florida_mcdrop) / len(epistemic_uncertainty_florida_mcdrop)
mean_total_entropy_florida_mcdrop = sum(total_entropy_florida_mcdrop) / len(total_entropy_florida_mcdrop)

#FLIPOUT
epistemic_uncertainty_florida_flipout = [0.00070258364, 0.00076054485, 0.0006875726, 0.00081629603, 0.00063089747, 0.0007607984, 0.00074461795, 0.00097203226, 0.0007657061, 0.00085669267]
total_entropy_florida_flipout = [0.012507716, 0.012260319, 0.011833811, 0.012007059, 0.011630265, 0.011906633, 0.011947422, 0.012194989, 0.011726516, 0.012539334]
mean_epistemic_florida_flipout = sum(epistemic_uncertainty_florida_flipout) / len(epistemic_uncertainty_florida_flipout)
mean_total_entropy_florida_flipout = sum(total_entropy_florida_flipout) / len(total_entropy_florida_flipout)

#ORIGINAL COASTAL

#MCDROPOUT
epistemic_uncertainty_original_mcdrop = [0.055584177, 0.05446641, 0.052533094, 0.054806005, 0.052860573, 0.05395947, 0.05300803, 0.05458558, 0.054274574, 0.052653052]
total_entropy_original_mcdrop = [0.16866897, 0.166037, 0.16664398, 0.16913435, 0.16738652, 0.16640992, 0.16410486, 0.1671045, 0.16752535, 0.16595182]
mean_epistemic_original_mcdrop = sum(epistemic_uncertainty_original_mcdrop) / len(epistemic_uncertainty_original_mcdrop)
mean_total_entropy_original_mcdrop = sum(total_entropy_original_mcdrop) / len(total_entropy_original_mcdrop)

#FLIPOUT 
epistemic_uncertainty_original_flipout = [0.0019606801, 0.00210394, 0.0019551422, 0.0020076423, 0.001964364, 0.0022618782, 0.0019084789, 0.0018780987, 0.0020608136, 0.0019188019]
total_entropy_original_flipout = [0.03355936, 0.033733983, 0.033367544, 0.033519305, 0.033064988, 0.03399055, 0.033137437, 0.033657443, 0.033914696, 0.033604063]
mean_epistemic_original_flipout = sum(epistemic_uncertainty_original_flipout) / len(epistemic_uncertainty_original_flipout)
mean_total_entropy_original_flipout = sum(total_entropy_original_flipout) / len(total_entropy_original_flipout)



species = ("MC Dropout", "VI Flipout")
batch_1 = {
    'Original Coastal': (mean_epistemic_original_mcdrop, mean_epistemic_original_flipout),
    'Florida Post Hurricane': (mean_epistemic_florida_mcdrop, mean_epistemic_florida_flipout),
    'Alaska': (mean_epistemic_alaska_mcdrop, mean_epistemic_alaska_flipout),
}

x = np.arange(len(species))  
width = 0.25  
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in batch_1.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel("Mean Epistemic Uncertainty", fontsize=18)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(species, fontsize=18)
ax.tick_params(axis='y', which='both', labelsize=18)
ax.legend(loc='upper right',fontsize=14)

plt.savefig(f"/data/kraken/coastal_project/coastal_proj_code/paper_bar_plots/coastal/epistemic_only.png")









