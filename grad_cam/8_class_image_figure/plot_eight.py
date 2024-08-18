
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as mpimg



coastal_cliff_img ='/data/kraken/coastal_project/test/CoastalCliffs/5171_resized.jpg'
coastal_rocky_img = '/data/kraken/coastal_project/test/CoastalRocky/200900807_resized.jpg'
coastal_waterway ='/data/kraken/coastal_project/test/CoastalWaterWay/IMG_0539_SecOP_Sum12_Pt3_resized.jpg'
dunes = '/data/kraken/coastal_project/test/Dunes/7699_resized.jpg'
mm ='/data/kraken/coastal_project/test/ManMadeStructures/IMG_0329_SecABD_Sum12_Pt1_resized.jpg'
salt_marsh ='/data/kraken/coastal_project/test/SaltMarshes/IMG_1663_SecHKL_Sum12_Pt2_resized.jpg'
sandy_beach ='/data/kraken/coastal_project/test/SandyBeaches/IMG_2664_SecABD_Sum12_Pt1_resized.jpg'
tidal_flat ='/data/kraken/coastal_project/test/TidalFlats/IMG_1394_SecBC_Spr12_resized.jpg'

coastal_images = [coastal_cliff_img, coastal_rocky_img, coastal_waterway, dunes, mm, salt_marsh, sandy_beach, tidal_flat,
              
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/21may_good_enumerate_8_class_image_figure_cam_0.jpg',
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/21may_good_enumerate_8_class_image_figure_cam_1.jpg',
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/29may_good_enumerate_8_class_image_figure_cam_2.jpg',
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/21may_good_enumerate_8_class_image_figure_cam_3.jpg',
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/21may_good_enumerate_8_class_image_figure_cam_4.jpg',
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/21may_good_enumerate_8_class_image_figure_cam_5.jpg',
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/21may_good_enumerate_8_class_image_figure_cam_6.jpg',
'/data/kraken/coastal_project/coastal_proj_code/grad_cam/21may_good_enumerate_8_class_image_figure_cam_7.jpg']


fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i in range(4):
    for j in range(4):
        # Calculate the index in the flat image_paths list
        index = i * 4 + j

        # Load the image using Matplotlib's imread
        img = mpimg.imread(coastal_images[index])

        # Display the image on the corresponding subplot
        axes[i, j].imshow(img)
        axes[i, j].axis('off')  # Turn off axis labels
    # Add labels (a, b, c, ...) to each image
        label = chr(ord('a') + index)  # Convert index to corresponding alphabet character
        axes[i, j].text(0.05, 0.95, label, transform=axes[i, j].transAxes, color='white',
                        fontsize=12, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(wspace=0,hspace=0)

plt.savefig('/data/kraken/coastal_project/coastal_proj_code/grad_cam/29may_good_20runs_eight_together.png')