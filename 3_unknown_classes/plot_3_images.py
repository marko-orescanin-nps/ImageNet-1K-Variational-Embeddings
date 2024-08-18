
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



alaska = '/data/kraken/coastal_project/coastal_proj_code/coastal_alaska/coastal_alaska_resized/test/A509AK_DSC_1193.jpg'
cable = '/data/kraken/coastal_project/coastal_proj_code/orange_cable_pics_resized/test/img8.jpeg'
florida = '/data/kraken/coastal_project/coastal_proj_code/florida_northern_5/florida_northern_5_resized/test/2004_0815_r026s05.jpg'


# def plot_images(image_paths):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     for i, path in enumerate(image_paths):
#         img = mpimg.imread(path)
#         axes[i].imshow(img)
#         axes[i].axis('off')

#     # plt.show()
#     plt.subplots_adjust(wspace=0.05, hspace=0)
#     plt.savefig('/data/kraken/coastal_project/coastal_proj_code/all_plots/3_unk_classes/3_imgs.png')

def plot_images(image_paths, labels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, label in zip(axes, labels):
        ax.axis('off')
        ax.set_title(label, fontsize=26)

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        img = mpimg.imread(path)
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig('/data/kraken/coastal_project/coastal_proj_code/all_plots/3_unk_classes/3_imgs.png')

# Example usage:
# image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
# labels = ['Label 1', 'Label 2', 'Label 3']
# plot_images(image_paths, labels)


image_paths = [florida, alaska, cable]
labels = ['Florida Post Hurricane','Alaska','Orange Cable']
plot_images(image_paths,labels)