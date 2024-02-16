import numpy as np
import imgaug.augmenters as iaa
import imageio.v2 as imageio  # Import imageio.v2 to address deprecation warning
import os

# Define the directory containing your images
input_dir = "D:/pythonProject8/Data/Normal"
output_dir = "augmented_images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check if the input directory exists and is a directory
if not os.path.exists(input_dir):
    print(f"Input directory '{input_dir}' does not exist.")
    exit()
elif not os.path.isdir(input_dir):
    print(f"Input path '{input_dir}' is not a directory.")
    exit()

# Create an augmenter with desired augmentations
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5), # vertical flips
    iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 1.0
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # add gaussian noise
    iaa.Affine(rotate=(-45, 45)), # rotate images between -45 to 45 degrees
    iaa.Resize({"height": 224, "width": 224}) # resize images to 224x224
])

# List all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Apply augmentations to each image and save the augmented images
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image = imageio.imread(image_path)
    images_aug = [augmenter(image=image) for _ in range(10)]  # Augment each image 10 times

    # Save augmented images
    for i, image_aug in enumerate(images_aug):
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")
        imageio.imwrite(output_path, image_aug)

print("Augmentation complete!")
