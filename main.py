import os
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
random.seed(42)

# Define paths
TINY_IMAGENET_PATH = "/Users/ruksatkhanshayoni/Downloads/tiny-imagenet-200/train"
OUTPUT_DATASET_PATH = "/Users/ruksatkhanshayoni/Downloads/tiny-imagenet-output"


TRAIN_PATH = os.path.join(OUTPUT_DATASET_PATH, "train")
VAL_PATH = os.path.join(OUTPUT_DATASET_PATH, "val")
TEST_PATH = os.path.join(OUTPUT_DATASET_PATH, "test")

# Define dataset parameters
NUM_CLASSES = 100
IMAGES_PER_CLASS = 500
IMAGE_SIZE = 256  # Required for AlexNet preprocessing

# Create directories for the new dataset
for split in ["train", "test", "val"]:
    os.makedirs(os.path.join(OUTPUT_DATASET_PATH, split), exist_ok=True)

# Select 100 random classes ignoring non-directory files
all_classes = [d for d in sorted(os.listdir(TINY_IMAGENET_PATH)) if os.path.isdir(os.path.join(TINY_IMAGENET_PATH, d))]
selected_classes = random.sample(all_classes, NUM_CLASSES)

# Access nested directories for images
selected_class_dirs = {cls: Path(TINY_IMAGENET_PATH) / cls / "images" for cls in selected_classes}

# Prepare dataset
train_images = []
test_images = []
val_images = []


for class_name in selected_class_dirs:
    class_path = os.path.join(TINY_IMAGENET_PATH, class_name, "images")
    images = sorted(os.listdir(class_path))[:IMAGES_PER_CLASS]  # Take first 500 images

    # Shuffle images to ensure randomness
    random.shuffle(images)

# Shuffle to distribute classes evenly for training, testing and validation
random.shuffle(train_images)
random.shuffle(test_images)
random.shuffle(val_images)

# Function to preprocess, resize and center crop the image to 256x256
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Original Shape
    h, w, _ = img.shape
    print(f"Original shape: {h}x{w}")  # Debug print for original size

    #resize
    scale = IMAGE_SIZE / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))

    print(f"Resized shape: {new_h}x{new_w}")  # Debug print for resized size

    # Center crop 256x256
    start_x = (new_w - IMAGE_SIZE) // 2
    start_y = (new_h - IMAGE_SIZE) // 2
    img = img[start_y:start_y + IMAGE_SIZE, start_x:start_x + IMAGE_SIZE]

    print(f"Cropped shape: {img.shape}")  # Debug print for cropped size

    return img

# Function to split and move images into their respective directories
def split_and_move_images():
    for class_name in selected_classes:
        class_dir = os.path.join(TINY_IMAGENET_PATH, class_name, "images")
        images = sorted(os.listdir(class_dir))[:IMAGES_PER_CLASS]  # Select 500 images per class

        # Shuffle the images for randomness
        random.shuffle(images)

        # Split into training, validation, and testing sets
        train_images = images[:300] #for training, 30000 images
        val_images = images[300:400] # for validation 10000 images
        test_images = images[400:500] # for testing 10000 images

        # Create directories for each class in the corresponding set
        class_train_dir = os.path.join(TRAIN_PATH, class_name)
        class_val_dir = os.path.join(VAL_PATH, class_name)
        class_test_dir = os.path.join(TEST_PATH, class_name)

        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)
        os.makedirs(class_test_dir, exist_ok=True)

        # Saving the training images
        for img_name in train_images:
            img_path = os.path.join(class_dir, img_name)
            img = preprocess_image(img_path)
            if img is not None:
                cv2.imwrite(os.path.join(class_train_dir, img_name), img)

        # Saving the validation images
        for img_name in val_images:
            img_path = os.path.join(class_dir, img_name)
            img = preprocess_image(img_path)
            if img is not None:
                cv2.imwrite(os.path.join(class_val_dir, img_name), img)

        # Saving the testing images
        for img_name in test_images:
            img_path = os.path.join(class_dir, img_name)
            img = preprocess_image(img_path)
            if img is not None:
                cv2.imwrite(os.path.join(class_test_dir, img_name), img)


# Function to visualize the image before and after preprocessing
def visualize_preprocessing(class_name, image_name):
    # Define original image path
    original_img_path = os.path.join(TINY_IMAGENET_PATH, class_name, "images", image_name)

    # Read the original image
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Original image {image_name} not found!")
        return
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    processed_img = preprocess_image(original_img_path)
    if processed_img is None:
        print(f"Error in processing image {image_name}!")
        return
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    # Plotting original and preprocessed images
    fig, ax = plt.subplots(1, 2, figsize=(2, 1))  # Reduced size to avoid pixelation

    # Original image plot
    ax[0].imshow(original_img)
    ax[0].set_title("Original")
    ax[0].axis("off")

    # Processed image plot
    ax[1].imshow(processed_img)
    ax[1].set_title("Processed")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


# Select a random class and image to visualize
selected_class = random.choice(selected_classes)  # Random class
selected_image = random.choice(os.listdir(os.path.join(TINY_IMAGENET_PATH, selected_class, "images")))  # Random image

# Visualize the images
visualize_preprocessing(selected_class, selected_image)

# Call the function to split and move the images
split_and_move_images()

print("Dataset preparation is complete!")