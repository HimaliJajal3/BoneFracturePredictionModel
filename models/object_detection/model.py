import os
import numpy as np






label_dirs = [
    "/content/data/BoneFractureYolo8/train/labels/",
    "/content/data/BoneFractureYolo8/val/labels/",
    "/content/data/BoneFractureYolo8/test/labels/"
]

# Class IDs to merge (3 -> 4)
old_class = 3
new_class = 4

# Iterate over all directories
for label_dir in label_dirs:
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            # Read and update label file
            with open(label_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])

                # Change class 3 to class 4
                if class_id == old_class:
                    parts[0] = str(new_class)

                new_lines.append(" ".join(parts))

            # Write updated labels back to file
            with open(label_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

# Old Classes
#nc:7
# class_names = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture',
#                'humerus', 'shoulder fracture', 'wrist positive']
# New Classes
# nc: 6
# names: ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus', 'shoulder fracture', 'wrist positive']

#update YAML file

yaml_file_path = "/content/data/BoneFractureYolo8/data.yaml"

yaml_content = """
train: /content/data/BoneFractureYolo8/train/images
val: /content/data/BoneFractureYolo8/val/images
test: /content/data/BoneFractureYolo8/test/images

nc: 6
names: ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus', 'shoulder fracture', 'wrist positive']
"""

with open(yaml_file_path, "w") as file:
    file.write(yaml_content)


#Shift class IDs by 1
#Labels needs to correctly formatted with class IDs 0-5

def shift_class_ids(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    modified = False

    for line in lines:
        parts = line.strip().split()

        # Skip empty or malformed lines
        if len(parts) == 0:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            print(f"⚠️ Skipping malformed line in {label_path}: {line.strip()}")
            continue  # Skip invalid lines

        # Shift class IDs greater than 3 down by one
        if class_id > 3:
            parts[0] = str(class_id - 1)
            modified = True

        new_lines.append(" ".join(parts))

    # If modifications were made, overwrite the file
    if modified:
        with open(label_path, "w") as f:
            f.writelines("\n".join(new_lines) + "\n")

# Iterate through all label files and fix class IDs
for label_dir in label_dirs:
    if os.path.exists(label_dir):
        for label_file in os.listdir(label_dir):
            if label_file.endswith(".txt"):
                shift_class_ids(os.path.join(label_dir, label_file))

#Augmentation and Expand dataset with augmented images
# Define directories
base_dir = "/content/data/BoneFractureYolo8/"
splits = ["train", "val", "test"]

# Augmentation functions
def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

def rotate_image(image, angle):
    """Rotate image by 90, -90, or 180 degrees"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def shear_image(image, shear_factor):
    """Apply shearing transformation"""
    rows, cols, _ = image.shape
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

# Apply augmentation and save
for split in splits:
    image_dir = os.path.join(base_dir, split, "images")
    label_dir = os.path.join(base_dir, split, "labels")

    images = glob(os.path.join(image_dir, "*.jpg"))  # Change to ".png" if needed

    for img_path in images:
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path).split(".")[0]

        # Get corresponding label file
        label_path = os.path.join(label_dir, img_name + ".txt")
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            label_data = f.read()

        # Apply augmentations
        augmentations = [
            ("flipH", flip_image(img, 1)),   # Horizontal Flip
            ("flipV", flip_image(img, 0)),   # Vertical Flip
            ("rot90", rotate_image(img, 90)),  # Rotate 90°
            ("rot180", rotate_image(img, 180)),  # Rotate 180°
            ("shear", shear_image(img, 0.1)),  # Shear +10%
        ]

        for aug_name, aug_img in augmentations:
            aug_img_name = f"{img_name}_{aug_name}.jpg"
            aug_img_path = os.path.join(image_dir, aug_img_name)
            aug_label_path = os.path.join(label_dir, f"{img_name}_{aug_name}.txt")

            # Save augmented image
            cv2.imwrite(aug_img_path, aug_img)

            # Copy the label file
            with open(aug_label_path, "w") as f:
                f.write(label_data)


#Pretrained Model on the dataset

data = "/content/data/BoneFractureYolo8/data.yaml"

model = YOLO("yolo12m.pt")

model.train(
    data=data,
    epochs=200,
    batch=8,
    imgsz=640,  # Resize to 640x640
    augment=False  # Disable extra augmentations
)
