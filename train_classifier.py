import os
import shutil
import random

SOURCE_FOLDER = "cropped_dataset"
SPLIT_FOLDER = "dataset_split"
CLASSES = ["neutral", "mask", "sunglasses", "mask_sunglasses"]
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
random.seed(42)

for split in ["train", "validation", "test"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(SPLIT_FOLDER, split, class_name),
            exist_ok=True
        )

for class_name in CLASSES:
    source_folder = os.path.join(SOURCE_FOLDER, class_name)
    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    validation_count = int(total * VALIDATION_RATIO)
    train_images = images[:train_count]
    validation_images = images[
        train_count:train_count + validation_count]
    test_images = images[
        train_count + validation_count:]

    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(SPLIT_FOLDER, "train", class_name, image))

    for image in validation_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(SPLIT_FOLDER, "validation", class_name, image))

    for image in test_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(SPLIT_FOLDER, "test", class_name, image))

    print(
        class_name,
        "-> Train:", len(train_images),
        "Validation:", len(validation_images),
        "Test:", len(test_images))
print("\nDataset split completed.")