import math
import shutil
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

def _rotate_and_zoom(img, angle):
    """
    Rotate an image and zoom so that no black corners appear.
    
    :param img: PIL Image
    :param angle: rotation angle in degrees
    :return: rotated and zoomed Image
    """

    w, h = img.size
    angle_rad = math.radians(angle % 360)

    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    scale = min(w / (w * cos_a + h * sin_a), h / (w * sin_a + h * cos_a))

    rotated = img.rotate(angle, expand=True)

    W, H = rotated.size
    new_w = w * scale
    new_h = h * scale
    left = (W - new_w) / 2
    top = (H - new_h) / 2
    right = left + new_w
    bottom = top + new_h

    cropped = rotated.crop((left, top, right, bottom))
    
    return cropped

def _random_augment(img: Image.Image):
    """Apply random augmentations (rotation, flip, brightness, blur)."""
    # rotation with zoom (no black corners)
    angle = random.uniform(-25, 25)
    img = _rotate_and_zoom(img, angle)

    # horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # brightness/contrast jitter
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    # slight blur
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img

def oversample_with_augmentation(train_dir):
    """
    Oversample undersampled classes with augmentation until each class
    has roughly the same size as the largest one.

    :param train_dir: path to training directory with class subfolders
    """
    train_dir = Path(train_dir)
    classes = [d for d in train_dir.iterdir() if d.is_dir()]

    class_counts = {cls.name: len(list(cls.glob("*"))) for cls in classes}
    max_count = max(class_counts.values())

    print("Class distribution before oversampling:", class_counts)

    for cls in classes:
        cls_name = cls.name
        count = class_counts[cls_name]
        num_to_generate = max_count - count

        print(f"Oversampling {cls_name}...")

        images = list(cls.glob("*"))
        for i in tqdm(range(num_to_generate)):
            img_path = random.choice(images)
            try:
                img = Image.open(img_path).convert("RGB")
                aug_img = _random_augment(img)
                new_name = f"aug{i}_{img_path.stem}.jpg"
                aug_img.save(cls / new_name, "JPEG")
            except Exception as e:
                print(f"Error with {img_path}: {e}")

    final_counts = {cls.name: len(list(cls.glob("*"))) for cls in classes}
    print("Class distribution after oversampling:", final_counts)

def split_dataset(dataset_path: str, output_path: str, 
                  train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset into train, val, and test folders while keeping class structure.

    :param dataset_path: Path to dataset with class subfolders
    :param output_path: Path where split dataset will be saved
    :param train_ratio: Fraction of images for training
    :param val_ratio: Fraction of images for validation
    :param seed: Random seed for reproducibility
    """
    print("Splitting dataset to train, val, and test...")
    random.seed(seed)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]

    for cls in classes:
        images = list((dataset_path / cls).glob("*"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_files = images[:n_train]
        val_files = images[n_train:n_train+n_val]
        test_files = images[n_train+n_val:]

        for split, files in zip(["train", "val", "test"], 
                                [train_files, val_files, test_files]):
            split_dir = output_path / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, split_dir / f.name)

    print(f"Dataset split complete! Saved to {output_path}")
    
def resize_images(
    dataset_path: str,
    target_size=(224, 224),
    keep_aspect_ratio=True
):
    """
    Preprocess images by resizing (and optionally padding) to target_size

    :param dataset_path: Path to the raw dataset with class subfolders
    :param target_size: Tuple of (width, height) for resized images
    :param keep_aspect_ratio: If True, pad images to maintain original aspect ratio
    """
    dataset_path = Path(dataset_path)
    
    img_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    img_paths = [p for p in dataset_path.rglob("*") if p.suffix.lower() in img_extensions]

    for img_path in tqdm(img_paths, desc="Resizing images"):
        try:
            img = Image.open(img_path).convert("RGB")

            if keep_aspect_ratio:
                img = ImageOps.pad(img, target_size, color=(0, 0, 0))
            else:
                img = img.resize(target_size)

            img.save(img_path) 
        except Exception as e:
            print(f"⚠️ Could not process {img_path}: {e}")

    print(f"✅ All images resized to {target_size} in {dataset_path}")


raw_dataset = "task2_ner_image_classification/data/animals10/raw-img"    
processed_dataset = "task2_ner_image_classification/data/animals10/processed" 

split_dataset(raw_dataset, processed_dataset)

train_folder = "task2_ner_image_classification/data/animals10/processed/train"
oversample_with_augmentation(train_folder)

resize_images(processed_dataset, target_size=(224,224), keep_aspect_ratio=True)
