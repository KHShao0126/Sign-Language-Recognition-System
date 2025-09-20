import cv2
import os
import numpy as np
import random
from glob import glob

def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def blur(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def flip(img):
    return cv2.flip(img, 1)  # 水平翻轉

def adjust_brightness_contrast(img):
    alpha = random.uniform(0.7, 1.3)  # 對比度
    beta = random.randint(-30, 30)    # 亮度
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def rotate(img):
    h, w = img.shape[:2]
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def augment_image(img):
    funcs = [add_noise, blur, flip, adjust_brightness_contrast, rotate]
    func = random.choice(funcs)
    return func(img)

def augment_dataset(input_dir="dataset1", target_size=100):
    labels = os.listdir(input_dir)

    for label in labels:
        path = os.path.join(input_dir, label)
        if not os.path.isdir(path):
            continue

        images = glob(os.path.join(path, "*.jpg"))
        orig_count = len(images)
        count = 0

        # 如果原始就超過 target_size 就跳過
        if orig_count >= target_size:
            print(f"⚡ {label} already has {orig_count}, skipped")
            continue

        while orig_count + count < target_size:
            img_path = random.choice(images)  # ✅ 隨機挑不同圖片
            img = cv2.imread(img_path)
            if img is None: continue

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            aug_img = augment_image(img)
            save_name = f"{base_name}_aug{count}.jpg"
            save_path = os.path.join(path, save_name)
            cv2.imwrite(save_path, aug_img)
            count += 1

        print(f"✅ Finished {label}: {orig_count + count} images (augmented {count})")

if __name__ == "__main__":
    augment_dataset("dataset1", target_size=200)