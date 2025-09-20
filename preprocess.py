import os, glob, random

def create_split(dataset_dir="dataset_yolo", out_dir="yolo_splits", train_ratio=0.8):
    os.makedirs(out_dir, exist_ok=True)
    all_images = glob.glob(os.path.join(dataset_dir, "*/*.jpg"))
    random.shuffle(all_images)

    split = int(len(all_images) * train_ratio)
    train_imgs = all_images[:split]
    val_imgs = all_images[split:]

    with open(os.path.join(out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_imgs))
    with open(os.path.join(out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_imgs))

    # 產生類別名稱檔
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    with open(os.path.join(out_dir, "obj.names"), "w") as f:
        f.write("\n".join(classes))

    with open(os.path.join(out_dir, "obj.data"), "w") as f:
        f.write(f"classes = {len(classes)}\n")
        f.write(f"train = {os.path.join(out_dir, 'train.txt')}\n")
        f.write(f"valid = {os.path.join(out_dir, 'val.txt')}\n")
        f.write(f"names = {os.path.join(out_dir, 'obj.names')}\n")
        f.write(f"backup = backup/\n")

    print(f"✅ Generated train/val split with {len(train_imgs)} train and {len(val_imgs)} val images")

if __name__ == "__main__":
    create_split("dataset_yolo")