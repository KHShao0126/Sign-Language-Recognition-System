import os
import cv2

def draw_yolo_labels(image_path, label_path, out_path, class_names=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot open {image_path}")
        return

    h, w, _ = img.shape
    if not os.path.exists(label_path):
        print(f"⚠️ No label file for {image_path}")
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id, x_center, y_center, bw, bh = map(float, parts)

        # YOLO -> OpenCV
        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)

        # 畫框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = str(int(cls_id))
        if class_names and int(cls_id) < len(class_names):
            label_text = class_names[int(cls_id)]
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"✅ Saved {out_path}")


def verify_dataset(img_dir="dataset_yolo", out_dir="verify_output", class_names=None):
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, fname)
                label_path = os.path.splitext(image_path)[0] + ".txt"

                # 保留子資料夾結構
                rel_path = os.path.relpath(image_path, img_dir)
                out_path = os.path.join(out_dir, rel_path)

                draw_yolo_labels(image_path, label_path, out_path, class_names)


if __name__ == "__main__":
    verify_dataset("dataset_yolo", "verify_output", class_names=["hand"])