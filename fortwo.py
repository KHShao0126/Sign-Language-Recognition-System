import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# 需要雙手合併成一個框的類別
double_hand_classes = {"name", "love", "house", "book", "stop"}

def hands_to_yolo_bbox(multi_hand_landmarks, img_w, img_h):
    """把兩隻手的 landmarks 合併成一個 YOLO 格式 bbox"""
    all_x, all_y = [], []
    for hand_landmarks in multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_x.append(lm.x)
            all_y.append(lm.y)

    if not all_x or not all_y:
        return None

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # YOLO 格式 (normalized)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

def relabel_double_hands_as_one(dataset_dir="dataset_yolo"):
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
        for cls in double_hand_classes:
            class_dir = os.path.join(dataset_dir, cls)
            if not os.path.exists(class_dir):
                print(f"⚠️ Skip {cls}, no folder found")
                continue

            for fname in os.listdir(class_dir):
                if not fname.endswith(".jpg"):
                    continue

                img_path = os.path.join(class_dir, fname)
                label_path = img_path.replace(".jpg", ".txt")

                img = cv2.imread(img_path)
                h, w, _ = img.shape
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    continue

                # 只要至少有兩隻手 → 合併成一個框
                if len(results.multi_hand_landmarks) >= 2:
                    line = hands_to_yolo_bbox(results.multi_hand_landmarks, w, h)
                    if line:
                        with open(label_path, "w") as f:
                            f.write(line)
                        print(f"✅ Relabeled {fname} (two hands → one bbox) in {cls}")
                else:
                    # 如果只偵測到一隻手 → 保持原本方式
                    line = hands_to_yolo_bbox(results.multi_hand_landmarks, w, h)
                    if line:
                        with open(label_path, "w") as f:
                            f.write(line)
                        print(f"ℹ️ Relabeled {fname} (only one hand detected) in {cls}")

if __name__ == "__main__":
    relabel_double_hands_as_one("dataset_yolo")