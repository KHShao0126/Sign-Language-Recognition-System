import cv2
import os
import mediapipe as mp

mp_hands = mp.solutions.hands

def process_folder(input_dir="dataset1", output_dir="dataset_yolo", class_id=0):
    os.makedirs(output_dir, exist_ok=True)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.lower().endswith((".jpg", ".png")):
                    continue

                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                h, w, _ = img.shape
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 取所有關鍵點座標
                        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # 加 padding，避免框太小
                        pad = 20
                        x_min = max(x_min - pad, 0)
                        y_min = max(y_min - pad, 0)
                        x_max = min(x_max + pad, w)
                        y_max = min(y_max + pad, h)

                        # 轉成 YOLO 格式 (normalize)
                        x_center = (x_min + x_max) / 2.0 / w
                        y_center = (y_min + y_max) / 2.0 / h
                        bbox_w = (x_max - x_min) / w
                        bbox_h = (y_max - y_min) / h

                        # 建立輸出資料夾 (同 label 結構)
                        rel_dir = os.path.relpath(root, input_dir)
                        save_dir = os.path.join(output_dir, rel_dir)
                        os.makedirs(save_dir, exist_ok=True)

                        # 複製圖片
                        save_img_path = os.path.join(save_dir, file)
                        cv2.imwrite(save_img_path, img)

                        # 輸出 txt (YOLO 格式)
                        txt_path = save_img_path.rsplit(".", 1)[0] + ".txt"
                        with open(txt_path, "w") as f:
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

                        print(f"✅ {file} → YOLO label saved at {txt_path}")
                else:
                    print(f"❌ No hand detected in {img_path}")

if __name__ == "__main__":
    # class_id 預設 0 (代表 hand)
    process_folder("dataset1", "dataset_yolo", class_id=0)