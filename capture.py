import cv2
import os

def capture_images(label, save_dir="dataset1", cam_index=0):
    # 建立存放目錄
    save_path = os.path.join(save_dir, label)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print(f"📷 Capturing images for label: {label}")
    print("👉 Press SPACE to capture, 'q' to quit")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot receive frame")
            break

        cv2.imshow("Capture - " + label, frame)
        key = cv2.waitKey(1)

        # 空白鍵存檔
        if key == 32:  # SPACE
            filename = os.path.join(save_path, f"{label}_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✅ Saved {filename}")
            count += 1

        # q 鍵退出
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📦 Finished capturing {count} images for label: {label}")

if __name__ == "__main__":
    label = input("Enter the word for this capture session: ")
    capture_images(label)