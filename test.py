from picamera2 import Picamera2
import cv2
import os
from datetime import datetime

# Initialize Pi Camera
picam2 = Picamera2()

# Configure camera (you can tweak resolution)
config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)
picam2.configure(config)

# Start camera
picam2.start()

print("📷 Camera started. Press ENTER to capture image. Press 'q' to quit.")

# Create save directory
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Show live preview
    cv2.imshow("Live Preview - Pi Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press ENTER to capture (Enter = 13)
    if key == 13:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File paths
        color_path = os.path.join(save_dir, f"image_{timestamp}_color.jpg")
        gray_path = os.path.join(save_dir, f"image_{timestamp}_gray.jpg")

        # Save original (RGB/BGR as OpenCV format)
        cv2.imwrite(color_path, frame)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(gray_path, gray)

        print("\n✅ Image Captured!")
        print(f"📁 Color Image Saved At: {os.path.abspath(color_path)}")
        print(f"📁 Grayscale Image Saved At: {os.path.abspath(gray_path)}\n")

    # Press 'q' to quit
    elif key == ord('q'):
        print("👋 Exiting...")
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
