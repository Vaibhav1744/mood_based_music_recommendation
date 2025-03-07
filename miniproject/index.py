import cv2

def list_available_cameras(max_tests=5):
    """Detect available camera indices with improved error handling"""
    available = []
    for index in range(max_tests):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
            cap.release()
        else:
            cap.release()  # Ensure proper cleanup
    return available

if __name__ == "__main__":
    print("Available camera indices:", list_available_cameras())
    print("\nNote: If no cameras appear, check:")
    print("1. Camera permissions")
    print("2. Physical connections")
    print("3. Driver installations")