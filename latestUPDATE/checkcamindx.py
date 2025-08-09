import cv2
import sys

def find_camera_indices(max_index=10):
    """
    Find available camera indices by testing each one.
    
    Args:
        max_index (int): Maximum camera index to test (default: 10)
    
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    
    print(f"Checking camera indices from 0 to {max_index}...")
    print("-" * 50)
    
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            # Try to read a frame to confirm camera is working
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✓ Camera {index}: Available ({width}x{height} @ {fps:.1f}fps)")
                available_cameras.append(index)
            else:
                print(f"✗ Camera {index}: Opened but cannot read frames")
        else:
            print(f"✗ Camera {index}: Not available")
        
        cap.release()
    
    return available_cameras

def test_camera(index):
    """
    Test a specific camera index and show live preview.
    
    Args:
        index (int): Camera index to test
    """
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {index}")
        return
    
    print(f"Testing camera {index}. Press 'q' to quit, 's' to save a test image.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Add frame counter to the image
        cv2.putText(frame, f"Camera {index} - Frame {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera {index} Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"camera_{index}_test_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved test image: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to run the camera index checker.
    """
    print("OpenCV Camera Index Checker")
    print("=" * 30)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "test" and len(sys.argv) > 2:
                # Test specific camera
                camera_index = int(sys.argv[2])
                test_camera(camera_index)
                return
            else:
                max_index = int(sys.argv[1])
        except ValueError:
            print("Invalid argument. Usage: python checkcamindx.py [max_index] or python checkcamindx.py test [camera_index]")
            return
    else:
        max_index = 10
    
    # Find available cameras
    available_cameras = find_camera_indices(max_index)
    
    print("-" * 50)
    if available_cameras:
        print(f"Found {len(available_cameras)} available camera(s): {available_cameras}")
        
        # Ask user if they want to test any camera
        if available_cameras:
            print("\nWould you like to test any camera? (y/n): ", end="")
            choice = input().lower()
            
            if choice == 'y':
                print(f"Available cameras: {available_cameras}")
                print("Enter camera index to test: ", end="")
                try:
                    test_index = int(input())
                    if test_index in available_cameras:
                        test_camera(test_index)
                    else:
                        print(f"Camera {test_index} is not available.")
                except ValueError:
                    print("Invalid input.")
    else:
        print("No cameras found.")
        print("\nTroubleshooting tips:")
        print("1. Make sure your camera is connected")
        print("2. Check if other applications are using the camera")
        print("3. Try running as administrator")
        print("4. Update your camera drivers")

if __name__ == "__main__":
    main()