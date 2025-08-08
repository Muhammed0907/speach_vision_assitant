import cv2
from deepface import DeepFace
import time

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting emotion detection... Press 'q' to quit")
    
    last_analysis_time = 0
    analysis_interval = 0.5
    current_emotion = "Unknown"
    emotion_confidence = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        current_time = time.time()
        
        if current_time - last_analysis_time > analysis_interval:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                
                emotions = result['emotion']
                current_emotion = max(emotions, key=emotions.get)
                emotion_confidence = emotions[current_emotion]
                last_analysis_time = current_time
                
            except Exception as e:
                print(f"Analysis error: {e}")
        
        cv2.putText(frame, f"Emotion: {current_emotion} ({emotion_confidence:.1f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
