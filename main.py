import absl.logging
from cvzone.FaceDetectionModule import FaceDetector
import cv2

# Initialize logging
absl.logging.set_verbosity(absl.logging.INFO)
absl.logging.use_absl_handler()

# Initialize video capture and face detector
cap = cv2.VideoCapture('tes.mp4')
detector = FaceDetector()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame
    frame, bboxes = detector.findFaces(frame, draw=False)
    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            face_img = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_img, (71, 71), 0)
            frame[y:y+h, x:x+w] = blurred_face
    
    # Display the frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()