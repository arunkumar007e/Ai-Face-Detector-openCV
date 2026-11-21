import cv2

# Load the Haar Cascade face detection model
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check whether the model loaded successfully
if face_model.empty():
    print("Error: Face detection model not found!")
    exit()

# Open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Webcam cannot be opened!")
    exit()

print("Webcam started. Press 'q' to exit.")

while True:
    # Read frame from webcam
    ret, frame = webcam.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert color image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_model.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display webcam output
    cv2.imshow("Face Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()