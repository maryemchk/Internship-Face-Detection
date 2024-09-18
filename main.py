import pathlib
import cv2

# Find the Haar Cascade XML file
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

# BUILDING A CLASSIFIER
clf = cv2.CascadeClassifier(str(cascade_path))

# Initialize video capture from a file
video_path = "vecteezy_young-woman-walking-in-city-street_47658446.mp4"
video_capture = cv2.VideoCapture(video_path)

# Capture a single frame
ret, frame = video_capture.read()


# Check if the frame was read correctly
if not ret:
    print("Failed to grab frame")
else:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    

    # Check and print if faces are detected
    if len(faces) > 0:
        print("Face detected")
    else:
        print("No face detected")



