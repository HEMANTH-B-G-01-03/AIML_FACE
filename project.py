import cv2 as cv
import numpy as np

def visualize(image, faces, result_text=None, similarity_score=None, thickness=2):
    """
    Visualizes detected faces on the image.

    Parameters:
    - image: The image on which to draw faces.
    - faces: List of detected faces.
    - result_text: Optional text to display.
    - similarity_score: Optional similarity score to display.
    - thickness: Thickness of the drawn shapes.
    """
    for face in faces:
        coords = face[:-1].astype(np.int32)
        cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
        cv.circle(image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
        cv.circle(image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
        cv.circle(image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
        cv.circle(image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
        cv.circle(image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

        if similarity_score is not None:
            cv.putText(image, f"Similarity Score: {similarity_score:.2f}", (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    if result_text:
        cv.putText(image, result_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

# Path to the reference image
reference_image_path = "D:\\Face\\query(11).png"

# Load the reference image

ref_image = cv.imread(reference_image_path)

if ref_image is None:
    print(f"Error: Unable to read the image from the path: {reference_image_path}")
    exit()

# Display the reference image for debugging
cv.imshow("Debug Reference Image", ref_image)
cv.waitKey(0)

# Print image properties
print(f"Image Shape: {ref_image.shape}")
print(f"Image Type: {type(ref_image)}")

# Resize the image to ensure it's within reasonable dimensions for detection
resized_image = cv.resize(ref_image, (640, 480))
cv.imshow("Resized Reference Image", resized_image)
cv.waitKey(0)

# Initialize face detector with adjusted parameters
score_threshold = 0.5
nms_threshold = 0.4
top_k = 5000
faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (resized_image.shape[1], resized_image.shape[0]), score_threshold, nms_threshold, top_k)

# Detect faces in the reference image
faceInAdhaar = faceDetector.detect(resized_image)
print("Detection Output for Reference Image:", faceInAdhaar)

if faceInAdhaar[1] is not None:
    print("Faces detected in reference image.")
    visualize(resized_image, faceInAdhaar[1])
    cv.imshow("Detected Faces in Reference Image", resized_image)
    cv.waitKey(0)
else:
    print("No face detected in reference image.")

# Initialize face recognizer
recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

# Open a connection to the webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

l2_similarity_threshold = 1.128

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    # Resize frame to match the dimensions used for face detection
    resized_frame = cv.resize(frame, (640, 480))
    faceDetector.setInputSize((resized_frame.shape[1], resized_frame.shape[0]))
    faceInQuery = faceDetector.detect(resized_frame)

    result_text = "No face detected"

    if faceInQuery[1] is not None:
        if faceInAdhaar[1] is not None:
            face1_align = recognizer.alignCrop(resized_image, faceInAdhaar[1][0])
            face2_align = recognizer.alignCrop(resized_frame, faceInQuery[1][0])

            face1_feature = recognizer.feature(face1_align)
            face2_feature = recognizer.feature(face2_align)

            l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)

            if l2_score <= l2_similarity_threshold:
                result_text = "Same Identity"
            else:
                result_text = "Different Identity"

            visualize(resized_frame, faceInQuery[1], result_text, l2_score)
        else:
            visualize(resized_frame, faceInQuery[1], result_text)
    else:
        visualize(resized_frame, [], result_text)

    cv.putText(resized_frame, "Press 'q' to quit", (10, resized_frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("Live Video", resized_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
