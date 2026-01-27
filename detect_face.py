import cv2

# Load Haar Cascade file
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Read image
image = cv2.imread("test.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

# Draw rectangle around faces
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Show output
cv2.imshow("AI Human Face Detection System", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
