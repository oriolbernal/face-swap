import cv2
from face_swap import swap_faces
from image_utils import concatenate_images

imgPath = "assets/img3.jpg"

# Read images
img = cv2.imread(imgPath)

# Swap faces
img_changed_face = swap_faces(img)

# Concatenate images horizontally
result = concatenate_images((img, img_changed_face), axis='h', border=5)

# Show final result and process
cv2.imshow("result", result)
cv2.waitKey(0)

# Save Results
cv2.imwrite('results/result81_1img.png', result)
