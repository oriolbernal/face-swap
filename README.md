# face-swap
Python script that swaps the faces from one or two images.


## Getting Started
The project has been inspired by the following resources:

* This [tutorial](https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/) from by Sergio Canu.

* The repo found in [wuhuikai - FaceSwap](https://github.com/wuhuikai/FaceSwap) has also been useful.


## Demo

This is how the results look like:

![Result](https://github.com/oriolbernal/face-swap/blob/master/results/result81.png)

And what the script does:

* First steps:
![Process](https://github.com/oriolbernal/face-swap/blob/master/results/first_steps81.png)

* Calculate the new face:
![Process2](https://github.com/oriolbernal/face-swap/blob/master/results/second_step81.gif)

* Swap the new faces and apply a color filter:
![Process3](https://github.com/oriolbernal/face-swap/blob/master/results/last_steps81.png)

## Resources

* The 81 points landmark detector model found in [codeniko - shape_predictor_81_face_landmarks](https://github.com/codeniko/shape_predictor_81_face_landmarks) has been used.

## Examples

We can swap the faces from one image:

```python
import cv2
from face_swap import swap_faces

# Read images
img = cv2.imread("assets/img3.jpg")

# Swap faces
img_changed_faces = swap_faces(img)

# Show results
cv2.imshow("img_changed_faces", img_changed_faces)
cv2.waitKey(0)
```

Or we can also swap the faces from two images:
```python
import cv2
from face_swap import swap_faces

# Read images
img = cv2.imread("assets/img1.jpg")
img2 = cv2.imread("assets/img2.jpg")

# Swap faces
img_changed_face, img2_changed_face = swap_faces(img, img2)

# Show results
cv2.imshow("img_changed_face", img_changed_face)
cv2.imshow("img2_changed_face", img2_changed_face)
cv2.waitKey(0)
```

## Authors

* **Oriol Bernal** - [oriolbernal](https://github.com/oriolbernal)

