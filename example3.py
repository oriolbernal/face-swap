import cv2
from face_swap import get_triangles_img, get_process
from image_utils import concatenate_images, resize_and_border_image, create_gif


def append_triangles_img(img, img2):
    for (i1, i2, img_new_face) in get_triangles_img(img, img2):

        result = concatenate_images((i1, i2, img_new_face), axis='h', border=5)
        result = cv2.resize(result, (int(result.shape[1] / 4), int(result.shape[0] / 4)))
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        yield result


imgPath = "assets/img1.jpg"
imgPath2 = "assets/img2.jpg"

# Read images
img = cv2.imread(imgPath)
img2 = cv2.imread(imgPath2)

# Resize and border image, so both images have same dimensions
img, img2 = resize_and_border_image(img, img2)

steps = get_process(img, img2)

# Concatenate images
first_steps = concatenate_images(
    (
        concatenate_images(steps[0], axis='h', border=5),
        concatenate_images(steps[1], axis='h', border=5)
    ),
    axis='v',
    border=0
)

first_steps = cv2.resize(first_steps, (int(first_steps.shape[1]/3), int(first_steps.shape[0]/3)))
cv2.imwrite('results/first_steps81.png', first_steps)

second_step = create_gif('results/second_step81.gif', append_triangles_img(img, img2))

# Concatenate images
last_steps = concatenate_images(
    (
        concatenate_images(steps[2], axis='h', border=5),
        concatenate_images(steps[3], axis='h', border=5)
    ),
    axis='v',
    border=0
)
last_steps = cv2.resize(last_steps, (int(last_steps.shape[1]/3), int(last_steps.shape[0]/3)))
cv2.imwrite('results/last_steps81.png', last_steps)


cv2.imshow("first_steps", first_steps)
cv2.imshow("last_steps", last_steps)
cv2.waitKey(0)
