import cv2
import imageio
import numpy as np


def resize_image_as(img, img2):
    """ Resize the smallest image (img or img2) as the other image (larger) without deforming it """

    height1, weight1 = img.shape[:2]
    height2, weight2 = img2.shape[:2]

    # if weight of img 1 is the smallest
    if weight1 <= min(weight2, height1, height2):
        increase_ratio = weight2 / float(weight1)
        dim = (weight2, int(height1 * increase_ratio))
        img = cv2.resize(img, dim)
    elif weight2 < min(weight1, height1, height2):
        increase_ratio = weight1 / float(weight2)
        dim = (weight1, int(height2 * increase_ratio))
        img2 = cv2.resize(img2, dim)
    elif height1 <= min(weight1, weight2, height2):
        increase_ratio = height2 / float(height1)
        dim = (int(weight1 * increase_ratio), height2)
        img = cv2.resize(img, dim)
    else:
        increase_ratio = height1 / float(height2)
        dim = (int(weight2 * increase_ratio), height1)
        img2 = cv2.resize(img2, dim)

    return img, img2


def put_borders(img, img2, color=[255, 255, 255]):
    """ Adds extra pixels to images so they have the same sizes """

    height1, weight1 = img.shape[:2]
    height2, weight2 = img2.shape[:2]

    if weight1 > weight2:
        border_left = int((weight1 - weight2) / 2)
        if (weight1 - weight2) % 2 == 0:
            border_right = border_left
        else:
            border_right = border_left + 1
        img2 = cv2.copyMakeBorder(img2, 0, 0, border_left, border_right, cv2.BORDER_CONSTANT, value=color)
    else:
        border_left = int((weight2 - weight1) / 2)
        if (weight2 - weight1) % 2 == 0:
            border_right = border_left
        else:
            border_right = border_left + 1
        img = cv2.copyMakeBorder(img, 0, 0, border_left, border_right, cv2.BORDER_CONSTANT, value=color)

    if height1 > height2:
        border_top = int((height1 - height2) / 2)
        if (height1 - height2) % 2 == 0:
            border_bottom = border_top
        else:
            border_bottom = border_top + 1
        img2 = cv2.copyMakeBorder(img2, border_top, border_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
    else:
        border_top = int((height2 - height1) / 2)
        if ((height2 - height1) / 2) % 2 == 0:
            border_bottom = border_top
        else:
            border_bottom = border_top + 1
        img = cv2.copyMakeBorder(img, border_top, border_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)

    return img, img2


def resize_and_border_image(img, img2):
    """ Resize the images to same size """

    img, img2 = resize_image_as(img, img2)
    img, img2 = put_borders(img, img2)

    return img, img2


def create_gif(result_path, generator):
    """ Creates a gif file and store it in result_path
    The generator must return images """

    print('creating gif...')

    with imageio.get_writer(result_path, mode='I') as writer:
        for result in generator:
            writer.append_data(result)


def draw_triangle(img, tr_pt1, tr_pt2, tr_pt3, color=(0, 255, 0), thickness=3):
    cv2.line(img, tr_pt1, tr_pt2, color, thickness)
    cv2.line(img, tr_pt2, tr_pt3, color, thickness)
    cv2.line(img, tr_pt1, tr_pt3, color, thickness)

    return img


def draw_triangles(img, triangles, color=(0, 255, 0), thickness=3):
    for triangle in triangles:
        img = draw_triangle(img, triangle[0], triangle[1], triangle[2], color=color, thickness=thickness)

    return img


def draw_square(img, points, color=(0, 255, 0), thickness=3):

    cv2.rectangle(img, points[0], points[1], color, thickness)
    return img


def draw_points(img, points, radius=1, color=(0, 255, 0), thickness=3):
    for point in points:
        img = cv2.circle(img, (point[0], point[1]), radius, color, thickness)

    return img


def put_border_to_img(img, color=(255, 255, 255), border=1):
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=color)


def concatenate_images(images, axis='h', border=None):
    if border is not None:
        images_with_border = []
        for image in images:
            images_with_border.append(put_border_to_img(image, border=border))

        images = tuple(images_with_border)

    if axis == 'h':
        result = np.hstack(images)
    else:
        result = np.vstack(images)

    return result
