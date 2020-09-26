import cv2
import numpy as np
import dlib
from image_utils import draw_triangle, draw_square, draw_points, draw_triangles


def get_faces(img_gray):
    detector = dlib.get_frontal_face_detector()

    faces = detector(img_gray)
    if len(faces) == 0:
        print("No faces found")
        quit()

    return faces


def get_landmarks_points(img_gray, face):
    # predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
    predictor_path = 'models/shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    landmarks = predictor(img_gray, face)

    landmarks_points = []
    for n in range(0, landmarks.num_parts):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    return landmarks_points


def get_faceImg_and_mask(img, img_gray, convexhull):
    mask = np.zeros_like(img_gray)

    # The pixels of the face are white, rest of the image is black
    cv2.fillConvexPoly(mask, convexhull, 255)

    # Only colored face (applied mask)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    return face_image_1, mask


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def get_delaunay_triangulation(landmarks_points, convexhull):
    points = np.array(landmarks_points, np.int32)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles


def get_new_face(img, img2, landmarks_points, landmarks_points2, indexes_triangles):
    img_new_face = np.zeros(img2.shape, np.uint8)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img_new_face_rect_area = img_new_face[y: y + h, x: x + w]
        img_new_face_rect_area_gray = cv2.cvtColor(img_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img_new_face_rect_area = cv2.add(img_new_face_rect_area, warped_triangle)
        img_new_face[y: y + h, x: x + w] = img_new_face_rect_area

    return img_new_face


def change_face(img, convexhull, new_face):
    # Face swapped (putting new_face into convexhull)
    img_face_mask = np.zeros_like(img[:, :, 0])
    img_head_mask = cv2.fillConvexPoly(img_face_mask, convexhull, 255)
    img_face_mask = cv2.bitwise_not(img_head_mask)

    img2_head_noface = cv2.bitwise_and(img, img, mask=img_face_mask)
    result = cv2.add(img2_head_noface, new_face)

    # Applying color filter
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img, img_head_mask, center_face2, cv2.NORMAL_CLONE)

    return seamlessclone


def swap_faces(img, img2=None):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img2 is not None:
        img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        face = get_faces(img_gray)[0]
        face2 = get_faces(img_gray2)[0]

        landmarks_points = get_landmarks_points(img_gray, face)
        landmarks_points2 = get_landmarks_points(img_gray2, face2)

        convexhull = cv2.convexHull(np.array(landmarks_points, np.int32))
        convexhull2 = cv2.convexHull(np.array(landmarks_points2, np.int32))

        indexes_triangles = get_delaunay_triangulation(landmarks_points, convexhull)
        indexes_triangles2 = get_delaunay_triangulation(landmarks_points2, convexhull2)

        img2_new_face = get_new_face(img, img2, landmarks_points, landmarks_points2, indexes_triangles)
        img_new_face = get_new_face(img2, img, landmarks_points2, landmarks_points, indexes_triangles2)

        img2_changed_face = change_face(img2, convexhull2, img2_new_face)
        img_changed_face = change_face(img, convexhull, img_new_face)

        return img_changed_face, img2_changed_face

    else:
        faces = get_faces(img_gray)
        face = faces[0]
        face2 = faces[1]
        landmarks_points = get_landmarks_points(img_gray, face)
        landmarks_points2 = get_landmarks_points(img_gray, face2)

        convexhull = cv2.convexHull(np.array(landmarks_points, np.int32))
        convexhull2 = cv2.convexHull(np.array(landmarks_points2, np.int32))

        indexes_triangles = get_delaunay_triangulation(landmarks_points, convexhull)
        indexes_triangles2 = get_delaunay_triangulation(landmarks_points2, convexhull2)

        img_new_face = get_new_face(img, img, landmarks_points2, landmarks_points, indexes_triangles2)
        img2_new_face = get_new_face(img, img, landmarks_points, landmarks_points2, indexes_triangles)

        img_changed_face1 = change_face(img, convexhull2, img2_new_face)
        img_changed_faces = change_face(img_changed_face1, convexhull, img_new_face)

        return img_changed_faces


def get_process(img, img2):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    face = get_faces(img_gray)[0]
    face2 = get_faces(img_gray2)[0]

    landmarks_points = get_landmarks_points(img_gray, face)
    landmarks_points2 = get_landmarks_points(img_gray2, face2)

    convexhull = cv2.convexHull(np.array(landmarks_points, np.int32))
    convexhull2 = cv2.convexHull(np.array(landmarks_points2, np.int32))

    indexes_triangles = get_delaunay_triangulation(landmarks_points, convexhull)
    indexes_triangles2 = get_delaunay_triangulation(landmarks_points2, convexhull2)

    img_new_face = get_new_face(img2, img, landmarks_points2, landmarks_points, indexes_triangles2)
    img2_new_face = get_new_face(img, img2, landmarks_points, landmarks_points2, indexes_triangles)

    img_changed_face_filtered = change_face(img, convexhull, img_new_face)
    img2_changed_face_filtered = change_face(img2, convexhull2, img2_new_face)

    color = (0, 255, 0)
    thickness = 2
    # RESULTS 1
    img_face = draw_square(img.copy(), [(face.left(), face.top()), (face.right(), face.bottom())], color=color,
                           thickness=thickness)
    img_landmarks = draw_points(img.copy(), landmarks_points, color=color, thickness=thickness)
    img_face_contour = cv2.polylines(img.copy(), [convexhull], True, color, thickness)
    face_image, face_mask = get_faceImg_and_mask(img, img_gray, convexhull)

    # White mask for showing results
    white_mask = cv2.fillConvexPoly(np.ones_like(img) * 255, convexhull, 0)
    face_image = white_mask + face_image

    triangles_points = get_triangles_points(landmarks_points, indexes_triangles)
    face_triangles = draw_triangles(face_image.copy(), triangles_points, color=color, thickness=thickness)

    # Face swap without applying filter
    img_face_mask = np.zeros_like(img[:, :, 0])
    img_head_mask = cv2.fillConvexPoly(img_face_mask, convexhull, 255)
    img_face_mask = cv2.bitwise_not(img_head_mask)
    img_head_noface = cv2.bitwise_and(img, img, mask=img_face_mask)
    img_changed_face = cv2.add(img_head_noface, img_new_face)

    # White mask for showing results
    img_face_mask_white = np.ones_like(img[:, :, :])*255
    img_head_mask_white = cv2.fillConvexPoly(img_face_mask_white, convexhull, 0)

    # RESULTS 2
    img_face2 = draw_square(img2.copy(), [(face2.left(), face2.top()), (face2.right(), face2.bottom())], color=color,
                            thickness=thickness)
    img_landmarks2 = draw_points(img2.copy(), landmarks_points2, color=color, thickness=thickness)
    img_face_contour2 = cv2.polylines(img2.copy(), [convexhull2], True, color, thickness)
    face_image2, face_mask2 = get_faceImg_and_mask(img2, img_gray2, convexhull2)

    # White mask for showing results
    white_mask2 = cv2.fillConvexPoly(np.ones_like(img2) * 255, convexhull2, 0)
    face_image2 = white_mask2 + face_image2

    triangles_points2 = get_triangles_points(landmarks_points2, indexes_triangles2)
    face_triangles2 = draw_triangles(face_image2.copy(), triangles_points2, color=color, thickness=thickness)

    # Face swap without applying filter
    img2_face_mask = np.zeros_like(img2[:, :, 0])
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    img2_changed_face = cv2.add(img2_head_noface, img2_new_face)

    # White mask for showing results
    img2_face_mask_white = np.ones_like(img2[:, :, :]) * 255
    img2_head_mask_white = cv2.fillConvexPoly(img2_face_mask_white, convexhull2, 0)

    result = [
        (img, img_face, img_landmarks, img_face_contour, face_image, face_triangles),
        (img2, img_face2, img_landmarks2, img_face_contour2, face_image2, face_triangles2),
        (img_new_face + img_head_mask_white, img_changed_face, img_changed_face_filtered),
        (img2_new_face + img2_head_mask_white, img2_changed_face, img2_changed_face_filtered)
    ]

    return result


def get_triangles_points(points, indexes):
    result_points = []
    for index in indexes:
        tr2_pt1 = points[index[0]]
        tr2_pt2 = points[index[1]]
        tr2_pt3 = points[index[2]]

        result_points.append([tr2_pt1, tr2_pt2, tr2_pt3])

    return result_points


def get_triangles_img(img, img2):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    face = get_faces(img_gray)[0]
    face2 = get_faces(img_gray2)[0]

    landmarks_points = get_landmarks_points(img_gray, face)
    landmarks_points2 = get_landmarks_points(img_gray2, face2)

    convexhull = cv2.convexHull(np.array(landmarks_points, np.int32))
    convexhull2 = cv2.convexHull(np.array(landmarks_points2, np.int32))
    indexes_triangles = get_delaunay_triangulation(landmarks_points, convexhull)

    img_new_face = np.zeros(img2.shape, np.uint8)
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img_new_face_rect_area = img_new_face[y: y + h, x: x + w]
        img_new_face_rect_area_gray = cv2.cvtColor(img_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img_new_face_rect_area = cv2.add(img_new_face_rect_area, warped_triangle)
        img_new_face[y: y + h, x: x + w] = img_new_face_rect_area

        i1 = draw_triangle(img.copy(), tr1_pt1, tr1_pt2, tr1_pt3)
        i2 = draw_triangle(img2.copy(), tr2_pt1, tr2_pt2, tr2_pt3)

        # White mask for showing results
        img2_face_mask_white = np.ones_like(img2[:, :, :]) * 255
        img2_head_mask_white = cv2.fillConvexPoly(img2_face_mask_white, convexhull2, 0)

        yield (i1, i2, img_new_face + img2_head_mask_white)
