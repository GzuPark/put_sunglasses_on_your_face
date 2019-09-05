import math
import os

import cv2
import dlib
import numpy as np


def detect_face(model):
    model_path = os.path.join(os.getcwd(), "assets", "models", model)

    faces = dlib.get_frontal_face_detector()
    landmarks = dlib.shape_predictor(model_path)

    return faces, landmarks


def load_assets(filename):
    image_filename = filename + ".png"
    points_filename = filename + ".txt"
    image_path = os.path.join(os.getcwd(), "assets", "images", "sunglasses", image_filename)
    points_path = os.path.join(os.getcwd(), "assets", "images", "sunglasses", points_filename)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    points = np.loadtxt(points_path, dtype="uint16")

    return image, points


def draw_landmarks(img, landmarks):
    for i, points in enumerate(landmarks.parts()):
        px = int(points.x)
        py = int(points.y)
        cv2.circle(img, (px, py), 1, (255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, str(i+1), (px, py), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0), 1)


def degree_to_radius(degree):
    return degree * math.pi / 180.0


def get_radius(theta, phi, gamma):
    rad_theta = degree_to_radius(theta)
    rad_phi = degree_to_radius(phi)
    rad_gamma = degree_to_radius(gamma)

    return rad_theta, rad_phi, rad_gamma


def get_perspective_projection_matrix(width, height, focal, theta, phi, gamma, dx, dy, dz):
    # project 2D -> 3D matrix
    A1 = np.array(
        [
            [1, 0, -width/2],
            [0, 1, -height/2],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    # rotation matrices arount the x, y, z axis
    RX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    RY = np.array(
        [
            [np.cos(phi), 0, -np.sin(phi), 0],
            [0, 1, 0, 0],
            [np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    )
    RZ = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0, 0],
            [np.sin(gamma), np.cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # composed rotation matrix with RX, RY, RZ
    R = np.dot(np.dot(RX, RY), RZ)

    # translation matrix
    T = np.array(
        [
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1],
        ]
    )

    # project 3D -> 2D matrix
    A2 = np.array(
        [
            [focal, 0, width/2, 0],
            [0, focal, height/2, 0],
            [0, 0, 1, 0],
        ]
    )

    # transformation matrix
    result = np.dot(A2, np.dot(T, np.dot(R, A1)))

    return result


def get_orientation(width, height, landmarks):
    image_points = np.array(
        [
            (landmarks[30].x, landmarks[30].y),     # nose tip
            (landmarks[8].x, landmarks[8].y),       # chin
            (landmarks[36].x, landmarks[36].y),     # left eye left corner
            (landmarks[45].x, landmarks[45].y),     # right eye right corner
            (landmarks[48].x, landmarks[48].y),     # left mouth corner
            (landmarks[54].x, landmarks[54].y)      # right mouth corner
        ],
        dtype="double",
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),             # nose tip
            (0.0, -330.0, -65.0),        # chin
            (-165.0, 170.0, -135.0),     # left eye left corner
            (165.0, 170.0, -135.0),      # right eye right corner
            (-150.0, -150.0, -125.0),    # left mouth corner
            (150.0, -150.0, -125.0)      # right mouth corner
        ]
    )

    center = (width/2, height/2)
    focal_length = center[0] / np.tan(60 / (2 * np.pi / 180))
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))
    _, r_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    r_vector_matrix = cv2.Rodrigues(r_vec)[0]

    project_matrix = np.hstack((r_vector_matrix, trans_vec))
    euler_angles = cv2.decomposeProjectionMatrix(project_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = math.degrees(math.asin(math.sin(roll)))
    yaw = -math.degrees(math.asin(math.sin(yaw)))

    return int(pitch), int(roll), int(yaw)


def rotate_along_axis(img, width, height, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
    sunglasses = cv2.resize(img, (width, height))
    rad_theta, rad_phi, rad_gamma = get_radius(theta, phi, gamma)

    # get focal length on z axis
    dist = np.sqrt(width**2 + height**2)
    focal = dist / (2 * np.sin(rad_gamma) if np.sin(rad_gamma) != 0 else 1)
    dz = focal

    # get projection matrix
    mat = get_perspective_projection_matrix(width, height, focal, rad_theta, rad_phi, rad_gamma, dx, dy, dz)
    result = cv2.warpPerspective(sunglasses, mat, (width, height))

    return result
