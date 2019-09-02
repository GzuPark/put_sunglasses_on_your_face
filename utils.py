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
