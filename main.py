import argparse
import glob
import os
import time

import cv2
import dlib
import numpy as np

from utils import (
    detect_face,
    load_assets,
    draw_landmarks,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="shape_predictor_68_face_landmarks.dat", help="dlib face landmarks detector model.")
    parser.add_argument("--image", type=str, default="example1", help="sunglasses image file name (without extension).")
    parser.add_argument("--landmarks", action="store_true", help="save landmarks with source images")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    
    detector_faces, detector_landmarks = detect_face(args.model)
    sunglasses, points = load_assets(args.image)

    sun_h, sun_w, _ = sunglasses.shape

    faces_path = os.path.join(os.getcwd(), "assets", "images", "faces", "*.jpg")

    for img_path in glob.glob(faces_path):
        _, filename = os.path.split(img_path)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_rectangles = detector_faces(img_rgb, 0)

        for i in range(len(face_rectangles)):
            rect = dlib.rectangle(
                            int(face_rectangles[i].left()),
                            int(face_rectangles[i].top()),
                            int(face_rectangles[i].right()),
                            int(face_rectangles[i].bottom()),
                        )
            landmarks = detector_landmarks(img_rgb, rect)

            if args.landmarks:
                img_copy = img.copy()
                draw_landmarks(img_copy, landmarks)

                landmarks_filename = f"landmarks_{filename}"
                landmarks_filepath = os.path.join(os.getcwd(), "results", landmarks_filename)
                cv2.imwrite(landmarks_filepath, img_copy)

            # nose top, left and right face end points
            x = int(landmarks.parts()[27].x)
            y = int(landmarks.parts()[27].y)
            x_18 = int(landmarks.parts()[17].x)
            x_27 = int(landmarks.parts()[26].x)

            # calculate new width and height, moving distance for adjusting sunglasses
            width = int(abs(x_18 - x_27))
            scale = width / sun_w
            height = int(sun_h * scale)

            move_x = -int(points[1] * scale)
            move_y = -int(points[2] * scale)

            # get region of interest on the face to change
            roi_color = img[(y + move_y):(y + height + move_y), (x + move_x):(x + width + move_x)]

            # find all non-transparent points
            sunglasses = cv2.resize(sunglasses, (width, height))

            index = np.argwhere(sunglasses[:,:,3] > 0)

            for j in range(3):
                roi_color[index[:,0], index[:,1], j] = sunglasses[index[:,0], index[:,1], j]

            # set the area of the image of the changed region with sunglasses
            img[(y + move_y):(y + height + move_y), (x + move_x):(x + width + move_x)] = roi_color

            result_filename = f"puton_{filename}"
            result_filepath = os.path.join(os.getcwd(), "results", result_filename)
            cv2.imwrite(result_filepath, img)


if __name__ == "__main__":
    main()
