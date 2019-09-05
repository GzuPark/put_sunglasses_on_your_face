import argparse
import glob
import os

import cv2
import dlib
import numpy as np

from utils import (
    detect_face,
    draw_landmarks,
    get_orientation,
    load_assets,
    rotate_along_axis,
)


class PutOn(object):
    def __init__(self, args):
        self.args = args
        self.detector_faces, self.detector_landmarks = detect_face(args.model)
        self.sunglasses, self.points = load_assets(args.image)

        self.sun_h, self.sun_w, _ = self.sunglasses.shape

    def save_landmarks(self, landmarks):
        img_copy = self.img.copy()
        draw_landmarks(img_copy, landmarks)

        landmarks_filename = f"landmarks_{self.filename}"
        landmarks_filepath = os.path.join(os.getcwd(), "results", landmarks_filename)
        cv2.imwrite(landmarks_filepath, img_copy)

    def save_result(self):
        result_filename = f"puton_{self.filename}"
        result_filepath = os.path.join(os.getcwd(), "results", result_filename)
        cv2.imwrite(result_filepath, self.img)

    def images(self):
        faces_path = os.path.join(os.getcwd(), "assets", "images", "faces", "*.jpg")

        for img_path in glob.glob(faces_path):
            _, self.filename = os.path.split(img_path)
            self.img = cv2.imread(img_path)
            self.run()

    def webcam(self):
        cap = cv2.VideoCapture(0)

        if (cap.isOpened() is False):
            print("Error opening video stream or file.")
        else:
            print("Press 'Q' if you want to quit.")

        while(cap.isOpened()):
            ret, self.img = cap.read()

            if ret is True:
                self.run()
                cv2.imshow("Put on the sunglasses on your face", self.img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    def run(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        face_rectangles = self.detector_faces(img_rgb, 0)

        for i in range(len(face_rectangles)):
            rect = dlib.rectangle(
                            int(face_rectangles[i].left()),
                            int(face_rectangles[i].top()),
                            int(face_rectangles[i].right()),
                            int(face_rectangles[i].bottom()),
                        )
            landmarks = self.detector_landmarks(img_rgb, rect)

            if self.args.landmarks and (not self.args.webcam):
                self.save_landmarks(landmarks)

            # nose top, left and right face end points
            x = int(landmarks.parts()[27].x)
            y = int(landmarks.parts()[27].y)
            x_18 = int(landmarks.parts()[17].x)
            x_27 = int(landmarks.parts()[26].x)

            # calculate new width and height, moving distance for adjusting sunglasses
            width = int(abs(x_18 - x_27))
            scale = width / self.sun_w
            height = int(self.sun_h * scale)

            move_x = int(self.points[1] * scale)
            move_y = int(self.points[2] * scale)

            if self.args.quaternion:
                _h, _w, _ = self.img.shape
                _, roll, yaw = get_orientation(_w, _h, landmarks.parts())
                sunglasses = rotate_along_axis(self.sunglasses, width, height, phi=yaw, gamma=roll)
            else:
                sunglasses = cv2.resize(self.sunglasses, (width, height))

            # get region of interest on the face to change
            roi_color = self.img[(y - move_y):(y + height - move_y), (x - move_x):(x + width - move_x)]

            # find all non-transparent points
            index = np.argwhere(sunglasses[:, :, 3] > 0)

            for j in range(3):
                roi_color[index[:, 0], index[:, 1], j] = sunglasses[index[:, 0], index[:, 1], j]

            # set the area of the image of the changed region with sunglasses
            self.img[(y - move_y):(y + height - move_y), (x - move_x):(x + width - move_x)] = roi_color

            if not self.args.webcam:
                self.save_result()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="shape_predictor_68_face_landmarks.dat", help="dlib landmarks.")
    parser.add_argument("-i", "--image", type=str, default="example1", help="sunglasses image file name (without extension).")
    parser.add_argument("-l", "--landmarks", action="store_true", help="save landmarks with source images.")
    parser.add_argument("-w", "--webcam", action="store_true", help="put on a sunglasses on real-time.")
    parser.add_argument("-q", "--quaternion", action="store_true", help="apply quaternion to a sunglasses image.")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if args.landmarks and args.webcam:
        raise TypeError("Webcam does not support save landmarks.")

    puton = PutOn(args)

    if args.webcam:
        puton.webcam()
    else:
        puton.images()


if __name__ == "__main__":
    main()
