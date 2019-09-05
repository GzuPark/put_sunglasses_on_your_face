import pytest


@pytest.fixture()
def check_url():
    url = {}
    filename = "shape_predictor_68_face_landmarks.dat"
    url["dlib"] = f"https://raw.githubusercontent.com/davisking/dlib-models/master/{filename}.bz2"

    return url


@pytest.fixture()
def get_args():
    class Args(object):
        model = "shape_predictor_68_face_landmarks.dat"
        image = "example1"
        landmarks = True
        webcam = False
        quaternion = False

    return Args
