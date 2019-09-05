import bz2
import os

from urllib import request


def main():
    realpath = os.path.dirname(os.path.realpath(__file__))
    filename = "shape_predictor_68_face_landmarks.dat"
    filepath = os.path.join(realpath, filename)

    url = f"https://raw.githubusercontent.com/davisking/dlib-models/master/{filename}.bz2"
    request.urlretrieve(url, f"{filepath}.bz2")
    print(f"Downloaded: {filename}.bz2")

    zipfile = bz2.BZ2File(f"{filepath}.bz2")
    data = zipfile.read()
    open(filepath, "wb").write(data)
    print(f"Decompressed: {filename}")

    os.chmod(f"{filepath}.bz2", 0o777)
    os.remove(f"{filepath}.bz2")


if __name__ == "__main__":
    main()
