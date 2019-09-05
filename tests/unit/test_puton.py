import os

from main import PutOn


def test_puton_saved_images(get_args):
    args = get_args

    puton = PutOn(args)
    puton.images()

    realpath = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.abspath(os.path.join(realpath, "..", "..", "results"))
    dst_path = os.path.abspath(os.path.join(realpath, "..", "data"))

    sources = []
    sources.append(os.path.join(src_path, "landmarks_center.jpg"))
    sources.append(os.path.join(src_path, "puton_center.jpg"))

    targets = []
    targets.append(os.path.join(dst_path, "landmarks_center.jpg"))
    targets.append(os.path.join(dst_path, "puton_center.jpg"))

    for src, tar in zip(sources, targets):
        with open(src, "rb") as img:
            src_img = img.read()
        with open(tar, "rb") as img:
            tar_img = img.read()

        assert src_img == tar_img


def test_puton_saved_quaternion_images(get_args):
    args = get_args
    args.landmarks = False
    args.quaternion = True

    puton = PutOn(args)
    puton.images()

    realpath = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.abspath(os.path.join(realpath, "..", "..", "results"))
    dst_path = os.path.abspath(os.path.join(realpath, "..", "data"))

    sources = []
    sources.append(os.path.join(src_path, "puton_soft_right.jpg"))

    targets = []
    targets.append(os.path.join(dst_path, "puton_soft_right.jpg"))

    for src, tar in zip(sources, targets):
        with open(src, "rb") as img:
            src_img = img.read()
        with open(tar, "rb") as img:
            tar_img = img.read()

        assert src_img == tar_img
