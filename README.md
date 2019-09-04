# Put Sunglasses on your Face

This is a python application that combine your face and a sunglasses image with [OpenCV](https://opencv.org/) and [Dlib](http://dlib.net/) library.

- Blogs
  - Korean: [Computer Vision - 선글라스를 2D 가상 착용해보기](https://gzupark.github.io/blog/Put-Sunglasses-on-your-Face/)

## Environment

- `Python ~= 3.6`
  - `numpy==1.17.1`
  - `dlib==19.17.0`
  - `opencv-python==4.1.0.25`
- Download the dlib landmark detector model file and unzip
  - `python assets/models/download_dlib_shape_predictor_68.py`

## How to use

### Load images

```sh
python main.py
```

```sh
# save landmarks
python main.py -l
```

### Using a webcam

```sh
python main.py -w
```

### Applying quaternion

```sh
python main.py -q
```

```sh
python main.py -q -w
```
