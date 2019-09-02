# Put Sunglasses on your Face

This is a python application that combine your face and a sunglasses image with OpenCV library.

## Environment

- `Python ~= 3.6`
  - `numpy==1.17.1`
  - `dlib==19.17.0`
  - `opencv-python==4.1.0.25`
- Download the dlib landmark detector model file and unzip
  - [download](https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2)
  - `bzip2 -d ${PATH}/assets/models/shape_predictor_68_face_landmarks.dat.bz2`

## How to use

### Load images

```sh
python main.py
```

```sh
# save landmarks
python main.py --landmarks
```

### Using a webcam

```sh
python main.py --webcam
```

### Applying quaternion

```sh
python main.py --quaternion
```

```sh
python main.py --quaternion --webcam
```
