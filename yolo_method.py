import os
import time
import cv2
import numpy as np
from yolo_model import YOLO


def process_image(img):
    """Returns resized image with pixel values 0-1."""
    image = np.zeros((416, 416))
    try:
        image = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
        image = np.array(image, dtype = "float32")
        image /= 255.
        image = np.expand_dims(image, axis=0)
    except:
        pass
    return image


def get_classes(file):
    """Returns list with names of classes read from a file."""
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draws the boxes over objects found in an image, adds their description and probability

    # Arguments:
        image : ndarray, image for object detection
        boxes : ndarray, boxes of detected objects
        scores : ndarray, scores of detected objects
        classes : ndarray, classes of detected objects
        all_classes : list, all available classes
    """
    for box, score, cl in zip(boxes, scores, classes):
        # make sure that box dimensions are not out of boundaries
        x, y, w, h = box
        top = max(0, np.round(x).astype(int))
        left = max(0, np.round(y).astype(int))
        right = min(image.shape[1], np.round(x+w).astype(int))
        bottom = min(image.shape[0], np.round(y+h).astype(int))
        # add box and info to the image
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, "{0} {1:.2f}".format(all_classes[cl], score), (top, left-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        print("class: {0}, score: {1:.2f}".format(all_classes[cl], score))
        print("box coordinates x, y, w, h: {0}".format(box))
    

def detect_image(image, yolo, all_classes):
    """Returns image with boxes drawn on detected objects

    # Arguments:
        image : ndarray, image for object detection
        yolo : class, YOLO method instance
        all_classes : list, all available classes

    # Returns:
        image : ndarray, image with drawn on boxes
    """
    pimage = process_image(image)
    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()
    print("time: {0:.2f}s".format(end - start))
    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)
    return image


def detect_video(video, yolo, all_classes):
    """Detects objects on a video from a path

    # Arguments:
        video : string, name of a video file
        yolo : class, YOLO method instance
        all_classes : list, all available classes
    """
    video_path = os.path.join("videos", "test", video)
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    # prepare for saving detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*"mpeg")
    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)
    # read the video and detect objects frame by frame
    while True:
        res, frame = camera.read()
        if not res:
            break
        image = detect_image(frame, yolo, all_classes)
        # save the video frame by frame
        vout.write(image)
        image = resize_with_shape(frame, 1000)
        cv2.imshow("detection", image)
        if cv2.waitKey(110) & 0xff == 27:
            break
    vout.release()
    camera.release()


def resize_with_shape(image, width=None, height=None):
    """Returns resized image while keeping proportions of the original

    # Arguments:
        image : ndarray, image to resize
        width : int, new width of the image
        height : int, new height of the image

    # Returns:
        image : ndarray, resized image
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        scale = height / float(h)
        dim = (int(w * scale), height)
    else:
        scale = width / float(w)
        dim = (width, int(h * scale))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# threshold parameters (object threshold and mns threshold)
yolo = YOLO(0.4, 0.5)
file = "coco_classes.txt"
all_classes = get_classes(file)
# Detecting images
img_name = "people2.jpg"
img_path = os.path.join("images", "test", img_name)
image = cv2.imread(img_path)
image = resize_with_shape(image, 1000)
image = detect_image(image, yolo, all_classes)
cv2.imwrite("images/res" + img_name, image)
cv2.imshow("detection result", image)
cv2.waitKey(100)
# Detecting video
video = "video.mp4"
detect_video(video, yolo, all_classes)
      