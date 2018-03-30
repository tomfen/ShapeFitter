import glob
import os
from random import shuffle

from element import Element
from piro_lib import *

print('OpenCV version: ' + cv2.__version__)

path = os.path.join('sets', '**', '*.png')

file_paths = glob.glob(path)
shuffle(file_paths)

for image_path in file_paths:

    print(image_path)

    element = Element(image_path)

    cv2.imshow("Image", element.representation())
    cv2.imshow("Cut", element.cut_representation())

    key = cv2.waitKey()
    if key == 27:  # ESC
        exit(0)
