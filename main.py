import glob
import os
from random import shuffle
import sys
from element import Element
from piro_lib import *

if len(sys.argv) > 1:

    path = sys.argv[1]
    N = int(sys.argv[2])

    image_paths = [os.path.join(path, '%d.png' % i) for i in range(N)]

    elements = [Element(image_path) for image_path in image_paths]
    element_number = len(elements)

    similarity = [[None]*element_number for _ in range(element_number)]

    for i in range(element_number):
        for j in range(element_number):
            similarity[i][j] = Element.similarity(elements[i], elements[j]) if i != j else float('-inf')

    for i in similarity:
        best_fit = sorted(range(len(i)), key=i.__getitem__, reverse=True)
        print(' '.join([str(x) for x in best_fit]))

else:
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
