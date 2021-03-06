import glob
import os
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
    for data_set in ['set0', 'set1', 'set2', 'set3', 'set4', 'set5', 'set6', 'set7', 'set8']:
        path = os.path.join('sets', data_set)
        N = len(glob.glob(os.path.join(path, '*.png')))

        image_paths = [os.path.join(path, '%d.png' % i) for i in range(N)]

        with open(os.path.join(path, 'correct.txt')) as correct_file:
            correct_answers = [int(x.strip()) for x in correct_file.readlines()]

        elements = [Element(image_path) for image_path in image_paths]
        element_number = len(elements)

        similarity = [[None] * element_number for _ in range(element_number)]

        for i in range(element_number):
            for j in range(element_number):
                similarity[i][j] = Element.similarity(elements[i], elements[j]) if i != j else float('-inf')

        best_fit = []
        for i in similarity:
            best_fit.append(sorted(range(len(i)), key=i.__getitem__, reverse=True))

        everything_correct = True

        for i in range(element_number):
            correct = correct_answers[i]
            predicted = best_fit[i][0]

            if correct != predicted:
                everything_correct = False

                def ordinal(n):
                    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

                print('%s| mismatched element %d: predicted %d, should be %d (%s guess)' %
                      (data_set, i, predicted, correct, ordinal(best_fit[i].index(correct) + 1)))

                element_compared = elements[i]
                element_predicted = elements[predicted]
                element_correct = elements[correct]

                cv2.imshow("Compared", element_compared.representation())
                cv2.imshow("Predicted", element_predicted.representation())
                cv2.imshow("Correct", element_correct.representation())

                cv2.imshow("mismatch comparison", Element.cut_representation(element_compared, element_predicted))
                cv2.imshow("correct comparison", Element.cut_representation(element_compared, element_correct))

                key = cv2.waitKey()
                if key == 27:  # ESC
                    exit(0)

        if everything_correct:
            print('Everything in %s was correct' % data_set)
