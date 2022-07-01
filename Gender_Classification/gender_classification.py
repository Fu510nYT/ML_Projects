import cv2
import cvlib
import sys
import numpy as np
from numpy.lib.type_check import imag
print("Successfully Imported Libraries")

cap = cv2.VideoCapture(0)
padding = 20

while True:
    _, img = cap.read()
    if img is None: continue

    face_coord, confidence = cvlib.detect_face(img)

    for i in face_coord:
        (x, y) = max(0, i[0] - padding), max(0, i[1] - padding)
        (x2, y2) = min(img.shape[1] - 1, i[2] + padding), min(img.shape[0] - 1, i[3] + padding)
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        crop = np.copy(img[y:y2, x:x2])
        (label, confidence) = cvlib.detect_gender(crop)
        idx = np.argmax(confidence)
        label = label[idx]
        label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
        Y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("img", img)

    if cv2.waitKey(1) in [ord('q'), 27]:break

cv2.destroyAllWindows()