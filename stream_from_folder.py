import os
import cv2
import time

# Set to fullscreen
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



paths = sorted([os.path.join("test_morphs", im) for im in os.listdir("test_morphs")])
print(paths)
for im in paths:
    frame = cv2.imread(im)
    cv2.imshow("window", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break