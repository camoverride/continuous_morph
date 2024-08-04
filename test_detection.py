import dlib
import cv2



def detect_face(frame):
    predictor_model_path='shape_predictor_68_face_landmarks.dat'
    shape_predictor = dlib.shape_predictor(predictor_model_path)
    detector = dlib.get_frontal_face_detector()
    dlib_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(dlib_image, 1)

    return dets


im = cv2.imread("__debug_cam_img.jpg")
im = cv2.imread("derp.png")

p = detect_face(im)


print(p)

