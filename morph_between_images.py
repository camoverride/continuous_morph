import os
import numpy as np
import cv2
from PIL import Image
import dlib
import time
import yaml
from scipy.ndimage import gaussian_filter
from collections import defaultdict

# Constants
PREDICTOR_MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'

# Exception class for when no face is found in the image
class NoFaceFound(Exception):
    """Raised when no face is detected in the image."""
    pass

# Apply affine transform calculated using srcTri and dstTri to src and output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size):
    """
    Applies an affine transform to an image region defined by triangular points.
    
    Args:
        src (array): Source image.
        srcTri (list of tuples): Source triangle vertices.
        dstTri (list of tuples): Destination triangle vertices.
        size (tuple): Size of the output image.
        
    Returns:
        array: Transformed image.
    """
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """
    Warps and alpha blends triangular regions between two images based on given triangles.
    
    Args:
        img1 (array): First image.
        img2 (array): Second image.
        img (array): Output image.
        t1 (list): Triangles in the first image.
        t2 (list): Corresponding triangles in the second image.
        t (list): Triangles in the destination image.
        alpha (float): Alpha blending amount.
    """
    # Calculate bounding rectangles for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by the top left corner of the respective bounding rectangles
    t1Rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2Rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    tRect = [(t[i][0] - r[0], t[i][1] - r[1]) for i in range(3)]

    # Create a mask for the triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Warp triangles from images to the output image
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, (r[2], r[3]))
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, (r[2], r[3]))

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Put the blended image back to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

# Landmarks detector class using dlib
class LandmarksDetector:
    def __init__(self, predictor_model_path=PREDICTOR_MODEL_PATH):
        """
        Initializes the landmarks detector with the path to the predictor model.
        
        Args:
            predictor_model_path (str): Path to the dlib face predictor model.
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        """
        Detects landmarks in a given image.
        
        Args:
            image (str): Path to the image file.
            
        Yields:
            list of tuples: Coordinates of the landmarks detected in the image.
        """
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except Exception as e:
                print(f"Exception in get_landmarks(): {e}")

# Function to align images based on face landmarks
def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
    """
    Aligns and crops an image based on detected face landmarks.
    
    Args:
        src_file (str): Path to the source image file.
        dst_file (str): Path to the output image file.
        face_landmarks (list): Detected face landmarks.
        output_size (int): Output size of the image.
        transform_size (int): Size for the transformation.
        enable_padding (bool): If true, enables padding.
        x_scale (float): Scale for the x-axis transformation.
        y_scale (float): Scale for the y-axis transformation.
        em_scale (float): Scale for the eye-mouth distance.
        alpha (bool): If true, handles alpha channel in the image.
        
    Returns:
        Image: The aligned image.
    """
    # Load and convert the image
    img = Image.open(src_file).convert('RGBA').convert('RGB')

    # Calculate auxiliary vectors for transformation
    lm = np.array(face_landmarks)
    eye_left = np.mean(lm[36:42], axis=0)
    eye_right = np.mean(lm[42:48], axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm[48]
    mouth_right = lm[54]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Set up the transformation
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Perform transformations
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.LANCZOS)

    # Save the aligned image
    img.save(dst_file, 'PNG')
    return img

# Function to generate morph sequence between two images
def generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri_list, size, output_dir, iteration):
    """
    Generates a morph sequence between two images based on provided parameters.
    
    Args:
        duration (int): Duration of the morph sequence in seconds.
        frame_rate (int): Frame rate of the output video.
        img1 (array): First image.
        img2 (array): Second image.
        points1 (list): Landmarks for the first image.
        points2 (list): Landmarks for the second image.
        tri_list (list): List of triangles for morphing.
        size (tuple): Size of the output images.
        output_dir (str): Directory to save the output images.
        iteration (int): Current iteration number.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_images = int(duration * frame_rate)
    
    for j in range(num_images):
        alpha = j / (num_images - 1)
        points = [(int((1 - alpha) * x1 + alpha * x2), int((1 - alpha) * y1 + alpha * y2)) for (x1, y1), (x2, y2) in zip(points1, points2)]
        morphed_frame = np.zeros_like(img1, dtype=img1.dtype)
        
        for i in range(len(tri_list)):
            x, y, z = tri_list[i]
            t1, t2, t = [points1[x], points1[y], points1[z]], [points2[x], points2[y], points2[z]], [points[x], points[y], points[z]]
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
        
        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save(os.path.join(output_dir, f"{iteration}frame_{j:04d}.jpg"), 'JPEG')

# Main execution starts here
if __name__ == "__main__":
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    cam_type = config.get("camera", "webcam")
    if cam_type == "webcam":
        cam = cv2.VideoCapture(0)
    elif cam_type == "picam":
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        picam2.start()

    landmarks_detector = LandmarksDetector()

    i = 2
    CURRENT_IMAGE = "captured_images/1.jpg"
    while True:
        if cam_type == "webcam":
            ret, frame = cam.read()
        elif cam_type == "picam":
            frame = picam2.capture_array()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = detect_face(gray)

        if faces_detected:
            NEW_IMAGE = f"captured_images/{i}.jpg"
            cv2.imwrite(NEW_IMAGE, frame)

            img1 = cv2.imread(CURRENT_IMAGE)
            img2 = cv2.imread(NEW_IMAGE)

            landmarks_1 = list(landmarks_detector.get_landmarks(CURRENT_IMAGE))[0]
            landmarks_2 = list(landmarks_detector.get_landmarks(NEW_IMAGE))[0]

            img1_aligned = image_align(CURRENT_IMAGE, "__current_output.png", landmarks_1, output_size=1024)
            img2_aligned = image_align(NEW_IMAGE, "__new_output.png", landmarks_2, output_size=1024)

            img1 = np.array(img1_aligned)
            img2 = np.array(img2_aligned)

            duration = 5  # seconds
            frame_rate = 10  # frames per second
            size = (240, 320)
            output_dir = "test_morphs"

            [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
            tri_list = make_delaunay(size[1], size[0], list3, img1, img2)

            generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri_list, size, output_dir, iteration=i)

            CURRENT_IMAGE = NEW_IMAGE
            i += 1
            time.sleep(1)

        else:
            time.sleep(5)
