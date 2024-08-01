import os
import time
import yaml

import cv2
import dlib # TODO: purge this library
import numpy as np
import scipy.ndimage

from PIL import Image # TODO: purge this library
import PIL.Image


def apply_affine_transform(src, srcTri, dstTri, size):
    """
    Apply affine transform calculated using srcTri and dstTri to src and output an image of size.
    """
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img
    """
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect, t2Rect, tRect = [], [], []

    for i in range(3):
        tRect.append((t[i][0] - r[0], t[i][1] - r[1]))
        t1Rect.append((t1[i][0] - r1[0], t1[i][1] - r1[1]))
        t2Rect.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


def generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri_list, size, output_dir, iteration):
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

        res.save(os.path.join(output_dir, f"{iteration - 1}-{iteration}_{j:03d}.jpg"), 'JPEG')
        print("saved image!")


def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
    """
    Align function from FFHQ dataset pre-processing step
    https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    """

    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file).convert('RGBA').convert('RGB')

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = PIL.Image.fromarray(img, 'RGBA')
        else:
            img = PIL.Image.fromarray(img, 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    img.save(dst_file, 'PNG')

    return img


class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass


def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]


def crop_image(x, img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        scale0 = size1[0]/size2[0]
        scale1 = size1[1]/size2[1]
        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        scale0 = size2[0]/size1[0]
        scale1 = size2[1]/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]
    
    else:
        return [img1[:,diff1:avg1],img2[-diff0:avg0,:]]


def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        return [img1,img2[-diff0:avg0,-diff1:avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]

    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]


def generate_face_correspondences(theImage1, theImage2):
    """
    Detect the points of face.
    """
    print("generate_face_correspondences - detecting!")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68,2))

    print("generate_face_correspondences - cropping")
    imgList = crop_image(theImage1,theImage2)
    list1 = []
    list2 = []
    j = 1


    print("generate_face_correspondences - loop")
    for img in imgList:

        size = (img.shape[0],img.shape[1])
        if(j == 1):
            currList = list1
        else:
            currList = list2

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        dets = detector(img, 1)

        try:
            if len(dets) == 0:
                raise NoFaceFound
        except NoFaceFound:
            print("Sorry, but I couldn't find a face in the image.")

        j=j+1

        for k, rect in enumerate(dets):
            
            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)
            # corresp = face_utils.shape_to_np(shape)
            
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y
                # cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

            # Add back the background
            currList.append((1,1))
            currList.append((size[1]-1,1))
            currList.append(((size[1]-1)//2,1))
            currList.append((1,size[0]-1))
            currList.append((1,(size[0]-1)//2))
            currList.append(((size[1]-1)//2,size[0]-1))
            currList.append((size[1]-1,size[0]-1))
            currList.append(((size[1]-1),(size[0]-1)//2))

    print("generate_face_correspondences - adding back")
    # Add back the background
    narray = corresp/2
    narray = np.append(narray,[[1,1]],axis=0)
    narray = np.append(narray,[[size[1]-1,1]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = np.append(narray,[[1,size[0]-1]],axis=0)
    narray = np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = np.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)

    print("generate_face_correspondences - finishing up!")
    
    return [size,imgList[0],imgList[1],list1,list2,narray]


def rect_contains(rect, point):
    """
    Check if a point is inside a rectangle
    """

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay(f_w, f_h, subdiv, dictionary1):
    """
    Write the delaunay triangles into a file
    """
    list_4 = []

    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            list_4.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))

    dictionary1 = {}
    return list_4


def make_delaunay(f_w, f_h, theList, img1, img2):

    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary. 
    theList = theList.tolist()
    points = [(int(x[0]),int(x[1])) for x in theList]
    dictionary = {x[0]:x[1] for x in list(zip(points, range(76)))}
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)
   
    # Return the list.
    return list4


def detect_face(frame):
    """
    Detect whether an image contains a face.
    """
    predictor_model_path='shape_predictor_68_face_landmarks.dat'
    shape_predictor = dlib.shape_predictor(predictor_model_path)
    detector = dlib.get_frontal_face_detector()
    dlib_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(dlib_image, 1)

    return dets



class LandmarksDetector:
    def __init__(self, predictor_model_path='shape_predictor_68_face_landmarks.dat'):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except:
                print("Exception in get_landmarks()!")

# Load model
landmarks_detector = LandmarksDetector()
def get_landmarks(face_image):
    """
    Gets the face landmarks in an image that contains a face.
    """
    return list(landmarks_detector.get_landmarks(face_image))[0]


if __name__ == "__main__":


    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Webcam or Pi Camera?
    if config["camera"] == "webcam":
        cam = cv2.VideoCapture(0)
    elif config["camera"] == "picam":
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        picam2.start()

    # Initial image (required to start)
    CURRENT_IMAGE = "captured_images/1.jpg"

    # Start my writing the second face.
    face_capture_index = 2

    # Main event loop
    while True:
        # Take a photo (from webcam or pi cam)
        if config["camera"] == "webcam":
            ret, frame = cam.read()
        elif config["camera"] == "picam":
            frame = picam2.capture_array()

        # Convert to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check if there is a face
        face = detect_face(gray)

        if face:
            # Write the new image.
            print("Face detected!")
            NEW_IMAGE = f"captured_images/{face_capture_index}.jpg"
            cv2.imwrite(NEW_IMAGE, frame)

            # Load the current face image and new face image
            print(f"---- morphing between {CURRENT_IMAGE} - {NEW_IMAGE}")
            img1 = cv2.imread(CURRENT_IMAGE)
            img2 = cv2.imread(NEW_IMAGE)

            # Get the landmarks
            landmarks_1 = get_landmarks(CURRENT_IMAGE)
            landmarks_2 = get_landmarks(NEW_IMAGE)

            # Align the images
            img1 = image_align(CURRENT_IMAGE, "__current_output.png", landmarks_1, output_size=1024) #x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
            img2 = image_align(NEW_IMAGE, "__new_output.png", landmarks_2, output_size=1024) #x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)

            # Resize images to reduce computational load
            img1 = np.array(img1)
            img2 = np.array(img2)

            # Generate correspondences
            [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)

            # Get the delauney triangle
            tri = make_delaunay(size[1], size[0], list3, img1, img2)

            # Generate the morph sequence
            generate_morph_sequence(duration=5,
                                    frame_rate=10,
                                    img1=img1,
                                    img2=img2,
                                    points1=points1,
                                    points2=points2,
                                    tri_list=tri,
                                    size=(240, 320),
                                    output_dir="test_morphs",
                                    iteration=face_capture_index)

            # Update the current image.
            CURRENT_IMAGE = NEW_IMAGE

            # Update the capture count.
            face_capture_index += 1

            # Wait a little bit
            time.sleep(1)
        

        else:
            time.sleep(5)
