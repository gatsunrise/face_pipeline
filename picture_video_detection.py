import cv2 as cv
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(prog='face detect video pipeline')
#parser.add_argument('model')
parser.add_argument('path')
args = parser.parse_args()

def detect_faces(image):
    face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces 

def face_overlay(faces, image):
    overlaid_image = image
    for (x, y, w, h) in faces:
        overlaid_image = cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 210), 2)
        overlaid_image =  image[y:y+h,x:x+w,:]
    return overlaid_image

def video_capture(device):
    video_capture = cv.VideoCapture(0)
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break

        faces = detect_faces(video_frame) 
        overlaid_frame = face_overlay(faces,video_frame)
        cv.imshow("Video face detection", overlaid_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            video_capture.release()
            cv.destroyAllWindows()
            break

def image_capture(path):
    image = cv.imread(path)
    faces = detect_faces(image)
    overlaid_image = cv.resize(face_overlay(faces, image),[800,450])
    
    cv.imshow("Single image detection", overlaid_image)
    cv.waitKey(0)

    return image

if __name__ == '__main__':

    capture_path = args.path
    if capture_path == '0':
        video_capture(capture_path)
    else:
        image_capture(capture_path)