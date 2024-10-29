import cv2 as cv
import tensorflow as tf
import argparse
import json
import numpy as np
from model import dummy_model
from deepface import DeepFace as df

# Argument parsing general configuration
parser = argparse.ArgumentParser(prog='face detect video pipeline')
parser.add_argument('--path', default='')
parser.add_argument('--video_device', default=0, type=int)
parser.add_argument('--database_file', default='face_db.json')
parser.add_argument('--physical_location', default='')
parser.add_argument('--detection_method', default='cv')
parser.add_argument('--random_embedding', default=False)
parser.add_argument('--debug', default =False)

args = parser.parse_args()
capture_path = args.path
video_device = args.video_device
database_file = args.database_file
physical_location = args.physical_location
face_detection_method = args.detection_method
random_embedding = args.random_embedding
debug = args.debug

# Model configuration
#model = MachineLearningModel(settings)
#latent_size = 1024
#latent_size = 4096
latent_size = 512
#model = dummy_model(latent_size=latent_size, random=random_embedding)
model_input_size = [128, 128]
face_threshold = 0.75

def model(image, model_name='ArcFace', latent_size=latent_size):
    embeds = [] 
    for index in range(np.shape(image)[0]):
        embed = df.represent(image[index,:,:,:], model_name, enforce_detection=False)
        embeds.append(embed[0]['embedding'])
    user_embeddings = np.array(embeds)
    return user_embeddings

def initialize_database(database_file):
    global face_database
    with open(database_file) as json_file:
        face_database = json.load(json_file)

    #breakpoint()
    for user in face_database:
        try:
            user_embeddings = np.load(face_database[user]['face_data'])
            face_database[user]['embeddings'] = user_embeddings
        except:
            print(f"User \"{face_database[user]['name']}\" entry exists on database but is not face-registered, run \"python face_register.py {face_database[user]['name']}\" to register")
            exit()
        

def detect_faces_cv(image):
    face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces 

def detect_faces_ml(image, model_name='VGG-Face'):
    faces = []
    embeds = df.represent(image, model_name, enforce_detection=False)
    #breakpoint()
    for face in embeds:
        x = face['facial_area']['x']
        y = face['facial_area']['y']
        w = face['facial_area']['w']
        h = face['facial_area']['h']
        faces.append((x, y, w,h))
    return faces 

def extract_faces(faces, image, target_dimensions):
    face_images = []
    for (x, y, w, h) in faces:
        face = image[y:y+h,x:x+w,:]
        resized_face = cv.resize(face, target_dimensions)
        resized_face = resized_face[np.newaxis,:,:,:]
        face_images.append(resized_face)

    return face_images

def encode_faces(faces):
    stacked_faces = np.empty([0,model_input_size[0], model_input_size[1], 3])
    for face in faces:
        stacked_faces = np.vstack((stacked_faces, face))
    embeddings = model(stacked_faces)
    embeddings_list = list(embeddings)
    return embeddings_list

def calculate_similarity(embedding_a, embedding_b, method = 'cosine_similarity'):
    if method == 'cosine_similarity':
        similarity = np.dot(embedding_a, embedding_b.T)/(np.linalg.norm(embedding_a, axis=1)*np.linalg.norm(embedding_b))
        similarity = np.max(similarity)
    return similarity

def search_database(embedding):
    greatest_similarity = 0
    detected_user  = 'unknown'

    for user in face_database:
        similarity = calculate_similarity(face_database[user]['embeddings'], embedding)
        if debug: print(f'{embedding} has {similarity} similarity with {face_database[user]["name"]}')
        if similarity > greatest_similarity and similarity > face_threshold:
            greatest_similarity = similarity
            detected_user = user
    
    display = 'unknown' if detected_user == 'unknown' else face_database[detected_user]["name"]

    if debug: print(f'detected: {display}')
    return detected_user

def text_overlay(image, text, color, position):
    font = cv.FONT_HERSHEY_SIMPLEX
    position = position
    fontScale = 0.5
    thickness = 1
    image = cv.putText(image, text, position, font, 
                    fontScale, color, thickness, cv.LINE_AA)
    return image


def data_overlay(faces, users, location, image):
    overlaid_image = image

    zipped_identity = zip(faces, users)
    for identity in zipped_identity:
        status = ''
        rectangle_color = (0,255,0)
        (x, y, w, h) = identity[0]
        user = identity[1]
        if user !=  'unknown':
            allowed_locations = face_database[user]['allowed_locations']
            if location not in allowed_locations:
                status = ' [trespassing]'
                rectangle_color = (0,0,255)
        else:
            status = ' [trespassing]'
            rectangle_color = (0,0,255)
        
        display = 'unknown' if user == 'unknown' else face_database[user]["name"]
        user_text = display + status
        overlaid_image = cv.rectangle(overlaid_image, (x, y), (x+w, y+h), rectangle_color, 2)
        overlaid_image = text_overlay(overlaid_image, user_text, rectangle_color,(x, y-10))
    return overlaid_image

def detection_pipeline(image, detection_method):
    users = []

    if detection_method == 'cv':
        faces = detect_faces_cv(image)
    elif detection_method == 'ml':
        faces = detect_faces_ml(image)
    else:
        print('Invalid detection method, exiting')
        exit()
    
    if len(faces):
        face_images = extract_faces(faces, image, model_input_size)
        embeddings = encode_faces(face_images)
        if debug: print(f' Extracted {len(embeddings)} faces')

        for embedding in embeddings:
            user = search_database(embedding)
            users.append(user)
            image = data_overlay(faces, users, location=physical_location, image=image)

    if debug: print(f'detected users: {users}') 
    
    
    return image

def video_inference(device, method):
    video_capture = cv.VideoCapture(device)
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break
        
        overlaid_frame = detection_pipeline(video_frame, method) 
        cv.imshow("Video face detection", overlaid_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            video_capture.release()
            cv.destroyAllWindows()
            break

def image_inference(path, method):
    image = cv.imread(path)
    overlaid_image = detection_pipeline(image, method)
    cv.imshow("Single image detection", overlaid_image)
    cv.waitKey(0)

    return image

if __name__ == '__main__':
    initialize_database(database_file)

    if capture_path == '':
        video_inference(video_device, face_detection_method)
    else:
        image_inference(capture_path, face_detection_method)