import cv2 as cv
import tensorflow as tf
import argparse
import json
import os
import numpy as np
from model import dummy_model
from deepface import DeepFace as df
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='face registration app')
parser.add_argument('user')
parser.add_argument('--database_file', default='./face_db.json')
parser.add_argument('--database_folder',default='./face_data/')
parser.add_argument('--image_folder', default = '')
parser.add_argument('--random', default = False)

args = parser.parse_args()

user = args.user
database_file = args.database_file
database_folder = args.database_folder
image_folder = args.image_folder
random_embedding = args.random
latent_size = 512
model = dummy_model(latent_size=latent_size, random=random_embedding)

def model(image, model_name='ArcFace', latent_size=latent_size):
    #breakpoint()
    image = image[:,:,:]
    embeds = df.represent(image, model_name, enforce_detection=False)
    user_embeddings = np.empty([0,latent_size])
    for embed in embeds:
        user_embeddings = np.vstack((user_embeddings, np.array(embed['embedding']).reshape([1,-1])))

    return user_embeddings

def encode_face (face_image):
    embedding = model(face_image)
    return embedding

def store_embedding(user, embedding):
    user_id = return_id(user)
    user_embedding_path = database[user_id]['face_data']

    os.makedirs(os.path.join("face_data/", user), exist_ok=True)

    try:
        user_embeddings = np.load(user_embedding_path)
    except:
        print(f'No embedding files initialized for user id: {user_id}, user: {user}, creating now.')
        user_embeddings = np.empty([0,latent_size])
        np.save(database[user_id]["face_data"], user_embeddings)
    if user_embeddings.size == 0:
        print('No previous embeddings, inserting')
    else:
        past_embeddings = len(user_embeddings)
        print(f'Found {past_embeddings} embeddings. Adding a new entry.')
    user_embeddings = np.vstack((user_embeddings, embedding))
    np.save(user_embedding_path, user_embeddings)

def return_id(user):
    for id in database.keys():
        if database[id]['name'] == user:
            return id
    return -1

def crop_square(image):
    image_dims = np.shape(image)
    if image_dims[0] < image_dims[1]:
        offset = (image_dims[1] - image_dims[0])//2
        cropped = image[:,offset:offset+image_dims[0],:]
    else:
        offset = (image_dims[0] - image_dims[1])//2
        cropped = image[offset:offset+image_dims[1],:,:]

    return cropped

def video_register(user):
    video_capture = cv.VideoCapture(0)
    os.makedirs(f'./images/{user}', exist_ok=True)
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break

        cv.imshow("Video face detection", crop_square(video_frame))

        if cv.waitKey(1) & 0xFF == ord("r"):
            print(f'Registering current frame')
            frame_embedding = encode_face(video_frame)
            cv.imwrite(f"./images/{user}/{user}_{int(time.time())}.jpg",video_frame)
            store_embedding(user, frame_embedding)

        elif cv.waitKey(1) & 0xFF == ord("q"):
            video_capture.release()
            cv.destroyAllWindows()
            break

def image_register(user, image_folder):
    file_list = os.listdir(image_folder)
    folder_full = os.path.join(os.path.abspath(os.getcwd()), image_folder)
    for image_file in tqdm(file_list):
        face_image = cv.imread(os.path.join(folder_full,image_file))
        embedding = encode_face(face_image)
        store_embedding(user, embedding)

if __name__ == '__main__':
    with open(database_file) as json_file:
        database = json.load(json_file)
    if image_folder == '':
        video_register(user)
    else:
        image_register(user, image_folder)