from deepface import DeepFace as df
import tensorflow as tf
import keras

tf.keras.models.load_model("./vgg_face_weights.h5")