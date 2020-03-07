import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
images_path ="D:\miia4406-movie-genre-classification\images"
df_train =open("D:\miia4406-movie-genre-classification/dataTraining.csv").readlines()
# df_test =open("D:\miia4406-movie-genre-classification/dataTesting.csv").readlines()
TPU_WORKER = 'grpc://10.240.1.2:8470'


resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

set_labels = set([])
dict_labels={}
def add_prefix(string):
    return images_path+"/"+string
def give_id(string):
    temp=string.split(",")
    return temp[0]
def transform_labels(string):
    label=[]
    for thing in string.split(","):
        label.append(set_labels.index(thing.strip()))
    return label
import numpy as np

def give_label(string):
    cos = string.partition("[")[2].partition("]")[0]
    cos = cos.replace("\'","")
    cos1 = cos.split(",")
    for i in cos1:
        set_labels.add(i.strip())
    return cos.strip()
del df_train[0]
labels =list(map(give_label,df_train))
set_labels =list(set_labels)
labels =list(map(transform_labels,labels))

# for i ,lab in enumerate(set_labels):
#     dict_labels[lab]=i
ids = list(map(add_prefix,list(map(give_id,df_train))))



feature_extractor_url="https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4"

with strategy.scope():
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224,224,3))
    feature_extractor_layer.trainable = False

    model = tf.keras.Sequential([
                feature_extractor_layer,
                layers.Dense(len(set_labels), activation=keras.activations.hard_sigmoid,dtype=tf.float32)])

    optim=keras.optimizers.Adam(learning_rate=0.0001)
metrics=
model.compile(optimizer=optim,loss=tf.keras.metrics.CategoricalAccuracy,)

pass