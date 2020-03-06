import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow import keras
import cv2
from sklearn.model_selection import train_test_split

# TPU_WORKER = 'grpc://10.240.1.2:8470'
#
#
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
#



import pandas as pd
images_path ="D:\miia4406-movie-genre-classification\images"
df_train =open("D:\miia4406-movie-genre-classification/dataTraining.csv").readlines()
# df_test =open("D:\miia4406-movie-genre-classification/dataTesting.csv").readlines()

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
def create_batch(X,Y,batch_size):
    sub_x=np.zeros((500,224,224,3))
    sub_y=np.zeros((500,24))
    indices=np.random.choice(range(len(Y)),batch_size)
    for i in indices:
        im =cv2.imread(X[i].replace("\"","")+".jpeg")
        im=cv2.resize(im,(224,224))
        sub_x[i,:,:,:]=im
        temp = np.zeros(1,24)
        temp[Y[i]]=1
        sub_y[i,:]=temp


    return tf.constant(sub_x,dtype=tf.float32),tf.constant(sub_y,dtype=tf.float32)



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
X_train,X_test,y_train,y_test=train_test_split(ids,labels,test_size=0.1)
BATCH_SIZE=500






feature_extractor_url="https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4"


feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224,224,3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(len(set_labels), activation=keras.activations.hard_sigmoid,dtype=tf.float32)
],name="Modelo_peliculas")
optim=keras.optimizers.Adam(learning_rate=0.0001)
metrics=[keras.metrics.CategoricalAccuracy(name="Categorical_accuracy"),
        keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn')]
model.compile(optimizer=optim,loss=tf.keras.metrics.CategoricalAccuracy,metrics=metrics)

for i in range(10):
    x,y =create_batch(X_train,y_train,BATCH_SIZE)
    x_test,y_test_mini = create_batch(X_test,y_test,BATCH_SIZE)
    model.fit(x=x,y=y,epochs=10,validation_data=[x_test,y_test_mini])


model.save("modelo.h5")
pass