import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# TPU_WORKER = 'grpc://10.240.1.2:8470'
#
#
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
#


# Esto se teiene que cambiar a la ruta donde esten las imágenes y el archivo CSV del dataset
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

def create_batch(X,Y,batch_size,tupl=False):
    sub_x=np.zeros((batch_size,224,224,3))
    sub_y=np.zeros((batch_size,24))
    indices=np.random.choice(range(len(Y)),batch_size)
    j=0
    for i in indices:
        im =ImportError.open(X[i].replace("\"","")+".jpeg")
        im=im.resize(224,224,Image.ANTIALIAS)
        im =np.asarray(im)
        sub_x[j,:,:,:]=im
        temp = np.zeros(24)
        temp[Y[i]]=1
        sub_y[j,:]=temp
        j+=1

    data= tf.data.Dataset.from_tensor_slices((tf.constant(sub_x, dtype=tf.float32), tf.constant(sub_y, dtype=tf.float32))).batch(10)
    tupla=tf.constant(sub_x,dtype=tf.float32),tf.constant(sub_y,dtype=tf.float32)
    if not tupl:
        return data
    return sub_x,sub_y


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


import pickle
with open("x_train","w+b") as fp:
    pickle.dump(X_train,fp)
with open("x_test","w+b") as fp:
    pickle.dump(X_test,fp)
with open("y_train","w+b") as fp:
    pickle.dump(y_train,fp)
with open("y_test", "w+b") as fp:
    pickle.dump(y_test,fp)



BATCH_SIZE=500

# X_train =None
# X_test =None
# y_train =None
# y_test =None
# with open("x_train","r+b") as fp:
#     X_train=pickle.load(fp)
# with open("x_test","r+b") as fp:
#     X_test=pickle.load(fp)
# with open("y_train","r+b") as fp:
#      y_train=pickle.load(fp)
# with open("y_test", "r+b") as fp:
#     y_test=pickle.load(fp)
feature_extractor_url="https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4"


feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224,224,3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(24, activation=keras.activations.hard_sigmoid,dtype=tf.float32)
],name="Modelo_peliculas")
optim=keras.optimizers.Adam(learning_rate=0.0001)
metrics=[keras.metrics.CategoricalAccuracy(name="Categorical_accuracy"),
        keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn')]
loss=tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optim,loss=loss,metrics=metrics)
with tf.device("GPU:0"):
    for i in range(10):
        x,y=create_batch(X_train,y_train,BATCH_SIZE,tupl=True)
        x_test,Y_test= create_batch(X_test,y_test,BATCH_SIZE,tupl=True)

        model.fit(x,y,batch_size=50,epoch=10,validation_data=(x_test,Y_test),validation_steps=10)
        # model.evaluate(x_test,Y_test)


model.save("modelo.h5")
pass