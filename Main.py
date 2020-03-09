# import cv2
import pickle

import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

images_path ="../images"
df_train =open("../dataTraining.csv").readlines()

feature_extractor_url="https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4"

# df_test =open("D:\miia4406-movie-genre-classification/dataTesting.csv").readlines()
PATH_TO_BATCH="BATCH"

TPU_WORKER = 'grpc://10.240.1.2:8470'
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
BATCH_SIZE=512
set_labels = set([])
dict_labels={}
# feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3))
# feature_extractor_layer.trainable = False
def crear_modelo():
    base_model=tf.keras.applications.ResNet101(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    model = tf.keras.Sequential([base_model,global_average_layer,keras.layers.BatchNormalization(),keras.layers.Dropout(rate=0.2),layers.Dense(len(set_labels), activation=keras.activations.hard_sigmoid,dtype=tf.float32)])
    return model
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

def create_batch(X,Y,batch_size,prueba=False):
    sub_x=np.zeros((batch_size,224,224,3))
    sub_y=np.zeros((batch_size,24))
    indices=np.random.choice(range(len(Y)),batch_size)
    j=0
    for i in indices:
        im =Image.open(X[i].replace("\"","")+".jpeg")

        im=im.resize((224,224),Image.ANTIALIAS)
        cosa=np.asarray(im)/255
        if len(cosa.shape)<3:
            continue
        if cosa.shape[2]>3:
            continue

        sub_x[j,:,:,:]=cosa
        temp = np.zeros(24)
        temp[Y[i]]=1
        sub_y[j,:]=temp
        j+=1
    #
    # data= tf.data.Dataset.from_tensor_slices((tf.constant(sub_x, dtype=tf.float32), tf.constant(sub_y, dtype=tf.float32))).batch(10)
    # tupla=tf.constant(sub_x,dtype=tf.float32),tf.constant(sub_y,dtype=tf.float32)
    if prueba:
        with open(PATH_TO_BATCH+"prueba","w+b") as fp :
            pickle.dump((sub_x,sub_y),fp)
    else:
        with open(PATH_TO_BATCH,"w+b") as fp :
            pickle.dump((sub_x,sub_y),fp)

def input_fn(prueba=False,batch_size=16):
    """An input_fn to parse 28x28 images from filename using tf.data."""
    # batch_size = params["batch_size"]
    # with tf.io.gfile.GFile(PATH_TO_BATCH, 'r+b') as f:
    #     txt = f.read()
    # source=tf.constant(txt, dtype=tf.float32)
    # print(source)
    if prueba:
        with open(PATH_TO_BATCH+"prueba","r+b") as fp:
            state_batch,q_values = pickle.load(fp)
        state_batch =tf.constant(state_batch,dtype=tf.float32)
        q_values = tf.constant(q_values,dtype=tf.float32)

        prob_dataset = tf.data.Dataset.from_tensor_slices((state_batch,q_values))

        batchd_prob = prob_dataset.batch(batch_size, drop_remainder=True)
        # batchd_prob =batchd_prob.cache()
        return batchd_prob.repeat()
    else:
        with open(PATH_TO_BATCH,"r+b") as fp:
            state_batch, q_values = pickle.load(fp)
        state_batch =tf.constant(state_batch, dtype=tf.float32)
        q_values = tf.constant(q_values, dtype=tf.float32)

        prob_dataset = tf.data.Dataset.from_tensor_slices((state_batch, q_values))

        batchd_prob = prob_dataset.batch(batch_size, drop_remainder=True)
        # batchd_prob =batchd_prob.cache()
        return batchd_prob.repeat()

del df_train[0]
labels =list(map(give_label,df_train))
set_labels =list(set_labels)
labels =list(map(transform_labels,labels))

# for i ,lab in enumerate(set_labels):
#     dict_labels[lab]=i
ids = list(map(add_prefix,list(map(give_id,df_train))))
X_train,X_test,y_train,y_test=train_test_split(ids,labels,test_size=0.1)


with strategy.scope():

    model = crear_modelo()
    optim=keras.optimizers.RMSprop(learning_rate=0.00001,momentum=0.0002)
    metrics=[keras.metrics.CategoricalAccuracy(name="Categorical_accuracy"),
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn')]
    loss=tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optim,loss=loss,metrics=metrics)

for i in range(10000+1):
    create_batch(X_train,y_train,BATCH_SIZE,prueba=False)
    create_batch(X_test,y_test,BATCH_SIZE,prueba=True)

    model.fit(input_fn(),epochs=10,steps_per_epoch=10)
    print("EAVULATION PHASE")
    model.evaluate(input_fn(prueba=True),steps=100)
    if i%100==0:
        model.save("modelo.h5")
        # model.evaluate(x_test,Y_test)



