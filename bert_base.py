import numpy as np
import tensorflow.keras as keras
from official.nlp.bert.bert_models import *
from official.nlp.bert.tokenization import FullTokenizer


# ESTO SE NECESITA DEBIDO A QUE  BERT EN TEORIA PUEDE ES RESIVIR 2 FRASES PERO NOSOTROS SOLO QUEEREMOS UNA, ADEMÁS ESTA FUINCIÓN TOKENIZA Y CREA LAS 3 COSAS QUE RESIVE BERT
#INPUT_INDS:  los ID que el tokenizer asigna a cada uno de las palabras dentro del vocabulario.
#INPUT_MASK: QUe indices quiero predecir (1 para los que quiero predecir y 0 para los que no)
#SGEMENT_ID: A CUAL DE LSA DOS SECUENCIAS PERTENEZZCO (0 o 1)

def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len - len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return np.array(input_ids),np.array(input_mask), np.array(segment_ids)

#lo mismo de arriba pero para un batch
def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids
natural_questions_dataset_path ="D:\datsets_tesis\Kaggle_competition\Tensorflow_Q_and_A_competition/"

tf.enable_eager_execution()
url= "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
max_seq_length = 128  # Your choice here.

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer(url,
                            trainable=True,name="Bert_model")
# modelo = hub.Module(url)
# cosa1,cosa2=modelo(["god fucking dammit nigga"])
frase="god fucking dammit nigga"
print(tf.executing_eagerly())
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
model=keras.Model(inputs=[input_word_ids,input_mask,segment_ids], outputs=[pooled_output,sequence_output])
model.build(input_shape=(None,None))
model.summary()
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
print(do_lower_case)
tokenizer = FullTokenizer(vocab_file, do_lower_case)
input_word_ids,input_mask,segment_ids =convert_sentence_to_features(frase,tokenizer,max_seq_length)
bert_inputs = dict(
    input_ids=input_word_ids,
    input_mask=input_mask,
    segment_ids=segment_ids)
# input_word_ids =tf.cast(input_word_ids,dtype=tf.int32)
# input_mask = tf.cast(input_mask,dtype=tf.int32)
# segment_ids = tf.cast(segment_ids)
#para bert slaen 2 cosas un vector de 1 por 1024 que representa toda la frase, o un bector de (MAX_LENGTH,1024) que representa cada toquen de la secuencia (cabe reslatar que así la frease sea más corta va a salir 128 columnas)

cosa1,cosa2=model([input_word_ids.reshape(1,-1)[:4], input_mask.reshape(1,-1)[:4], segment_ids.reshape(-1,1)][:4])
print(cosa1.numpy().shape)
print(cosa2.numpy().shape)
