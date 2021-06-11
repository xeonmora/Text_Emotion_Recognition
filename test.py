import bert
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import f1_score, precision_score, recall_score

from model import DeepConvolutionalEmbedding

NB_FILTERS = 50
FFN_UNITS = 128
NB_CLASSES = 7

DROPOUT_RATE = 0.8

Dcnn = DeepConvolutionalEmbedding(nb_filters=NB_FILTERS,
                                  FFN_units=FFN_UNITS,
                                  nb_classes=NB_CLASSES,
                                  dropout_rate=DROPOUT_RATE)

Dcnn.compile(loss="sparse_categorical_crossentropy",
             optimizer="adam",
             metrics=["sparse_categorical_accuracy"])

checkpoint_path = "./weight/cp.ckpt"
Dcnn.load_weights(checkpoint_path)

FullTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def encode_sentence(sentence):
    return ["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"]


def get_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)


def get_mask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)


def get_segments(tokens):
    seg_ids = []
    current_seg_id = 0
    for token in tokens:
        seg_ids.append(current_seg_id)
        if token == "[SEP]":
            current_seg_id = 1 - current_seg_id
    return seg_ids


def get_prediction(sentence):
    tokens = encode_sentence(sentence)
    input_ids = get_ids(tokens)
    input_mask = get_mask(tokens)
    segment_ids = get_segments(tokens)
    inputs = tf.stack(
        [tf.cast(input_ids, dtype=tf.int32),
         tf.cast(input_mask, dtype=tf.int32),
         tf.cast(segment_ids, dtype=tf.int32)],
        axis=0)
    inputs = tf.expand_dims(inputs, 0)  # simulates a batch
    output = Dcnn(inputs, training=False)

    print("Story sentence is ", sentence)
    print('probability ', output)
    emotion = ['A', 'D', 'F', 'H', 'N', 'Sa', 'Su'][np.argmax(output)]
    print('Predicted emotion is ', emotion)
    return emotion


"""
{'Emotion': {'A': 0, 'D': 1, 'F': 2, 'H': 3, 'N': 4, 'Sa': 5, 'Su': 6}}
"""

test_df = pd.read_csv("./Dataset/test/frog_prince.csv")
print(test_df.head())

test_df['Emotion'] = test_df['annotate1']


def extraction(emotion):
    emotion = emotion[0]
    return emotion


test_df['Emotion'] = test_df['Emotion'].apply(extraction)
test_df['prediction'] = test_df['sent'].apply(get_prediction)

test_y = test_df['Emotion']
test_y_pred = test_df['prediction']

precisionScore = precision_score(test_y, test_y_pred, average='weighted')
print(precisionScore)

f1Score = f1_score(test_y, test_y_pred, average='weighted')
print(f1Score)

recallScore = recall_score(test_y, test_y_pred, average='weighted')
print(recallScore)

test_df.to_csv('./Dataset/test/speech_input.csv')