import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import bert
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
from model import DeepConvolutionalEmbedding

df = pd.read_csv("./Dataset/train/stories.csv")
df['Emotion'] = df.loc[:, 'annotate1'].str.split(':')


def extraction(emotion):
    emotion = emotion[0]
    return emotion


df['Emotion'] = df['Emotion'].apply(extraction)

columns = [1, 4]
df = df.iloc[:, columns]
df.head(13)

print(df.Emotion.value_counts())

Y = df['Emotion']
plt.figure(figsize=(12, 6))
plt.xticks(rotation=90)
sns.countplot(x=df['Emotion'])

df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su-', 'Su')
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su+', 'Su')

FullTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def encode_sentence(sentence):
    return ["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"]


df['encoded_sentence'] = df['sent'].apply(encode_sentence)


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


label_encoder = preprocessing.LabelEncoder()
df['Emotion'] = label_encoder.fit_transform(df['Emotion'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
le_mapping = {'Emotion': le_name_mapping}
print(df['Emotion'].values)
print(le_mapping)

y = df['Emotion'].values

data_with_length = [[sentence, y[i], len(sentence)]
                    for i, sentence in enumerate(df['encoded_sentence'])]

data_with_length.sort(key=lambda x: x[2])

sorted_sentence_emotions = [([get_ids(sentence_emotion[0]),
                              get_mask(sentence_emotion[0]),
                              get_segments(sentence_emotion[0])],
                             sentence_emotion[1])
                            for sentence_emotion in data_with_length if sentence_emotion[2] > 7]

dataset = tf.data.Dataset.from_generator(lambda: sorted_sentence_emotions,
                                         output_types=(tf.int32, tf.int32))

# print(next(iter(dataset)))

BATCH_SIZE = 16
all_batched = dataset.padded_batch(BATCH_SIZE,
                                   padded_shapes=((3, None), ()),
                                   padding_values=(0, 0))
print(all_batched)

total_batches = math.ceil(len(sorted_sentence_emotions) / BATCH_SIZE)
test_batches = total_batches // 10
all_batched.shuffle(total_batches)
test_dataset = all_batched.take(test_batches)
train_dataset = all_batched.skip(test_batches)

print(total_batches)

print(test_batches)

NB_FILTERS = 50
FFN_UNITS = 128
NB_CLASSES = 7
DROPOUT_RATE = 0.8
NB_EPOCHS = 1

Dcnn = DeepConvolutionalEmbedding(nb_filters=NB_FILTERS,
                                  FFN_units=FFN_UNITS,
                                  nb_classes=NB_CLASSES,
                                  dropout_rate=DROPOUT_RATE)

Dcnn.compile(loss="sparse_categorical_crossentropy",
             optimizer="adam",
             metrics=["sparse_categorical_accuracy"])

checkpoint_path = "./weight/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)

history = Dcnn.fit(train_dataset,
                   epochs=NB_EPOCHS,
                   callbacks=[cp_callback])

print(Dcnn.summary())


results = Dcnn.evaluate(test_dataset)
print(results)
