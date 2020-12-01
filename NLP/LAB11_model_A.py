import pandas as pd
import glob
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import re
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout, InputLayer, Bidirectional, TimeDistributed, Activation, Embedding
from keras.optimizers import Adam
import seaborn as sn

f = glob.glob("swda/sw*/sw*.csv")
frames = []
for i in range(0, len(f)):
    frames.append(pd.read_csv(f[i]))

result = pd.concat(frames, ignore_index=True)

print("Number of converations in the dataset:", len(result))

"""The dataset has many different features, we are only using act_tag and text for this training."""

reduced_df = result[['act_tag', 'text']]

"""Reduce down the number of tags to 43 - converting the combined tags to their generic classes:"""


# Imported from "https://github.com/cgpotts/swda"
# Convert the combination tags to the generic 43 tags


def damsl_act_tag(input):
    """
    Seeks to duplicate the tag simplification described at the
    Coders' Manual: http://www.stanford.edu/~jurafsky/ws97/manual.august1.html
    """
    d_tags = []
    tags = re.split(r"\s*[,;]\s*", input)
    for tag in tags:
        if tag in ('qy^d', 'qw^d', 'b^m'):
            pass
        elif tag == 'nn^e':
            tag = 'ng'
        elif tag == 'ny^e':
            tag = 'na'
        else:
            tag = re.sub(r'(.)\^.*', r'\1', tag)
            tag = re.sub(r'[\(\)@*]', '', tag)
            if tag in ('qr', 'qy'):
                tag = 'qy'
            elif tag in ('fe', 'ba'):
                tag = 'ba'
            elif tag in ('oo', 'co', 'cc'):
                tag = 'oo_co_cc'
            elif tag in ('fx', 'sv'):
                tag = 'sv'
            elif tag in ('aap', 'am'):
                tag = 'aap_am'
            elif tag in ('arp', 'nd'):
                tag = 'arp_nd'
            elif tag in ('fo', 'o', 'fw', '"', 'by', 'bc'):
                tag = 'fo_o_fw_"_by_bc'
        d_tags.append(tag)
    # Dan J says (p.c.) that it makes sense to take the first;
    # there are only a handful of examples with 2 tags here.
    return d_tags[0]


reduced_df["act_tag"] = reduced_df["act_tag"].apply(lambda x: damsl_act_tag(x))

"""There are 43 tags in this dataset. Some of the tags are Yes-No-Question('qy'), Statement-non-opinion('sd') and Statement-opinion('sv'). Tags information can be found here http://compprag.christopherpotts.net/swda.html#tags.

To get unique tags:
"""

unique_tags = set()
for tag in reduced_df['act_tag']:
    unique_tags.add(tag)

one_hot_encoding_dic = pd.get_dummies(list(unique_tags))

tags_encoding = []
for i in range(0, len(reduced_df)):
    tags_encoding.append(one_hot_encoding_dic[reduced_df['act_tag'].iloc[i]])

"""The tags are one hot encoded.

To create sentence embeddings:
"""

sentences = []
for i in range(0, len(reduced_df)):
    sentences.append(reduced_df['text'].iloc[i].split(" "))

wordvectors = {}
index = 1
for s in sentences:
    for w in s:
        if w not in wordvectors:
            wordvectors[w] = index
            index += 1

# Max length of 137
MAX_LENGTH = len(max(sentences, key=len))

sentence_embeddings = []
for s in sentences:
    sentence_emb = []
    for w in s:
        sentence_emb.append(wordvectors[w])
    sentence_embeddings.append(sentence_emb)

"""Then we split the dataset into test and train."""
X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, np.array(tags_encoding))

"""And pad the sentences with zero to make all sentences of equal length."""

MAX_LENGTH = 137

train_sentences_X = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post')

# Split Train into Train and Validation - about 10% into validation - In order to validate the model as it is training

train_input = train_sentences_X[:140000]
val_input = train_sentences_X[140000:]

train_labels = y_train[:140000]
val_labels = y_train[140000:]

"""# Model 1 - 

The first approach we'll try is to treat DA tagging as a standard multi-class text classification task, in the way you've done before with sentiment analysis and other tasks. Each utterance will be treated independently as a text to be classified with its DA tag label. This model has an architecture of:

- Embedding  
- BLSTM  
- Fully Connected Layer
- Softmax Activation

The model architecture is as follows: Embedding Layer (to generate word embeddings) Next layer Bidirectional LSTM. Feed forward layer with number of neurons = number of tags. Softmax activation to get the probabilities.
"""

VOCAB_SIZE = len(wordvectors)  # 43,731
MAX_LENGTH = len(max(sentences, key=len))
EMBED_SIZE = 100  # arbitary
HIDDEN_SIZE = len(unique_tags)

# Building the network

# Include 2 BLSTM layers, in order to capture both the forward and backward hidden states

# Embedding layer
# Bidirectional 1
# Bidirectional 2
# Dense layer
# Activation
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_LENGTH))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
model.add(Bidirectional(LSTM(HIDDEN_SIZE)))
model.add(Dense(HIDDEN_SIZE))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model - using validation 
history = model.fit(train_input, train_labels,
                    epochs=3,
                    batch_size=100,
                    validation_data=(val_input, val_labels),
                    verbose=1)

score = model.evaluate(test_sentences_X, y_test, batch_size=100)

print("Overall Accuracy:", score[1] * 100)

"""## Evaluation


The overall accuracy is 67%, an effective accuracy for this task.

In addition to overall accuracy, you need to look at the accuracy of some minority classes. Signal-non-understanding ('br') is a good indicator of "other-repair" or cases in which the other conversational participant attempts to repair the speaker's error. Summarize/reformulate ('bf') has been used in dialogue summarization. Report the accuracy for these classes and some frequent errors you notice the system makes in predicting them. What do you think the reasons are？

## Minority Classes
"""

# Generate predictions for the test data
label_pred = model.predict(test_sentences_X, batch_size=100)

# Build the confusion matrix off these predictions
# one_hot_encoding_dic['bf'].argmax()


# create confusion matrix
conf_matrix = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), label_pred.argmax(axis=1))
labels = list(unique_tags)
df_cm = pd.DataFrame(conf_matrix, labels, labels)
plt.figure(figsize=(43, 43))
# sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, fmt='d', annot=True)  # font size
#plt.show()
plt.savefig("modelA_confusion_matrix.png")


tag_labels = list(unique_tags)
br_index = tag_labels.index('br')
bf_index = tag_labels.index('bf')
sum_of_true_labels = np.sum(conf_matrix, axis=1)  # row has true labels
TP_br = conf_matrix[br_index, br_index]
br_total = sum_of_true_labels[br_index]
print("correctly identified %d out of %d ,accuracy for br tag: %f" % (
TP_br, br_total, (TP_br / sum_of_true_labels[br_index] * 100)))

TP_bf = conf_matrix[bf_index, bf_index]
bf_total = sum_of_true_labels[bf_index]
print("correctly identified %d out of %d ,accuracy for bf tag: %f \n" % (
TP_bf, bf_total, (TP_bf / sum_of_true_labels[bf_index] * 100)))

