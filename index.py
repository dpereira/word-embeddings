import io
import os
import re
import shutil
import string
import tensorflow as tf
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
sw = stopwords.words('english')


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])

model.summary()

weights = model.get_layer('embedding').get_weights()[0]
vocab = [v for v in vectorize_layer.get_vocabulary() if v not in sw]

index = dict(zip(vocab, weights))

def vectors(words):
    for word in words.split(" "):
        if word in sw:
            continue
        try:
            yield word, index[word]
        except:
            pass


# hack together an "inverted index"
documents = []
inverted_index = {}

counter = 0
for i, t in enumerate(train_ds):
    for j, s in enumerate(t[0]):
        d = s.numpy().decode()
        print(f"Indexing @ {counter} {i, j}: {d[:20]}...")
        embeddings = vectors(d)
        documents.append(d)
        for w, e in embeddings:
            key = tuple(e.tolist())
            e_index = inverted_index.get(key, [])
            e_index.append(counter)
            inverted_index[key] = e_index
        counter += 1

#import sys
#sys.exit(0)
#
#for i, document in enumerate(train_ds):
#    embeddings = vectors(str(document[0]))
#    documents.append(document[0])
#    for w, e in embeddings:
#        key = tuple(e.tolist())
#        e_index = inverted_index.get(key, [])
#        e_index.append(i)
#        inverted_index[key] = e_index


lookup = {tuple(e.tolist()): w for w, e in index.items()}
def search(terms, threshold=.3):
    import numpy
    termvector = terms.split()
    embeddings = vectors(terms)
    scores = {}
    for i, (_, e) in enumerate(embeddings):
        for ie, documents in inverted_index.items():
            c = numpy.corrcoef([e, ie])[0][1]
            if abs(c) < threshold:
                continue
            for d in documents:
                score, explanation = scores.get(d, (0, []))
                explanation.append((termvector[i], lookup[ie], c))
                score += c
                scores[d] = score, explanation

    return sorted([(d, s[0], sorted(s[1], key=lambda c: c[2])) for d, s in scores.items()], key=lambda k: k[1], reverse=True)

def correlation_score(term):
    import numpy
    t, embedding = list(vectors(term))[0]

    corrs = []

    for w, e in index.items():
        c = numpy.corrcoef([embedding, e])[0][1]
        corrs.append((c, w))

    return corrs
