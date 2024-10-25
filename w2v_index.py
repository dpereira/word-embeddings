import glob
import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


EMBEDDING_DIM = 128


def remove_encoding_errors(files):
    ok = []
    contents = []
    for f in files:
        with open(f) as stream:
            try:
                contents.append(stream.read())
                ok.append(f)
            except Exception as e:
                print(f"Error for {f}: {e}")

    return ok, contents


#files = list(glob.glob("aclImdb/train/**/*.txt", recursive=True)) + list(glob.glob("aclImdb/test/**/*.txt", recursive=True))
files = list(glob.glob("literature/**/*.txt", recursive=True))
files, documents = remove_encoding_errors(files)


def train():
    global files
    SEED = 42
    AUTOTUNE = tf.data.AUTOTUNE


    sentence = "The wide road shimmered in the hot sun"
    tokens = list(sentence.lower().split())
    print(len(tokens))

    vocab, index = {}, 1  # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    for token in tokens:
      if token not in vocab:
        vocab[token] = index
        index += 1
    vocab_size = len(vocab)
    print(vocab)

    inverse_vocab = {index: token for token, index in vocab.items()}
    print(inverse_vocab)

    example_sequence = [vocab[word] for word in tokens]
    print(example_sequence)

    window_size = 2
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          example_sequence,
          vocabulary_size=vocab_size,
          window_size=window_size,
          negative_samples=0)
    print(len(positive_skip_grams))

    for target, context in positive_skip_grams[:5]:
      print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

    # Get target and context words for one positive skip-gram.
    target_word, context_word = positive_skip_grams[0]

    # Set the number of negative samples per positive context.
    num_ns = 4

    context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
    negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=context_class,  # class that should be sampled as 'positive'
        num_true=1,  # each positive skip-gram has 1 positive context class
        num_sampled=num_ns,  # number of negative context words to sample
        unique=True,  # all the negative samples should be unique
        range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
        seed=SEED,  # seed for reproducibility
        name="negative_sampling"  # name of this operation
    )
    print(negative_sampling_candidates)
    print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])


    # Reduce a dimension so you can use concatenation (in the next step).
    squeezed_context_class = tf.squeeze(context_class, 1)

    # Concatenate a positive context word with negative sampled words.
    context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)

    # Label the first context word as `1` (positive) followed by `num_ns` `0`s (negative).
    label = tf.constant([1] + [0]*num_ns, dtype="int64")
    target = target_word

    print(f"target_index    : {target}")
    print(f"target_word     : {inverse_vocab[target_word]}")
    print(f"context_indices : {context}")
    print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
    print(f"label           : {label}")

    print("target  :", target)
    print("context :", context)
    print("label   :", label)

    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
    print(sampling_table)


    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.
    def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
      # Elements of each training example are appended to these lists.
      targets, contexts, labels = [], [], []

      # Build the sampling table for `vocab_size` tokens.
      sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

      # Iterate over all sequences (sentences) in the dataset.
      for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
              sequence,
              vocabulary_size=vocab_size,
              sampling_table=sampling_table,
              window_size=window_size,
              negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
          context_class = tf.expand_dims(
              tf.constant([context_word], dtype="int64"), 1)
          negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
              true_classes=context_class,
              num_true=1,
              num_sampled=num_ns,
              unique=True,
              range_max=vocab_size,
              seed=seed,
              name="negative_sampling")

          # Build context and label vectors (for one target word)
          context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
          label = tf.constant([1] + [0]*num_ns, dtype="int64")

          # Append each element from the training example to global lists.
          yield target_word, context, label
          #targets.append(target_word)
          #contexts.append(context)
          #labels.append(label)

      #return targets, contexts, labels

    text_ds = tf.data.TextLineDataset(files).filter(lambda x: tf.cast(tf.strings.length(x), bool))


    # Now, create a custom standardization function to lowercase the text and
    # remove punctuation.
    def custom_standardization(input_data):
      lowercase = tf.strings.lower(input_data)
      return tf.strings.regex_replace(lowercase,
                                      '[%s]' % re.escape(string.punctuation), '')


    # Define the vocabulary size and the number of words in a sequence.
    vocab_size = 4096
    sequence_length = 10

    # Use the `TextVectorization` layer to normalize, split, and map strings to
    # integers. Set the `output_sequence_length` length to pad all samples to the
    # same length.
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)


    vectorize_layer.adapt(text_ds.batch(1024))

    # Save the created vocabulary for reference.
    inverse_vocab = vectorize_layer.get_vocabulary()
    print(inverse_vocab[:20])

    # Vectorize the data in text_ds.
    text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()


    sequences = list(text_vector_ds.as_numpy_iterator())
    print(len(sequences))

    for seq in sequences[:5]:
      print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

    data_generator = generate_training_data(
        sequences=sequences,
        window_size=2,
        num_ns=4,
        vocab_size=vocab_size,
        seed=SEED
    )

    targets = None
    contexts = None
    labels = None

    for target, context, label in data_generator:
        if targets is None:
            targets = np.array([target])
        else:
            np.append(targets, target)

        if contexts is None:
            contexts = np.array([context])
        else:
            np.append(contexts, context)

        if labels is None:
            labels = np.array([label])
        else:
            np.append(labels, label)


    #targets = np.array(targets)
    #contexts = np.array(contexts)
    #labels = np.array(labels)

    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)


    class Word2Vec(tf.keras.Model):
      def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                           embedding_dim)

      def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
          target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


    def custom_loss(x_logit, y_true):
          return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

    embedding_dim = EMBEDDING_DIM
    word2vec = Word2Vec(vocab_size, embedding_dim)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")


    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()


    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
      if index == 0:
        continue  # skip 0, it's padding.
      vec = weights[index]
      out_v.write('\t'.join([str(x) for x in vec]) + "\n")
      out_m.write(word + "\n")
    out_v.close()
    out_m.close()

    try:
      from google.colab import files
      files.download('vectors.tsv')
      files.download('metadata.tsv')
    except Exception:
      pass

    return vocab, weights


import pickle, os

VOCAB = "vocab.dump"
WEIGHTS = "weights.dump"
INDEX = "index.dump"

def save(vocab, weights):
    with open(VOCAB, "wb+") as f:
        pickle.dump(vocab, f)
    with open(WEIGHTS, "wb+") as f:
        pickle.dump(weights, f)


def load():
    with open(VOCAB, "rb") as f:
        vocab = pickle.load(f)
    with open(WEIGHTS, "rb") as f:
        weights = pickle.load(f)

    return vocab, weights


if os.path.isfile(VOCAB) and os.path.isfile(WEIGHTS):
    vocab, weights = load()
else:
    vocab, weights = train()
    save(vocab, weights)

index = dict(zip(vocab, weights))


def vectors(words):
    for word in words.split(" "):
        #if word in sw:
        #    continue
        try:
            yield word, index[word]
        except:
            pass


# hack together an "inverted index"
#inverted_index = {}
#
#
#for i, d in enumerate(documents):
#    print(f"Indexing @ {i}: {d[:20]}...")
#    embeddings = vectors(d)
#    for w, e in embeddings:
#        key = tuple(e.tolist())
#        e_index = inverted_index.get(key, [])
#        e_index.append(i)
#        inverted_index[key] = e_index
#

# create ann index
from annoy import AnnoyIndex

aindex = AnnoyIndex(EMBEDDING_DIM, 'angular')
for i, d in enumerate(documents):
    print(f"Indexing @ {i}: {d[:20]}...")
    embeddings = vectors(d)
    for w, e in embeddings:
        aindex.add_item(i, e.tolist())

aindex.build(10)

def asearch(terms, threshold=-1):
    embeddings = vectors(terms)
    scores = {}
    for i, (_, e) in enumerate(embeddings):
        documents, term_scores = aindex.get_nns_by_vector(e, threshold, include_distances=True)
        for d, s in zip(documents, term_scores):
            score = scores.get(d, 0)
            score += s
            scores[d] = score

    return sorted([(d, s) for d, s in scores.items()], key=lambda i: i[1])


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

def c(t, n=10): return sorted(correlation_score(t), key=lambda k: k[0], reverse=True)[:n]
