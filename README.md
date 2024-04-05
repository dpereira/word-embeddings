# Word Embeddings

Experiments with word embeddings.

## Log

Got some interesting results by adding the literature corpus for training:

```
In [32]: sorted(w2v_index.correlation_score("family"), key=lambda k: k[0], reverse=True)[:10]
Out[32]:
[(1.0, 'family'),
 (0.4974841296209473, 'household'),
 (0.4290080389931964, 'families'),
 (0.4173665266276115, 'friends'),
 (0.4084904759534541, 'company'),
 (0.4047620333786069, 'friend'),
 (0.3957563250992789, 'library'),
 (0.3952409729910389, 'sister'),
 (0.3902683895046882, 'acquaintance'),
 (0.38742905588785803, 'tent')]
 ```

 Much better than previous runs. (With word2vec model).

## References

[Word embeddings @ Tensorflow](https://www.tensorflow.org/text/guide/word_embeddings)

[Word2Vec @ Tensorflow](https://www.tensorflow.org/text/tutorials/word2vec)

[Classic literature corpus](https://www.kaggle.com/datasets/mylesoneill/classic-literature-in-ascii)
