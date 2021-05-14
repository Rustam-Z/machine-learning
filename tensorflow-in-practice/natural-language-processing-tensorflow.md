# [Natural Language Processing in TensorFlow](https://www.coursera.org/learn/natural-language-processing-tensorflow/home/welcome)

    - Week 1: How to convert the text into number representation, Tokenizer, fit_on_texts, texts_to_sequences, pad_sequences
    - Week 2: Word Embeddings

- Week 1 - [Sentiment in text](#Sentiment-in-text)
- Week 2 - [Word Embeddings](#Word-Embeddings)
- Week 3 - [Sequence models](#Sequence-models)
- Week 4 - [Sequence models and literature](#Sequence-models-and-literature) 

## Sentiment in text
- [Week 1 notebook](notebooks/Course_3_Week_1.ipynb)
- How to load in the texts, pre-process it and set up your data so it can be fed to a neural network.
- https://rishabhmisra.github.io/publications/
- `Tokenizer` is used to tokenize the sentences, `oov_token=<Token>`can be used to encode unknown words
- `fit_on_texts(sentences)` is used to tokenize the list of sentences
    - Output: `{'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}`
- `texts_to_sequences(sentences)` - the method to encode a list of sentences to use those tokens
    - Output: `[[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]`

## Word Embeddings


## Sequence models


## Sequence models and literature
