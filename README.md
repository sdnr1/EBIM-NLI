# EBIM-NLI
This is an implementation of Enhanced BiLSTM Inference Model for Natural Language Inference in Keras. The model is based on a paper by Chen et al. Link : [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf)

Dataset used is [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/). The model uses pre-trained word vectors, [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/).

### Usage ###
Keras is needed to train the model, and test it. `NLI.ipynb` needs to be executed for training and testing the model.

Once the model is trained, the following files are generated : 
```
1. tokenizer.pickle - tokenizes sentences
2. embeddings.npy - Word embeddings based on the GloVe model.
3. NLI.h5 - trained weights for the EBIM model
```
These files are used by `app.py` for predicting the class of a given input.
