# Abstractive-Text-Summarization

Text summarization is the process of shortening a corpus of text to a few sentences that together hold the most essential informations of the original contents in the source corpus.Abstractive Text Summarization is the process of ﬁnding the most essential meaning of a text and re-write them in a summary. The resulted summary is an interpretation of the source.

Installation Required
Python 3.5
OpenNmt library
Pytorch framework
flask
Google Colab

## Quickstart

## Step 1: Preprocess the data

In this model, we need to preprocess the dataset such that the same set of words are used for the input and output sentences.The dataset used is the CNN/dailymail dataset which is a collection of the articles mostly news, interviews that have been published on the two popular websites CNN.com and dailymail.com.

' 
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt 
    -valid_tgt data/tgt-val.txt -save_data data/demo -share_vocab -dynamic_dict -src_vocab_size 50000
'

The data consists of parallel source (src) and target (tgt) data containing one example per line with tokens separated by a space:

- src-train.txt
- tgt-train.txt
- src-val.txt
- tgt-val.txt

## Step 2: Train the model
The basic command would be:

python train.py -data data/demo -save_model demo_model -share_embeddings
