# Abstractive-Text-Summarization

Text summarization is the process of shortening a corpus of text to a few sentences that together hold the most essential informations of the original contents in the source corpus.Abstractive Text Summarization is the process of ﬁnding the most essential meaning of a text and re-write them in a summary. The resulted summary is an interpretation of the source.

### Installation Required
- Python 3.5 or higher
- OpenNMT library
- Pytorch framework
- flask
- Google Colab

## Quickstart

To run the files all the files have to be uploaded on the Google Colab or any other high end gpu device.

## Step 1: Preprocess the data

In this model, we need to preprocess the dataset such that the same set of words are used for the input and output sentences.The dataset used is the CNN/dailymail dataset which is a collection of the articles mostly news, interviews that have been published on the two popular websites CNN.com and dailymail.com.The dataset can be found [here]( https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz).

The command for preprocessing the data is given below

    'python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo -share_vocab -dynamic_dict'

The data consists of parallel source (src) and target (tgt) data containing one example per line with tokens separated by a space:

- src-train.txt
- tgt-train.txt
- src-val.txt
- tgt-val.txt

After running the preprocessing, the following files are generated:

- demo.train.pt:-PyTorch file containing training data
- demo.valid.pt:-PyTorch file containing validation data
- demo.vocab.pt:-PyTorch file containing vocabulary data

## Step 2: Train the model

The command to train the model is given by:

    'python train.py -data data/demo -save_model transformer_model -share_embeddings'

The above command would train the whole data on the transformer architecture and will consume a lot of processing and time.This command will give a pretrained file which is our transformer model and from now we will use this file as our model for the purpose of evaluation.

## Step 3: Summarize
The command to generate the summary from model we run the following command

    'python translate.py -model transformer_model_epochX_PPL.pt -src data/src-test.txt -o output_pred.txt -beam_size 10 -share_vocab'

Now we have a model which you can use to predict on new data. This will output predictions into pred.txt.

## Step 4: Evaluate with ROUGE

The main evaluation metrics for summarization used is the ROGUE method which compares the system summary with the reference summary.
we have used the file rouge.py for ROGUE evaluation.

The command for running the rogue file is as follows

    'python rogue.py -s output_pred.txt -t data/test.txt.tgt'

## Step 5: Connecting to the UI

Now as the model training is done we have to connect it to the User Interface.We have used flask micro-web framework for hosting our web page.The html and css files are in the static and template folder respectively.

We have to run the server.py file as it will allow us use the pretrained file for predicting any new summary from the article entered in thr User Interface.

    'python server.py'
    
Now to get the user interface to be hosted on the localhost we have to run the given command
    
    'python app.py'
    
This will get the User Interface running and then the user can input a article and the smmary will be generated and will be shown to the ouput of the user Input.

