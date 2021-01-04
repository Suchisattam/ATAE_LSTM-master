
## Directory Structure ##

```bash
.
├── datasets
│   └── semeval14
├── figure_plots
│   ├── laptop_2way_acc.jpeg
│   ├── laptop_2way_loss.jpeg
│   ├── laptop_3way_acc.jpeg
│   ├── laptop_3way_loss.jpeg
│   ├── restaurant_2way_acc.jpeg
│   ├── restaurant_2way_loss.jpeg
│   ├── restaurant_3way_acc.jpeg
│   └── restaurant_3way_loss.jpeg
├── layers
│   ├── __pycache__
│   ├── attention.py
│   ├── dynamic_rnn.py
│   ├── __init__.py
│   └── squeeze_embedding.py
├── colorize.html
├── atae_lstm.py
├── attention_visualization.py
├── data_utils.py
├── glove.42B.300d.txt(NEEDS TO BE DOWNLOADED SEPERATELY)
├── infer_example.py
└── train.py
```



## DATASET ##

Dataset is in the folder [datasets](datasets/semeval14). It contains the training and testing data for Restaurants and Laptop reviews with respect to different aspects.


## GloVe Vectors ##

The file "glove.42B.300d.txt" contains the Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors) available on [stanfordnlp](https://github.com/stanfordnlp/GloVe) repository. The direct link is given [here](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip).

NOTE : The file size of GloVe vectors is around 5GB when extracted from ZIP.



## Code ##

[data_utils.py](data_utils.py) - Contains the code for preparing the SemEval14 dataset, loading the pretrained word vectors from GloVe and builds the embedding matrix which will be used for the aspect embedding.  


[atae_lstm.py](atae_lstm.py) - The model architecture. This file contains the code for ATAE_LSTM model using inbuilt modules available in PyTorch and a few custom layers like Attention, Dynamic RNN and Squeeze Embedding which are defined in [layers](layers/). 


[train.py](train.py) - The main python file to train Attenion based LSTM with aspect embedding. The default parameters are set to the values as given in the paper by authors.  


[layers](layers/) - This contains [attention.py](layers/attention.py) which contains the code for Attention Mechanism used in the ATAE model in this implementation. [dynamic_rnn.py](layers/dynamic_rnn.py) contains the code for a Dynamic LSTM class able to take in variable length inputs with other properties like Bidirectional LSTM and Dropout too. And [squeeze_embedding.py](layers/squeeze_embedding.py) squeezes sequence embedding length to the longest one in the batch.


[infer_example.py](infer_example.py) - The code for running inference on trained models. Trained models will be saved in state_dict folder. Change the model in this [line](https://github.com/harshshah99/ABSA-PyTorch/blob/master/infer_example.py#L79) to the one you want to run inference with. Also you can change the sentences and aspects you want to test by changing these [lines](https://github.com/harshshah99/ABSA-PyTorch/blob/master/infer_example.py#L9-L39). Change the polarity [here](https://github.com/harshshah99/ABSA-PyTorch/blob/master/infer_example.py#L107) to 2 or 3 as required. 


[attention_visualization.py](attention_visualization.py) - Change the sentences, aspects and attention weights in these [lines](https://github.com/harshshah99/ABSA-PyTorch/blob/master/attention_visualization.py#L20-L54) and run the script. It will create a file called [colorize.html](colorize.html) in the main directory containing Attention Heatmap of the input sentence with respect to the aspect provided. 

## Sample  Attention Weights ##

A few attention weights with respect to sample sentences have been provided [here](https://github.com/harshshah99/ABSA-PyTorch/blob/master/attention_visualization.py#L31-L36). These have been obtained by running [infer_example.py](infer_example.py) which will print Attention weights of provided sentences with aspects in the terminal. 
