import torch
import torch.nn.functional as F
import argparse

from data_utils import build_tokenizer, build_embedding_matrix
from atae_lstm import ATAE_LSTM


rest_raw_texts = [
    'The menu had a lot of varieties', 
    'the fajita we tried was tasteless and burnt and the sauce was way too sweet.',
    'The service is terrible but the food was good' ,
    'The service is terrible but the food was good', 
    'they have one of the fastest delivery times in the city', 
    'The staff of the restaurant were really polite and helpful', 
    'The spice level of food was moderate', 
    'The ambience of restaurant was good, perfectly adjusted light set the mood', 
    'The restaurant maintained hygiene standards, keeping the surroundings perfectly clean', 
    'Though the waiter was very rude and disobidiend their food was amazing and tasty',
    'Though the waiter was very rude and disobidiend their food was amazing and tasty', 
    'The food at restaurant was expensive but place was very spacious']

rest_aspect_tokens = ['cheese' , 'food', 'food' , 'service' , 'service' , 'staff' , 'food' , 'ambience' , 'hygiene' , 'waiter' , 'food', 'price' ]


laptop_raw_texts = [
    'The battery life is above average but the laptop heats really quick' , 
    'The laptop is attractive for youngsters because of the features it offers' , 
    'The system has delayed response might be due bugs in operating system' , 
    'The service center people are very polite and helpful, but the laptop is overpriced' , 
    'Spending 2000Rs extra for webcam was totally justified' , 
    'Though my system is working fine even after 5 years of purchasing it, sometimes software gets unresponsive', 
    'Though my system is working fine even after 5 years of purchasing it, sometimes software gets unresponsive', 
    'Laptop has 1 TB hard drive for storage and 8 GB RAM with 6 GB nvidia graphics card', 
    'Laptop comes with 1 year warranty which can be extended upto 3 years within offer period' , 
    'Laptop loads the operating system very fast maybe due to presence of 256 Solid State drive' , 
    'Laptop has High Definition display which give amazing real life experience while watching movies, but the webcam picture quality is very bad'] 

laptop_aspect_tokens=['battery' , 'features' , 'system' , 'service center' , 'webcam' , 'software', 'system' , 'hard drive' , 'warranty' , 'speed' , 'display']


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
            word2idx=self.tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_texts,aspect_tokens):
        context_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip()) for raw_text in raw_texts]
        aspect_seqs = [self.tokenizer.text_to_sequence(asp.lower().strip()) for asp in aspect_tokens]
        context_indices = torch.tensor(context_seqs, dtype=torch.int64).to(self.opt.device)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64).to(self.opt.device)
        print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE  ",aspect_indices)

        t_inputs = [context_indices, aspect_indices]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs


if __name__ == '__main__':
    model_classes = {'atae_lstm': ATAE_LSTM}
    # set your trained models here
    model_state_dict_paths = { 'atae_lstm': 'state_dict/atae_lstm_laptop_val_acc0.7022'}


    class Option(object): pass
    opt = Option()
    opt.model_name = 'atae_lstm'
    opt.model_class = model_classes[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]

    dataset_files_dict = {
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }

    opt.dataset = opt.state_dict_path.split('_')[3]
    
    opt.dataset_file = dataset_files_dict[opt.dataset]

    
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 80
    opt.polarities_dim = 3 #change as required
    opt.hops = 10
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if(opt.state_dict_path.split('_')[3] == 'restaurant'):
        raw_texts = rest_raw_texts
        aspect_tokens = rest_aspect_tokens
    else:
        raw_texts = laptop_raw_texts
        aspect_tokens = laptop_aspect_tokens

    inf = Inferer(opt)
    t_probs = inf.evaluate(raw_texts,aspect_tokens)
    if(opt.polarities_dim==3):
        print(t_probs.argmax(axis=-1) - 1)
        for sent in range(0,len(raw_texts)):
            print(raw_texts[sent] , '------------' ,aspect_tokens[sent],'---------',(t_probs.argmax(axis=-1) - 1)[sent])
    else:
        print(t_probs.argmax(axis=-1))
        for sent in range(0,len(raw_texts)):
            print(raw_texts[sent] , '------------',aspect_tokens[sent],'---------',(t_probs.argmax(axis=-1))[sent])

