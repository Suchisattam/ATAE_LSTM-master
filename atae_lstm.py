from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn

from layers.squeeze_embedding import SqueezeEmbedding


class ATAE_LSTM(nn.Module):
	def __init__(self, embedding_matrix, opt):
		super(ATAE_LSTM, self).__init__()
		self.opt = opt #Parser arguments assigned
		self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float)) #Initialized the word embedding matrix with the GloVe word embeddings for fine tuning
		self.squeeze_embedding = SqueezeEmbedding() #Instance of Squeeze Embedding Class from layers.squeeze_embedding created
		self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True) #Instance of DynamicLSTM Class from layers.dynamic_rnn created parameters passed, input_size=opt.embed_dim*2, hidden_size=opt.hidden_dim
		self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear') #Instance of NoQueryAttention Class from layers.attention created parameters passes, embed_dim=opt.hidden_dim+opt.embed_dim and score function is 'bi_linear'
		self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim) #A linear layer is created with in_features=opt.hidden_dim, out_features=opt.polarities_dim

	def forward(self, inputs):
		"""
		Intialising text_raw_indices and aspect_indices using glove vector->Copying the aspect embedding value=max length of the sentence->Concatenated sentence and aspect embeddings
		feeded to LSTM network->Output of LSTM network again concatenated with Aspect embeddings feeded to Attention network->Matrix Multiplication of Output score of Attention 
		network with h squeezed in dimension=1 and feeded to Linear Layer-> Output of Linear Layer is Final Output
		"""
		text_raw_indices, aspect_indices = inputs[0], inputs[1] #text_raw_indices are sentences with aspects included, aspect_indices are the aspects for which polarity of sentences are to be calculated
		print("text_raw_indices  SHAPE : \n\n\n\n\n\n",text_raw_indices.shape )
		print("aspect_indices  SHAPE : \n\n\n\n\n\n",aspect_indices.shape )

		#text_raw_indices row wise sum is calculated and max value is assigned to x_len_max
		x_len = torch.sum(text_raw_indices != 0, dim=-1)
		x_len_max = torch.max(x_len)

		#aspect_len is assigned value as row wise sum of aspect_indices
		aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).to(self.opt.device)


		x = self.embed(text_raw_indices) # x is initialised with value of glove embeddings of text_raw_indices
		x = self.squeeze_embedding(x, x_len) #Squeeze sequence embedding length to the longest one in the batch
		aspect = self.embed(aspect_indices) # aspect is initialised with value of glove embeddings of aspect_indices
		
		#aspect shape is matched with max length of the sentence, by copying the aspect embedding value=max length of the sentence
		aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
		aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)

		x = torch.cat((aspect, x), dim=-1) #aspect embedding and text_raw_indices embedding are concatenated to be given as input to LSTM network (concatenated tensor assigned to x)


		h, (_, _) = self.lstm(x, x_len) # x and sentence length are given as input to LSTM network and output from each LSTM unit is assigned to hidden layer h 
		ha = torch.cat((h, aspect), dim=-1) # h is again concatenated with aspect embedding to be given as input to Attention network (concatenated tensor assigned to ha)
		_, score = self.attention(ha) # ha is given as input to Attention network and value returned is assigned to score
		print("Attention weights  : " , score)

		output = torch.squeeze(torch.bmm(score, h), dim=1) #Batch matrix-matrix product of matrices stored in score and h is performed and the output is squeezed in dimension=1 (squeezed tensor assigned to output)

		out = self.dense(output) #output, is an input to linear layer whose output gives polarity wrt given aspect, output dimensions are specified by opt.polarities_dim (assigned to out)
		return out
