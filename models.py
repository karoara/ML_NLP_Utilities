# MODELS ----------------------------------------------------------------------
# 
# Useful model abstractions (Pytorch).


import torch
from train_utils import write
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TRANSFORMATION (LINEAR/NONLINEAR) -------------------------------------------
#
# Expects vector input, returns vector output after transformation
# 
# - applies xavier initialization to weights
# - allows specifying dimensionality of input/output
# - allows specifying whether or not to apply sigmoid nonlinearity

class TRANS(torch.nn.Module):
  
  def __init__(self, in_d, out_d, non_l):
    super(TRANS, self).__init__()
    self.non_l = non_l
    self.linear = torch.nn.Linear(in_features=in_d, out_features=out_d)
    for param in self.linear.parameters():
      if len(param.size()) > 1: torch.nn.init.xavier_normal_(param)
      else: param.data.fill_(0)
  
  def forward(self, input):
    sigmoid = torch.nn.Sigmoid()
    output = self.linear.forward(input)
    if self.non_l == False: return output
    else: return sigmoid(output)


# 1 LAYER NET -----------------------------------------------------------------
#
# Basic 1-hidden-layer neural net. Expects vector input, returns vector output.
# 
# - Allows specifying input/output dimensionality
# - Hidden layer defaults to input dimensionality
# - Applies Xavier initialization to weights

class NET(torch.nn.Module):
  
  def __init__(self, in_d, out_d):
    super(NET, self).__init__()
    self.l1 = torch.nn.Linear(in_features=in_d, out_features=in_d)
    self.l2 = torch.nn.Linear(in_features=in_d, out_features=out_d)
    for param in list(self.l1.parameters()) + list(self.l2.parameters()):
      if len(param.size()) > 1: torch.nn.init.xavier_normal_(param)
      else: param.data.fill_(0)
  
  def forward(self, inputs):
    relu, sigmoid = torch.nn.LeakyReLU(), torch.nn.Sigmoid()
    h = relu(self.l1.forward(inputs))
    return sigmoid(self.l2.forward(h))


# VARIABLE FEEDFORWARD NET ----------------------------------------------------
# 
# Feedforward neural net whose architecture the user can specify, in the form of
# a list (the arch_spec argument). For example: setting arch_spec to [100, 50, 10]
# will create a network with a 100-dimensional input layer, a 50-dimensional
# hidden layer, and a 10-dimensional output layer. Expects vector input, returns 
# vector output. 
# 
# - applies Xavier initialization to weights

class SPEC_NET(torch.nn.Module):
  
  def __init__(self, arch_spec):
    super(SPEC_NET, self).__init__()
    self.params = torch.nn.ParameterList()
    for i in range(len(arch_spec)-1):
      self.params.append(torch.nn.Parameter(torch.Tensor(arch_spec[i+1], arch_spec[i]).float()))
      self.params.append(torch.nn.Parameter(torch.zeros(arch_spec[i+1]).float()))
    for param in [self.params[i] for i in range(len(self.params)) if (i%2 == 0)]:
      torch.nn.init.xavier_normal_(param)

  def forward(self, sentence):
    logistic, relu = torch.nn.Sigmoid(), torch.nn.ReLU()
    for i in range(int((len(self.params)/2))-1): 
      sentence = relu(torch.matmul(self.params[2*i], sentence) + self.params[2*i+1])
    return logistic(torch.matmul(self.params[-2], sentence) + self.params[-1])


# LSTM MODULE -----------------------------------------------------------------
# 
# Expects packed sequence of vectors as input, returns either final hidden state
# or hidden states at all timesteps (based on argument) as output.
# 
# - Xavier initialization applied to weights
# - Allows specifying layers, whether uni/bi-directional, dropout
# - Initial hidden/cell state values set up as parameters

class LSTM(torch.nn.Module):
  
  # initialization method
  def __init__(self, in_d, h_d, layers=1, dropout=0.0, bi=False, all_states=False):
    super(LSTM, self).__init__()
    
    # of states returned, bidirectionality, hidden/cell states, inner LSTM
    self.all_states = all_states
    self.out_vects = 2 if bi else 1
    self.h_init = torch.nn.Parameter(torch.Tensor(layers*self.out_vects, 1, h_d).float())
    self.c_init = torch.nn.Parameter(torch.Tensor(layers*self.out_vects, 1, h_d).float())
    self.lstm = torch.nn.LSTM(input_size=in_d, hidden_size=h_d, \
                              num_layers=layers, dropout=dropout, bidirectional=bi)
    
    # xavier initialization for parameters
    for param in self.lstm.parameters(): 
      if len(param.size()) > 1: torch.nn.init.xavier_normal_(param)
    for param in [self.h_init, self.c_init]: param.data.normal_()
  
  # forward propagation
  def forward(self, inputs, b_size):
    
    # set initial states for current batch size
    h_0 = self.h_init.repeat(1, b_size, 1).to(device)
    c_0 = self.c_init.repeat(1, b_size, 1).to(device)
    
    # compute output
    h_all, (h_final, _) = self.lstm.forward(inputs, (h_0, c_0))
    if self.lang: final_output = h_all
    else: # if binary classifier
      if self.out_vects == 2: final_output = torch.cat((h_final[-2], h_final[-1]), 1) 
      else: final_output = h_final[-1]
    
    return final_output


# BINARY CLASSIFIER -----------------------------------------------------------
# 
# Neural model that performs binary classification for language data (for stuff
# like sentiment analysis, etc). Expects a set of vectors (words in a sentence) 
# as input, returns a label (0/1) as output. Consists of an LSTM & classfn layer. 
# 
# - allows specifying # of layers, uni/bi-directionality, dropout for the LSTM
# - allows specifying dimensionality of input vectors
# - applies Xavier initialization to weights. 

class BINARY_CLASSIFIER(torch.nn.Module):
  
  # initialization method
  def __init__(self, in_d, h_d, layers, dropout, bi):
    super(BINARY_CLASSIFIER, self).__init__()
    
    # modules
    out_vects = 2 if bi else 1
    self.lstm = LSTM(in_d, h_d, layers, dropout, bi)
    self.classifier = TRANS(in_d=h_d*out_vects, out_d=1, non_l=True)
  
  # forward propagation
  def forward(self, inputs, targets, lengths):
    
    # set up batch in pytorch
    b_size = len(inputs)
    inputs, indices = batch_setup(inputs, lengths)
    targets = torch.from_numpy(targets).float()
    targets = targets[indices][:,None]
    
    # data --> GPU
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # forward computation
    inputs_proc = self.lstm.forward(inputs, b_size)
    outputs = self.classifier.forward(inputs_proc)
    
    return outputs, targets


# LANGUAGE MODEL --------------------------------------------------------------
# 
# Neural model that performs language modeling (predicts the next word, given
# previous words, for every word in a sequence). Expects a series of indices
# (into corresponding word vectors) as input, and returns a set of scores
# (over subsequent words) for words/indices at every timestep as output. 
# 
# - allows specifying number of layers, dropout
# - applies Xavier initialization to weights
# - allows specifying pretrained vectors for linear layer that produces word scores

class LANGUAGE_MODEL(torch.nn.Module):
  
  # initialization method
  def __init__(self, in_d, h_d, layers, dropout,
               vocab_size, vectors, output_layer=None, output_learning=True):
    super(LANGUAGE_MODEL, self).__init__()
    
    # standard modules
    self.embeddings = vectors
    self.lstm = LSTM(in_d, h_d, layers, dropout, bi=False, all_states=True)
    self.output_l = TRANS(h_d, vocab_size)

    # is provided, initialize output layer to given values, set learning
    if output_layer != None:
      weights, bias = list(self.output_l.parameters())
      weights.data = torch.from_numpy(output_vectors).float()
      bias.data = torch.zeros(bias.size()).float()
      for param in output_l.parameters: param.requires_grad = output_learning
  
  # forward propagation
  def forward(self, input_inds, targets, lengths):
    
    # set up batch in pytorch
    b_size = len(input_inds)
    inputs = get_sentence_vectors(input_inds, self.vectors)
    inputs, indices = batch_setup(inputs, lengths)
    targets, _, _ = batch_setup(targets, lengths, pack=False, pad_val=-1, x_or_y="Y")
    
    # data --> GPU
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # forward computation (LSTM output)
    inputs_proc = self.lstm.forward(inputs, b_size)
    inputs_proc, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs_proc, batch_first=True)

    # get processed inputs, targets in correct form for output layer - (B x S)
    inputs_proc = inputs_proc.contiguous()
    inputs_proc = inputs_proc.view(-1, inputs_proc.shape[2])
    targets = targets.contiguous()
    targets = targets.view(-1)

    return self.output_l.forward(inputs_proc), targets

