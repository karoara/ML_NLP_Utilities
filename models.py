# MODELS ----------------------------------------------------------------------
# 
# Useful model abstractions in Pytorch. 


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

class MY_LSTM(torch.nn.Module):
  
  # initialization method
  def __init__(self, in_d, h_d, layers=1, dropout=0.0, bi=False, all_states=False):
    super(MY_LSTM, self).__init__()
    
    # # of states returned, bidirectionality, hidden/cell states, inner LSTM
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
    if not self.lang: 
      if self.out_vects == 2: final_output = torch.cat((h_final[-2], h_final[-1]), 1) 
      else: final_output = h_final[-1]
    else: final_output = h_all
    
    return final_output
