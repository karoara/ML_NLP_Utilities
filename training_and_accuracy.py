# TRAINING AND ACCURACY UTILITIES ---------------------------------------------
# 
# Functions for training models & calculating model performance (Pytorch).


import torch
from train_utils import write
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Returns number (or percentage) of correct outputs from those given.
def bin_cla_acc_helper(outputs, targets, avg=False):
  
  outputs = torch.round(outputs)
  indicators = torch.abs(outputs - targets)
  classfn_acc = len(indicators) - torch.sum(indicators)
   
  if avg: return classfn_acc/len(outputs)
  else: return classfn_acc


# If type = binary classification, returns percentage of correctly classified
# datapoints. Otherwise (if type = language modeling), returns avg crossentropy &
# perplexity on all words in the data.
def acc_helper(model, inputs, targets, lengths, batch_size, type):
    
    model.eval()
    if type == "bin_classfn": classfn_acc = 0
    else: # if language modeling
      crossent_loss, tot_words = 0, np.sum(lengths)
      loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    
    # switch off gradients
    with torch.no_grad():
      
      # set up, iterate over batches of data
      if len(inputs) % batch_size == 0: iterations = len(inputs)//batch_size
      else: iterations = len(inputs)//batch_size + 1
      for batch in range(iterations):
        
        # get batch, perform forward pass
        batch_inputs = inputs[batch*batch_size : (batch+1)*batch_size]
        batch_targets = targets[batch*batch_size : (batch+1)*batch_size]
        batch_lengths = lengths[batch*batch_size : (batch+1)*batch_size]
        batch_outputs, batch_targets = model.forward(batch_inputs, batch_targets, batch_lengths)
        
        # update accuracy
        if type == "bin_classfn": classfn_acc += bin_cla_acc_helper(batch_outputs, batch_targets)
        else: crossent_loss += loss_fn(batch_outputs, batch_targets).item()
    
    model.train()
    if type == "bin_classfn": return classfn_acc/len(inputs)
    else: return crossent_loss/tot_words, np.exp(crossent_loss)

