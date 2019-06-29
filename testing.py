
import torch

a = torch.nn.Linear(5, 2)
# for param in a.parameters(): 
  # print(param)
  # print(param.size())
b, c = list(a.parameters())
print(b)
print("\n")
print(b.data)
print("\n")
b.data = torch.zeros(2, 5)
print(b)
print("\n")
print(c)
print("\n")
print("Now try")
for param in a.parameters(): 
  print(param.requires_grad)
  param.requires_grad = True
  print(param)
  print(param.requires_grad)