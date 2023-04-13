import torch

# print(torch.__version__)
# print("GPU: ", torch.cuda.is_available())

a = torch.randn(2, 3)
print(isinstance(a, torch.FloatTensor))

