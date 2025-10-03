import torch

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1, 2, 3])
tensor2d = torch.tensor([[1, 2], [3, 4]])
tensor3d = torch.tensor([[[1,2], [3,4]], [[5,6], [7,8]]])

print(tensor3d)

print("1.-----------------------")

tensorInt = torch.tensor([1,2,3])
print(tensorInt.dtype)

tensorFloat = torch.tensor([1.0, 2.0, 3.0])
print(tensorFloat.dtype)

tensorInt = tensorInt.to(torch.float32)
print(tensorInt.dtype)


print("2.-----------------------")
tensor2d = torch.tensor([[1,2,3], [4,5,6]])
print(tensor2d.shape)

tensor2dt = tensor2d.T
print(tensor2dt)
print(tensor2dt.shape)

print(tensor2d@tensor2dt)

