import torch
import torch.nn.functional as F
from torch.autograd import grad

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

print("3.-----------------------")

y = torch.tensor([1.0])

x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2])
b = torch.tensor([0.0])
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a,y)
print(loss)

print("4.-----------------------")

y = torch.tensor([1.0])

x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a,y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)

loss.backward()
print(w1.grad)
print(b.grad)

print("5.-----------------------")

