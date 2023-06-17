from torchvision import models, utils
import torch
import matplotlib.pyplot as plt

MODEL='C:\PhD\experiments\code\pytorch-image-classification\models\resnet18.pth'


# Load the model for testing
model = models.resnet18(weights=None)
model.eval()

# print(model.conv1.weight.data)

for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        print(m.weight.data)

        w = m.weight.data
        grid = utils.make_grid(w, nrow=10, normalize=True, scale_each=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid[0,:])


        plt.show()