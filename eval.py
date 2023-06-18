import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import sys, argparse
import pandas as pd

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="Training mode: resnet18/vgg11/mobilenetv2")
ap.add_argument("--bs", required=True, help="batch size")
ap.add_argument("--num_epochs", required=True, help="number of epochs")
ap.add_argument("--image_path", required=True, help="image_path")
ap.add_argument("--num_images", required=True, help="num_images")
ap.add_argument("--sub_path", required=True, help="sub_path")
ap.add_argument("--image_size", required=True, help="image_size")
args= vars(ap.parse_args())

# Set training mode
train_mode=args["model"]
# Batch size
bs = int(args["bs"])
# Number of epochs
num_epochs = int(args["num_epochs"])
# Number of images 
num_img = int(args["num_images"])
image_size = int(args["image_size"])
image_path = args["image_path"]
sub_path = args["sub_path"]
augment = image_path.replace('images','')
# Paths for image directory and model
EVAL_DIR=image_path+'/'+sub_path
# EVAL_MODEL='models/mobilenetv2.pth'
# Set the model save path
EVAL_MODEL = train_mode + '_one_train_is'+ str(image_size) +'_bs' + str(bs) + '_e' + str(num_epochs) + '_i'+str(num_img) + augment + '.pth'
# Load the model for evaluation
model = torch.load("models/"+EVAL_MODEL)
model.eval()
print("")
print(EVAL_MODEL)
print(sub_path)

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

resize = find_between(augment, "resize", "_con")
contrast_reduce = find_between(augment, "Reduce", "_num_image")
# Configure batch size and nuber of cpu's
num_cpu = multiprocessing.cpu_count()
# bs = 1

# Prepare the eval data loader
eval_transform=transforms.Compose([
        transforms.Resize(size=image_size),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
        ])

eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes=len(eval_dataset.classes)
dsize=len(eval_dataset)

# Class label names
class_names=['plus','min']

# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

# Evaluate the model accuracy on the dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy=100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, 
    overall_accuracy))

# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-'*16)
print(conf_mat,'\n')

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print('Per class accuracy')
print('-'*18)
for label,accuracy in zip(eval_dataset.classes, class_accuracy):
     class_name=class_names[int(label)]
     print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))

with open(train_mode+'_one_eval.csv', 'a') as fd:
    fd.write(f'\n{sub_path},{bs},{image_size},{num_epochs},{class_accuracy[0]/100},{class_accuracy[1]/100},{correct/total},{resize},{contrast_reduce},{EVAL_MODEL}')
'''
Sample run: python eval.py eval_ds
'''
