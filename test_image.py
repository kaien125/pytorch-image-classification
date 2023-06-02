from PIL import Image, ImageChops
from numpy import *

image_path = 'pytorch-image-classification/images/learning_test/0/s_plus+0.jpg'
image = Image.open(image_path)

img_array = array(image)

print("shape: ", img_array.shape)

print("count: ", img_array.shape)

print(unique(img_array, return_counts=True))

print(len(unique(img_array)))

print(img_array[0])