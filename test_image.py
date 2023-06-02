from PIL import Image, ImageChops
from numpy import *

image_path = 'pytorch-image-classification/images_origin/composite_test/0/g_plus_new_l.png'
image = Image.open(image_path)

img_array = array(image)

print("shape: ", img_array.shape)

print("count: ", img_array.shape)

print(unique(img_array, return_counts=True))

print(len(unique(img_array)))

print(img_array[0])