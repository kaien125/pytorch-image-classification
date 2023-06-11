from PIL import Image, ImageChops
from numpy import *

image_path = '/home/kaien125/experiments/code/bee vision/pytorch-image-classification/images_resize0.5_contrastReduce101-120_num_image10000/composite_test/0/g_plus_new_l+1.png'
image = Image.open(image_path)

img_array = array(image)

print("shape: ", img_array.shape)

print("count: ", img_array.shape)

print(unique(img_array, return_counts=True))

print(len(unique(img_array)))

print(img_array[0])