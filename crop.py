# Importing Image class from PIL module
from PIL import Image, ImageChops
import random
from pathlib import Path
import os
from tqdm import tqdm
from numpy import *

def random_resize(image, scale_range):
    # Generate random scaling factors for width and height
    scale_factor = random.uniform(*scale_range)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    # Resize the image while preserving the aspect ratio
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

# Opens a image in RGB mode
# im = Image.open(r"images/part_whole_test/1/g_min_l_plus.jpg")

s_plus = (70, 65, 435, 430)
s_min = (60, 55, 470, 470)
g_plus_new_l = (60, 75, 450, 470)
g_min_new_l = (55, 50, 465, 470)
g_plus = (75, 85, 430, 430)
g_min = (70, 60, 470, 455)
g_min_l_plus = (50, 40, 490, 490)
 
# Cropped image of above dimension
def relocate(image_path):
    image_name = Path(image_path).stem.split('+')[0]
    print(image_name)
    if image_name == 's_plus':
        dimension = (70, 65, 435, 430)
    elif image_name == 's_min':
        dimension = (60, 55, 470, 470)
    elif image_name == 'g_plus_new_l':
        dimension = (60, 75, 450, 470)
    elif image_name == 'g_min_new_l':
        dimension = (55, 50, 465, 470)
    elif image_name == 'g_plus':
        dimension = (75, 85, 430, 430)
    elif image_name == 'g_min':
        dimension = (70, 60, 470, 455)
    elif image_name == 'g_min_l_plus':
        dimension = (50, 40, 490, 490)
    else:
        print("invalid image name.")
    image = Image.open(image_path)
    im1 = image.crop(dimension)
    # Define the range of scaling factors 
    scale_range = (0.5, 1.0)
    # Randomly resize the image
    im1 = random_resize(im1, scale_range)
    im1= im1.convert('1')
    # im1 = im1.rotate(random.uniform(-5, 5))
    # Create a blank white image
    blank_image = Image.new('RGB', (526, 526), 'white')
    # Generate random coordinates within the blank image
    x = random.randint(0, blank_image.width - im1.width)
    y = random.randint(0, blank_image.height - im1.height)
    # Paste the smaller image onto the blank image
    blank_image.paste(im1, (x, y))
    im1arr = array(blank_image)
    # im1arr = im1arr + random.randint(0,122)
    im1arr[im1arr == 255] -= random.randint(0,200)
    im1arr[im1arr == 0] += random.randint(0,200)
    blank_image = Image.fromarray(im1arr)
    # Make transform matrix, to multiply R by 0.2-1
    # Matrix = ( random.uniform(0.2, 1), 0,  0, 0, 
    #         0,   random.uniform(0.2, 1),  0, 0, 
    #         0,   0,  random.uniform(0.2, 1), 0,) 
    # output_Image = output_Image.convert("RGB", Matrix) 

    # Display the final image
    return blank_image



def main():
    repeat = 10000
    root = 'C:\PhD\experiments\code\pytorch-image-classification\images'
    for path, subdirs, files in tqdm(os.walk(root)):
        for name in files:
            image_path = os.path.join(path, name)
            for i in range(repeat):
                image_name = Path(image_path).stem
                output_path = image_path.replace(image_name, image_name+'+'+str(i))
                new_image = relocate(image_path)
                new_image.save(output_path)

if __name__ == "__main__":
    main()