# Importing Image class from PIL module
from PIL import Image, ImageChops
import random, argparse
from pathlib import Path
import os
from tqdm import tqdm
from numpy import *

random.seed(100)

def random_resize(image, scale_range):
    # Generate random scaling factors for width and height
    scale_factor = random.uniform(*scale_range)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    # Resize the image while preserving the aspect ratio
    resized_image = image.resize((new_width, new_height))
    return resized_image
 
def image_transform(image_path, resize_scale, contrast_reduce):
    image = Image.open(image_path).resize((448, 448))
    # Define the range of scaling factors 
    scale_range = (resize_scale, 1.0)
    # Randomly resize the image
    img_resize = random_resize(image, scale_range)
    # im1= im1.convert('1')
    # # im1 = im1.rotate(random.uniform(-5, 5))
    # # Create a blank white image
    output_image = Image.new('RGB', (448, 448), 'white')
    # # Generate random coordinates within the blank image
    x = random.randint(0, output_image.width - img_resize.width)
    y = random.randint(0, output_image.height - img_resize.height)
    # # Paste the smaller image onto the blank image
    output_image.paste(img_resize, (x, y))
    img_array = array(output_image)
    img_array[img_array >= 127] = 255
    img_array[img_array <= 127] = 0
    
    # im1arr = array(blank_image)
    # # im1arr = im1arr + random.randint(0,122)
    if contrast_reduce:
        contrast_reduce = contrast_reduce.split('-')
        img_array[img_array == 255] -= random.randint(contrast_reduce[0],contrast_reduce[1])
        img_array[img_array == 0] += random.randint(contrast_reduce[0],contrast_reduce[1])
    output_image = Image.fromarray(img_array)
    # blank_image = Image.fromarray(im1arr)
    # # Make transform matrix, to multiply R by 0.2-1
    # # Matrix = ( random.uniform(0.2, 1), 0,  0, 0, 
    # #         0,   random.uniform(0.2, 1),  0, 0, 
    # #         0,   0,  random.uniform(0.2, 1), 0,) 
    # # output_Image = output_Image.convert("RGB", Matrix) 

    # Display the final image
    return output_image



def main():
    # Construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--resize_scale", default=0.5, help="resize_scale 0 - 1")
    ap.add_argument("--contrast_reduce", default='0-10', help="contrast reduce tuple between 0 and 255")
    ap.add_argument("--num_image", default=10, help="size of dataset")
    args= vars(ap.parse_args())

    resize_scale = float(args["resize_scale"])
    contrast_reduce = args["contrast_reduce"]
    num_image = int(args["num_image"])

    root = '/home/kaien125/experiments/code/bee vision/pytorch-image-classification/images_occlusion'
    output_subfolder_name = f"images_occlusion_resize{str(resize_scale)}_contrastReduce{contrast_reduce}_num_image{num_image}"

    # root = '/home/kaien125/experiments/code/bee vision/pytorch-image-classification/images'
    # output_subfolder_name = f"images_resize{str(resize_scale)}_contrastReduce{contrast_reduce}_num_image{num_image}"

    for path, subdirs, files in tqdm(os.walk(root)):
        for name in files:
            image_path = os.path.join(path, name)
            for i in range(num_image):
                image_name = Path(image_path).stem
                
                output_path = image_path.replace(image_name, image_name+'+'+str(i)).replace('images',output_subfolder_name)
                output_folder = output_path.rsplit('/', 1)[0]
                Path(output_folder).mkdir(parents=True, exist_ok=True)
                new_image = image_transform(image_path, resize_scale, contrast_reduce)
                new_image.save(output_path)

if __name__ == "__main__":
    main()