MODEL="resnet18"
ROOT="C:\PhD\experiments\code\pytorch-image-classification\\"
BS=4
NUM_EPOCHS=10
RESIZE_SCALE=0.5
NUM_IMAGE=10000
ONE_TRAIN='False' # True or False
CONTRAST_REDUCE="121-140" # Choose from "1-20" "21-40" "41-60" "61-80" "81-100" "101-120" "121-140" "141-160" "161-180" "181-200"
IMAGE_SIZE=224 # Choose from 2 3 4 7 14 28 56 112 224 336 448
IMAGE_PATH="images_resize${RESIZE_SCALE}_contrastReduce${CONTRAST_REDUCE}_num_image${NUM_IMAGE}"
SUB_PATH="part_whole_test" # "learning_test" "part_whole_test" "global_test" "composite_test" "part_whole_flip_test" "learning_occ_test" "one_test" "part_whole_occ_test" "part_whole_flip_occ_test" 

python -m captum_bee.insights.attr_vis.example --model $MODEL --root $ROOT --image_size $IMAGE_SIZE --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $NUM_IMAGE --sub_path $SUB_PATH --one_train $ONE_TRAIN