MODEL="resnet18"
IMAGE_PATH="images_resize_relocate_10000"
NUM_IMAGES=10000
BS=4
NUM_EPOCHS=10
IMAGE_SIZE=224

TRAIN_FILE_NAME="training_${MODEL}_is${IMAGE_SIZE}_bs${BS}_e${NUM_EPOCHS}_i${NUM_IMAGES}_${IMAGE_PATH}.txt"
EVAL_FILE_NAME="eval_${MODEL}_is${IMAGE_SIZE}_bs${BS}_e${NUM_EPOCHS}_i${NUM_IMAGES}_${IMAGE_PATH}.txt"

python train.py --model $MODEL --image_size $IMAGE_SIZE --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $NUM_IMAGES > ${TRAIN_FILE_NAME} 

for SUB_PATH in "learning_test" "part_whole_test" "global_test" "composite_test" "part_whole_flip_test" 
do
   python eval.py --model $MODEL --image_size $IMAGE_SIZE --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $NUM_IMAGES --sub_path $SUB_PATH >> ${EVAL_FILE_NAME} 
done
