MODEL="resnet18"
IMAGE_PATH="images"
NUM_IMAGES=2
BS=2
NUM_EPOCHS=10

TRAIN_FILE_NAME="training_${MODEL}_bs${BS}_e${NUM_EPOCHS}_i${NUM_IMAGES}_${IMAGE_PATH}.txt"
EVAL_FILE_NAME="eval_${MODEL}_bs${BS}_e${NUM_EPOCHS}_i${NUM_IMAGES}_${IMAGE_PATH}.txt"

python train.py --model $MODEL --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $NUM_IMAGES > ${TRAIN_FILE_NAME} 

for SUB_PATH in "learning_test" "part_whole_test" "global_test" "composite_test"
do
   python eval.py --model $MODEL --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $NUM_IMAGES --sub_path $SUB_PATH > ${EVAL_FILE_NAME} 
done
