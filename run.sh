MODEL="resnet18"
BS=4
NUM_EPOCHS=10
IMAGE_SIZE=224
RESIZE_SCALE=0.5
REPEAT=10


for CONTRAST_REDUCE in "1-20" "21-40" 
do
   # generate dataset 
   python data_prepare.py --resize_scale $RESIZE_SCALE --repeat $REPEAT --contrast_reduce $CONTRAST_REDUCE
   # training
   IMAGE_PATH="images_resize${RESIZE_SCALE}_contrastReduce${CONTRAST_REDUCE}_repeat${REPEAT}"
   TRAIN_FILE_NAME="training_${MODEL}_is${IMAGE_SIZE}_bs${BS}_e${NUM_EPOCHS}_${IMAGE_PATH}.txt"
   python train.py --model $MODEL --image_size $IMAGE_SIZE --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $REPEAT >> ${TRAIN_FILE_NAME} 

   # evaluate
   for SUB_PATH in "learning_test" "part_whole_test" "global_test" "composite_test" "part_whole_flip_test" 
   do
      EVAL_FILE_NAME="eval_${MODEL}_is${IMAGE_SIZE}_bs${BS}_e${NUM_EPOCHS}_i${REPEAT}_${IMAGE_PATH}.txt"
      python eval.py --model $MODEL --image_size $IMAGE_SIZE --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $REPEAT --sub_path $SUB_PATH >> ${EVAL_FILE_NAME} 
   done

done




