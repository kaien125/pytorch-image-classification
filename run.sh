MODEL="resnet18"
BS=4
NUM_EPOCHS=10
RESIZE_SCALE=0.5
NUM_IMAGE=10000
training_output_folder="training_one_output_${MODEL}_bs${BS}_e${NUM_EPOCHS}_resize${RESIZE_SCALE}_num_img${NUM_IMAGE}"
mkdir ${training_output_folder}
eval_output_folder="eval_one_output_${MODEL}_bs${BS}_e${NUM_EPOCHS}_resize${RESIZE_SCALE}_num_img${NUM_IMAGE}"
mkdir ${eval_output_folder}


for CONTRAST_REDUCE in "1-20" "121-140" "181-200"
# for CONTRAST_REDUCE in "1-20" "21-40" "41-60" "61-80" "81-100" "101-120" "121-140" "141-160" "161-180" "181-200"
do
   # generate dataset 
   # python data_prepare.py --resize_scale $RESIZE_SCALE --num_image $NUM_IMAGE --contrast_reduce $CONTRAST_REDUCE

   # for IMAGE_SIZE in 2 3 4 7 14 28 56 112 224 336 448
   for IMAGE_SIZE in 56 112 224
   do
      # training
      IMAGE_PATH="images_resize${RESIZE_SCALE}_contrastReduce${CONTRAST_REDUCE}_num_image${NUM_IMAGE}"
      TRAIN_FILE_NAME="${training_output_folder}/training_one_${MODEL}_is${IMAGE_SIZE}_bs${BS}_e${NUM_EPOCHS}_${IMAGE_PATH}.txt"
      python train.py --model $MODEL --image_size $IMAGE_SIZE --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $NUM_IMAGE >> ${TRAIN_FILE_NAME} 

      # evaluate
      # for SUB_PATH in "learning_test" "part_whole_test" "global_test" "composite_test" "part_whole_flip_test" "learning_occ_test" "one_test" "part_whole_occ_test" "part_whole_flip_occ_test" 
      for SUB_PATH in "part_whole_test" "learning_test"
      do
         EVAL_FILE_NAME="${eval_output_folder}/eval_one_${MODEL}_is${IMAGE_SIZE}_bs${BS}_e${NUM_EPOCHS}_i${NUM_IMAGE}_${IMAGE_PATH}.txt"
         python eval.py --model $MODEL --image_size $IMAGE_SIZE --bs $BS --num_epochs $NUM_EPOCHS --image_path $IMAGE_PATH --num_images $NUM_IMAGE --sub_path $SUB_PATH >> ${EVAL_FILE_NAME} 
      done
   done

done




