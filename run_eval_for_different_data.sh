MODEL="resnet18"
BS=4
NUM_EPOCHS=10
RESIZE_SCALE=0.5
NUM_IMAGE=10000
train_image_size=56
train_contrast_reduce="121-140"


for CONTRAST_REDUCE in "1-20" "21-40" "41-60" "61-80" "81-100" "101-120" "121-140" "141-160" "161-180" "181-200"
do

   # for test_image_size in 7 14 28 56 112 224 336 448
   for test_image_size in 2
   do
      IMAGE_PATH="images_resize${RESIZE_SCALE}_contrastReduce${CONTRAST_REDUCE}_num_image${NUM_IMAGE}"
      # evaluate
      for SUB_PATH in "learning_test" "part_whole_test" "part_whole_flip_test" 
      do
         EVAL_FILE_NAME="eval_${MODEL}_bs${BS}_e${NUM_EPOCHS}_trainIs${train_image_size}_trainContrastReduce${train_contrast_reduce}_testIs${test_image_size}_i${NUM_IMAGE}_test${IMAGE_PATH}.txt"
         python eval_for_different_data.py \
         --model $MODEL \
         --test_image_size $test_image_size \
         --bs $BS \
         --num_epochs $NUM_EPOCHS \
         --image_path $IMAGE_PATH \
         --num_images $NUM_IMAGE \
         --sub_path $SUB_PATH \
         --train_image_size $train_image_size \
         --train_contrast_reduce $train_contrast_reduce>> ${EVAL_FILE_NAME} 
      done
   done

done




