#!/bin/bash

#Set jetson power mode:
#nvpmodel -m 0/1/2/...
#use argument --force to force automatic reboot
#for orin: 0=MAXN, 1=10W, 2=15W, 3=25W
#for nano: 0=maxn, 1=5W

#system_descriptor="Jetson-Orin-NX-16GB-Kit"
#system_descriptor="Moritz_Tower,i9-9900k,32GB2300MHz_RTX3080"
#system_descriptor="Lenovo-TP-T15,i5-1135G7,16GB3200MHz_RTX3060"
#system_descriptor="Jetson-Nano-2GB"
#system_descriptor="Raspi_4B"
#system_descriptor="Raspi_5"
#system_descriptor="Tower_14700_4080"
#system_descriptor="ESSEO22006-01_R5-7530U"
system_descriptor="mifcomLaptop_8845HS_RTX4060"

#inference_file="main_test.py" #regular tensorflow
inference_file="tflite_inference.py" #tflite

#model_path="original_cloudnet.h5" #weights for regular cloudnet in tensorflow
#model_path="convertedModel8BitWeights.tflite" #tflite dynamically quantized to 8 bit
model_path="convertedModelFloat16.tflite" #tflite quantized to float16
#model_path="convertedModelFloat32.tflite" #tflite quantized to float32
predict_on_cpu=0
enable_prefetch=0
PRED_FOLDER="cloudnetMXH"
logging_time="20m"

#--------------------------------------------------
execute_cloudnet_experiment() {
#run test script and logging script at the same time, continue when both are finished
python $inference_file $batch_sz $predict_on_cpu $enable_prefetch $PRED_FOLDER $experiment_name $system_descriptor $logging_time $model_path & 
bash logging_script_v2.sh $PRED_FOLDER"/"$experiment_name $logging_time &
wait
echo "test finished, deleting tiles and scenes"
rm -r $PRED_FOLDER"/"$experiment_name"/tiles"
rm -r $PRED_FOLDER"/"$experiment_name"/entire_masks_tiles"
echo "tiles and scenes deleted"
echo "---------------------------------------------------------------------"
}
#--------------------------------------------------

#regular tensorflow, float32 CPU
#inference_file="main_test.py" 
#model_path="original_cloudnet.h5"
#PRED_FOLDER="cloudnetMXH_CPU"
#predict_on_cpu=1
#batch_sz=1
#experiment_name="batchsz1"
#execute_cloudnet_experiment

#batch_sz=2
#experiment_name="batchsz2"
#execute_cloudnet_experiment

#regular tensorflow, GPU
#inference_file="main_test.py" 
#model_path="original_cloudnet.h5"
#PRED_FOLDER="cloudnetMXH_CPU"
#predict_on_cpu=0
#batch_sz=1
#experiment_name="batchsz1"
#execute_cloudnet_experiment

#batch_sz=2
#experiment_name="batchsz2"
#execute_cloudnet_experiment

#tflite float32
inference_file="tflite_inference.py" 
model_path="convertedModelFloat32.tflite"
PRED_FOLDER="cloudnetMXH_tflitefloat32"
predict_on_cpu=1
batch_sz=1
experiment_name="batchsz1"
execute_cloudnet_experiment

#batch_sz=2
#experiment_name="batchsz2"
#execute_cloudnet_experiment

#tflite float16
inference_file="tflite_inference.py" 
model_path="convertedModelFloat16.tflite"
PRED_FOLDER="cloudnetMXH_tflitefloat16"
predict_on_cpu=1
batch_sz=1
experiment_name="batchsz1"
execute_cloudnet_experiment

#batch_sz=2
#experiment_name="batchsz2"
#execute_cloudnet_experiment

#tflite dynamic 8 bit
inference_file="tflite_inference.py" 
model_path="convertedModel8BitWeights.tflite"
PRED_FOLDER="cloudnetMXH_tflite8bit"
predict_on_cpu=1
batch_sz=1
experiment_name="batchsz1"
execute_cloudnet_experiment

#batch_sz=2
#experiment_name="batchsz2"
#execute_cloudnet_experiment