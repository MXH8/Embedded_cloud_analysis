from __future__ import print_function
import os
#import evaluate_preds # TODO: needs to be imported here on jetson orin
import numpy as np
import cloud_net_model
#custom prediction generator is used 
from generators import mybatch_generator_prediction, pre_load_images
import tifffile as tiff
import pandas as pd
from utils import get_input_image_names
import tensorflow as tf
import datetime #used to measure inference time
import sys #used to pass command line arguments
import evaluate_preds #TODO: needs to be imported here on machines other than jetson orin

#if we run into a problem using the whole test dataset at once, we can split that into multiple parts
def split_list(lst, x):
    # Calculate the size of each sublist
    sublist_size = len(lst) // x
    sublists = [lst[i:i+sublist_size] for i in range(0, len(lst), sublist_size)]
    return sublists

def prediction():
    model = cloud_net_model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.load_weights(weights_path)

    print("\nExperiment name: ", experiment_name)
    print("Prediction started... ")
    print("Input image size = ", (in_rows, in_cols))
    print("Number of input spectral bands = ", num_of_channels)
    print("Batch size = ", batch_sz)

    #Number of subsets the testdata will be split in to not exceed VRAM (since not the whole test dataset and the constructed images will be stored t once)
    noOfSubsets = int(len(test_img)/batch_sz) #in 38-cloud there are 9201 images to infer in total
    test_img_subsets = split_list(test_img, noOfSubsets)
    test_ids_subsets = split_list(test_ids, noOfSubsets)
    processedSubset=0 #Information which subset is processed, also address for test_ids_subsets
    maxInferenceTime = 0.000
    minInferenceTime = 99.999
    maxCycleTime = 0.000
    minCycleTime = 99.999
    cycleStart = datetime.datetime.now()
    cycleEnd = datetime.datetime.now()
    cyclesToAverage = 9201 #max number of cacles to take the average over
    inferenceTimes = []
    cycleTimes = []
    f = open(PRED_FOLDER+'/'+experiment_name+'/params.txt', "a")
    f.write('start_time '+ repr(datetime.datetime.now())+'\n')
    f.close()
    for sublist in test_img_subsets:
        if processedSubset >=1:
            if (cycleEnd - cycleStart).total_seconds() > maxCycleTime:
                maxCycleTime = (cycleEnd - cycleStart).total_seconds()
            if (cycleEnd - cycleStart).total_seconds() < minCycleTime:
                minCycleTime = (cycleEnd - cycleStart).total_seconds()
            if len(cycleTimes) < cyclesToAverage:
                cycleTimes.append((cycleEnd - cycleStart))
        cycleStart = datetime.datetime.now()
        if processedSubset%25 == 0:
            print("Processing subset ",processedSubset)
        x = pre_load_images(sublist, in_rows, in_cols, batch_sz, max_bit)
        #imgs_mask_test = model.predict(x, steps=np.ceil(len(sublist) / batch_sz))
        b4_predict = datetime.datetime.now()
        imgs_mask_test_tensor = model(x) #probably faster than using presict?!
        after_predict = datetime.datetime.now()
        #measure/log inference time
        if (after_predict - b4_predict).total_seconds() > maxInferenceTime:
            maxInferenceTime = (after_predict - b4_predict).total_seconds()
        if (after_predict - b4_predict).total_seconds() < minInferenceTime:
            minInferenceTime = (after_predict - b4_predict).total_seconds()
        if len(inferenceTimes) < cyclesToAverage:
            inferenceTimes.append((after_predict - b4_predict))
        imgs_mask_test = imgs_mask_test_tensor.numpy()
        #print("Saving predicted cloud masks on disk... \n")

        pred_dir = PRED_FOLDER+'/'+experiment_name+'/tiles'
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        for image, image_id in zip(imgs_mask_test, test_ids_subsets[processedSubset]):
            image = (image[:, :, 0]).astype(np.float32)
            tiff.imsave(os.path.join(pred_dir, str(image_id)), image)
        processedSubset=processedSubset+1
        cycleEnd = datetime.datetime.now()
    avgInferenceTime = 0.00
    avgCycleTime = 0.00
    #calculate average inference time and cycle time
    for i in range(len(cycleTimes)):
        avgCycleTime = avgCycleTime + cycleTimes[i].total_seconds()
    for i in range(len(inferenceTimes)):
        avgInferenceTime = avgInferenceTime + inferenceTimes[i].total_seconds()
    avgInferenceTime = avgInferenceTime/len(inferenceTimes)
    avgCycleTime = avgCycleTime/len(cycleTimes)
    print("Prediction finished")
    print("Appending timing information to .txt")
    f = open(PRED_FOLDER+'/'+experiment_name+'/params.txt', "a")
    f.write('end_time '+ repr(datetime.datetime.now())+'\n')
    f.write('no_of_batches '+ repr(processedSubset)+'\n')
    f.write('maxInferenceTime '+ repr(maxInferenceTime)+'\n')
    f.write('minInferenceTime '+ repr(minInferenceTime)+'\n')
    f.write('avgInferenceTime '+ repr(avgInferenceTime)+'\n')
    f.write('maxCycleTime '+ repr(maxCycleTime)+'\n')
    f.write('minCycleTime '+ repr(minCycleTime)+'\n')
    f.write('avgCycleTime '+ repr(avgCycleTime)+'\n')
    f.close()

#evaluate command line arguments
batch_sz = int(sys.argv[1]) #for batch_sz = 1, one image is computed at a time.
predict_on_cpu = int(sys.argv[2]) #1 = CPU, 0 = GPU
enable_prefetch = int(sys.argv[3]) #0 = no prefetch
PRED_FOLDER = str(sys.argv[4])
experiment_name = str(sys.argv[5])
system_descriptor = str(sys.argv[6]) #String that describes the hardware system (e.g. "jetson Orin NX 16GB" or "lenovo i5-1135G7")
logging_time = str(sys.argv[7]) #time that the logging script logs stuff
weights_path = str(sys.argv[8]) #model path

TRAIN_FOLDER = '38-cloud/38-Cloud_training'
TEST_FOLDER = '38-cloud/38-Cloud_test'

#size of quadratical input images
in_rows = 192 #384
in_cols = 192 #384
#Number of input images at the same time / Number of Bands
num_of_channels = 4
#Final layer is conv2D with 1x1 kernel; num_of_classes determines the depth of the output space
num_of_classes = 1
#batch size is used in the custom prediction batch generator

max_bit = 65535  # maximum gray level in landsat 8 images

# getting input images names
df_test_img = pd.read_csv("/home/mxh/Embedded_cloud_analysis/38-cloud/38-Cloud_test/test_patches_38-Cloud.csv") #TODO: for linux
#df_test_img=pd.read_csv('38-cloud/38-Cloud_test/test_patches_38-cloud.csv') #TODO: for windows
test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER, if_train=False)

#Log all relevant experiment parameters to file
if not os.path.exists(PRED_FOLDER+'/'+experiment_name):
    os.makedirs(PRED_FOLDER+'/'+experiment_name)
f = open(PRED_FOLDER+'/'+experiment_name+'/params.txt', "w")
f.write('framework tensorflow' +' ' + tf.__version__ + '\n')
f.write('batch_sz '+ repr(batch_sz)+'\n')
f.write('predict_on_cpu '+ repr(predict_on_cpu)+'\n')
f.write('enable_prefetch '+ repr(enable_prefetch)+'\n')
f.write('PRED_FOLDER '+ PRED_FOLDER+'\n')
f.write('experiment_name '+ experiment_name+'\n')
f.write('in_rows '+ repr(in_rows)+'\n')
f.write('in_cols '+ repr(in_cols)+'\n')
f.write('weights_path '+ weights_path+'\n')
f.write('system_descriptor '+ system_descriptor+'\n')
f.write('logging_time '+ logging_time+'\n')
f.close()

#Allow Tensorflow free growth:
if not predict_on_cpu: #predict on gpu
    #from tensorflow.compat.v1 import ConfigProto
    #from tensorflow.compat.v1 import InteractiveSession
    #config = ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)
    gpus = tf.config.list_physical_devices('GPU')
    try: 
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    with tf.device('/gpu:0'):
        prediction()
#use cpu
else: 
    with tf.device('/cpu:0'):
        prediction()

#execute evaluate_predictions(preds_folder, preds_folder_root, gt_folder_path):
evaluate_preds.evaluate_predictions('tiles', PRED_FOLDER+'/'+experiment_name, TEST_FOLDER+'/Entire_scene_gts')





