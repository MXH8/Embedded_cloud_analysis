# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:38:51 2023

@author: moh86vh
"""

import os
#import evaluate_preds #TODO: Import here on jetson orin
import numpy as np
from generators import mybatch_generator_prediction, pre_load_images
import datetime #used to measure inference time
import time #used to sleep

import tifffile as tiff
import pandas as pd
import sys #used to pass command line arguments

tflite_model_path = str(sys.argv[8])

#use tensorflow-lite via regular tf installation
#framework = 'framework tflite_imported_from_tf'
#import tensorflow as tf
#interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

#use tensorflow lite via tflite_runtime
framework = 'framework tflite_runtime'
import tflite_runtime as tfliteHead
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path=tflite_model_path)

import evaluate_preds #TODO: import here on devices other than jetson orin

from tqdm import tqdm
def get_input_image_names(list_names, directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in tqdm(list_names['name'], miniters=1000):
        nred = 'red_' + filenames
        nblue = 'blue_' + filenames
        ngreen = 'green_' + filenames
        nnir = 'nir_' + filenames

        if if_train:
            dir_type_name = "train"
            fl_img = []
            nmask = 'gt_' + filenames
            fl_msk = directory_name + '/train_gt/' + '{}.TIF'.format(nmask)
            list_msk.append(fl_msk)

        else:
            dir_type_name = "test"
            fl_img = []
            fl_id = '{}.TIF'.format(filenames)
            list_test_ids.append(fl_id)

        fl_img_red = directory_name + '/' + dir_type_name + '_red/' + '{}.TIF'.format(nred)
        fl_img_green = directory_name + '/' + dir_type_name + '_green/' + '{}.TIF'.format(ngreen)
        fl_img_blue = directory_name + '/' + dir_type_name + '_blue/' + '{}.TIF'.format(nblue)
        fl_img_nir = directory_name + '/' + dir_type_name + '_nir/' + '{}.TIF'.format(nnir)
        fl_img.append(fl_img_red)
        fl_img.append(fl_img_green)
        fl_img.append(fl_img_blue)
        fl_img.append(fl_img_nir)

        list_img.append(fl_img)

    if if_train:
        return list_img, list_msk
    else:
        return list_img, list_test_ids

#if we run into a problem using the whole test dataset at once, we can split that into multiple parts
def split_list(lst, x):
    # Calculate the size of each sublist
    sublist_size = len(lst) // x
    # Create the sublists
    sublists = [lst[i:i+sublist_size] for i in range(0, len(lst), sublist_size)]
    return sublists


def prediction():
    print("\nExperiment name: ", experiment_name)
    print("Prediction started... ")
    print("Input image size = ", (in_rows, in_cols))
    print("Number of input spectral bands = ", num_of_channels)
    print("Batch size = ", batch_sz)

    #prepare batch inference
    #reformat input shape to accept batches
    input_shape[0] = batch_sz
    interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
    new_input_details = interpreter.get_input_details()
    new_output_details = interpreter.get_output_details()
    interpreter.allocate_tensors() #has to be called to pre-plan tensors to optimize inference
    
    #Number of subsets the testdata will be split in to not exceed VRAM (since not the whole test dataset and the constructed images will be stored t once)
    noOfSubsets = int(len(test_img)/batch_sz) #575#9201
    test_img_subsets = split_list(test_img, noOfSubsets)
    test_ids_subsets = split_list(test_ids, noOfSubsets)
    processedSubset=0 #Information which subset is processed, also address for test_ids_subsets
    maxInferenceTime = 0.000
    minInferenceTime = 99.999
    maxCycleTime = 0.000
    minCycleTime = 99.999
    cycleStart = datetime.datetime.now()
    cycleEnd = datetime.datetime.now()
    #For average times, log 2000 cycles and then divide by that at the end
    cyclesToAverage = 9201 #probably problematic, if system throttles after some time
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
        
        #TODO WIP: check if sublist contains batch_sz images
            #If sublist is shorter, then the total number of images (9201) is not divisible by batch_sz and the last batch has the incorrect size
            #inference will not work and script will error out; so log inference times etc. here
        #print("len(sublist): ", len(sublist))
        #print("batch_sz: ", batch_sz)
        if len(sublist) != batch_sz:
            avgInferenceTime = 0.00
            avgCycleTime = 0.00
            #inferenceTimes list should be 1 longer than cycleTimes
            for i in range(len(cycleTimes)):
                avgCycleTime = avgCycleTime + cycleTimes[i].total_seconds()
            for i in range(len(inferenceTimes)):
                avgInferenceTime = avgInferenceTime + inferenceTimes[i].total_seconds()
            avgInferenceTime = avgInferenceTime/len(inferenceTimes)
            avgCycleTime = avgCycleTime/len(cycleTimes)
            print("last batch has wrong size; Logging times before erroring out")
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
            f.write('logged times before erroring out during last batch  (wrong vector length)\n')
            f.close()
        
        x = pre_load_images(sublist, in_rows, in_cols, batch_sz, max_bit) #returns the images already as a numpy array
        #print("loaded batch to predict")
        x = np.float32(x)
        b4_predict = datetime.datetime.now()
        #actual prediction with tflite
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        interpreter.set_tensor(new_input_details[0]['index'], x)
        #print("invoking interpreter")
        interpreter.invoke() #the thing with options is to run on gpu
        #print("interpreter finished")
        prediction_tensor = interpreter.get_tensor(new_output_details[0]['index'])
        after_predict = datetime.datetime.now()
        #measure/log inference time
        if (after_predict - b4_predict).total_seconds() > maxInferenceTime:
            maxInferenceTime = (after_predict - b4_predict).total_seconds()
        if (after_predict - b4_predict).total_seconds() < minInferenceTime:
            minInferenceTime = (after_predict - b4_predict).total_seconds()
        if len(inferenceTimes) < cyclesToAverage:
            inferenceTimes.append((after_predict - b4_predict))
        #print("Saving predicted cloud masks on disk... \n")
        pred_dir = PRED_FOLDER+'/'+experiment_name+'/tiles'
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        for image, image_id in zip(prediction_tensor, test_ids_subsets[processedSubset]):
            image = (image[:, :, 0]).astype(np.float32)
            tiff.imsave(os.path.join(pred_dir, str(image_id)), image)
        processedSubset=processedSubset+1
        cycleEnd = datetime.datetime.now()

        #TODO: calculate average inference time up until this point for every batch and print, in case inference is super slow
        #avgInferenceTime = 0.00
        #if len(inferenceTimes) >0:
        #    for i in range(len(inferenceTimes)):
        #        avgInferenceTime = avgInferenceTime + inferenceTimes[i].total_seconds()
        #avgInferenceTime = avgInferenceTime/len(inferenceTimes)
        #print("average inference times until now: ", avgInferenceTime)

        #print("cycle ended")
    avgInferenceTime = 0.00
    avgCycleTime = 0.00
    #inferenceTimes list should be 1 longer than cycleTimes
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
#arguments: intBatchSize   0or1UseCPU   0or1EnablePrefetch   resultsFolder   experiment_name
batch_sz = int(sys.argv[1]) #for batch_sz = 1, one image is computed at a time.
predict_on_cpu = int(sys.argv[2]) #1 = CPU, 0 = GPU
enable_prefetch = int(sys.argv[3]) #0 = no prefetch
PRED_FOLDER = str(sys.argv[4])
experiment_name = str(sys.argv[5])
system_descriptor = str(sys.argv[6]) #String that describes the hardware system (e.g. "jetson Orin NX 16GB" or "lenovo i5-1135G7")
logging_time = str(sys.argv[7]) #time that the logging script logs stuff
tflite_model_path = str(sys.argv[8])

interpreter.allocate_tensors() #has to be called to pre-plan tensors to optimize inference
input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details() #debug
input_shape = input_details[0]['shape']
#print("input_shape: ", input_shape) #debug
#get input and output dimensions
#print("input_details: ", input_details) #debug
#print("output_details: ", output_details) #debug

TRAIN_FOLDER = '38-cloud/38-Cloud_training'
TEST_FOLDER = '38-cloud/38-Cloud_test'

#size of input images, standard 384
in_rows = 192
in_cols = 192
#Number of input images at the same time / Number of Bands
num_of_channels = 4
#batch size is used in the custom prediction batch generator
#for batch_sz = 1, one image is computed at a time.

max_bit = 65535  # maximum gray level in landsat 8 images

# getting input images names
#df_test_img = pd.read_csv("/home/mxh/Desktop/gitlab/cloudNetMXH/38-cloud/38-Cloud_test/test_patches_38-Cloud.csv") #for linux
df_test_img = pd.read_csv("/home/mheimbach/cloudNetMXH/38-cloud/38-Cloud_test/test_patches_38-Cloud.csv") #for linux
#df_test_img = pd.read_csv("/cloudNetMXH/38-cloud/38-Cloud_test/test_patches_38-Cloud.csv") #for linux
#df_test_img=pd.read_csv('38-cloud/38-Cloud_test/test_patches_38-cloud.csv') #for windows
test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER, if_train=False)

#Log all relevant experiment parameters to file
if not os.path.exists(PRED_FOLDER+'/'+experiment_name):
    os.makedirs(PRED_FOLDER+'/'+experiment_name)
f = open(PRED_FOLDER+'/'+experiment_name+'/params.txt', "w")
if 'runtime' in framework:
    f.write(framework +' '+ tfliteHead.__version__ + '\n')
else:
    f.write(framework +' ' + tf.__version__ + '\n')
f.write('batch_sz '+ repr(batch_sz)+'\n')
f.write('predict_on_cpu '+ repr(predict_on_cpu)+'\n')
f.write('enable_prefetch '+ repr(enable_prefetch)+'\n')
f.write('PRED_FOLDER '+ PRED_FOLDER+'\n')
f.write('experiment_name '+ experiment_name+'\n')
f.write('in_rows '+ repr(in_rows)+'\n')
f.write('in_cols '+ repr(in_cols)+'\n')
f.write('tflite_model_path '+ tflite_model_path+'\n')
f.write('system_descriptor '+ system_descriptor+'\n')
f.write('logging_time '+ logging_time+'\n')
f.close()

#Allow Tensorflow free growth:
#if not predict_on_cpu: #predict on gpu
#    from tensorflow.compat.v1 import ConfigProto
#    from tensorflow.compat.v1 import InteractiveSession
#    config = ConfigProto()
#    config.gpu_options.allow_growth = True
#    session = InteractiveSession(config=config)
#    with tf.device('/gpu:0'):
#        prediction()
#use cpu
#else: 
#    with tf.device('/cpu:0'):
prediction()

#execute evaluate_predictions(preds_folder, preds_folder_root, gt_folder_path):
evaluate_preds.evaluate_predictions('tiles', PRED_FOLDER+'/'+experiment_name, TEST_FOLDER+'/Entire_scene_gts')
