# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:14:19 2023

This is supposed to replace the evaluation.m-file
stitch predictions together and get jaccard index etc

@author: moh86vh
"""

import os
import cv2
import numpy as np
from PIL import Image

# Define the folder paths and parameters
#paths for windows environment
#gt_folder_path = 'C:\\Users\\moh86vh\\Desktop\\gitlab\\cloudNetMXH\\38-cloud\\38-Cloud_test\\Entire_scene_gts' #contains ground truth scenes
#preds_folder_root = 'C:\\Users\\moh86vh\\Desktop\\gitlab\\cloudNetMXH\\Predictions' #parent folder that contains the folder with all the prediction tiles
#paths for linux environment
#gt_folder_path = "/home/mxh/Desktop/mxh/cloudNetMXH/38-cloud/38-Cloud_test/Entire_scene_gts" #contains ground truth scenes
#preds_folder_root = "/home/mxh/Desktop/mxh/cloudNetMXH/Predictions" #contains ground truth scenes
#preds_folder = 'test1' #folder that contains all the predicted tiles that together form scenes
#size of predicted images
pr_patch_size_rows = 384
pr_patch_size_cols = 384
#how many patches in cols and rows does the largest scene have?
largest_col_nr = 24
largest_row_nr = 24
#0 = clear, 1 = cloud
#threshold under which pixel will be classified as 0
thresh = 12 / 255

# Function to extract unique scene IDs from prediction folder
def extract_unique_sceneids(result_root, preds_dir):
    path_4landtype = os.path.join(result_root, preds_dir)
    folders_inside_landtype = os.listdir(path_4landtype)
    
    sceneid_lists = []
    for folder_name in folders_inside_landtype:
        raw_result_patch_name = os.path.splitext(folder_name)[0]
        str = [raw_result_patch_name]
        loc1 = str[0].find('LC')
        leng = len(raw_result_patch_name)
        sceneid_lists.append(raw_result_patch_name[loc1:leng])
    
    uniq_sceneid = list(set(sceneid_lists[2:]))  # Removing duplicates
    return uniq_sceneid

# Function to extract row and column number from patch name
#used to determine where the patch will be in the scene
def extract_rowcol_each_patch(name):
    name = os.path.splitext(name)[0]
    #print("name: ", name)
    loc1 = name.find('LC')
    #print("loc1: ", loc1)
    loc2 = name.find('h_')
    #print("loc2: ", loc2)
    patchbad = name[loc2 + 2:loc1 - 1]
    #print("patchbad: ", patchbad)
    loc3 = patchbad.split('_')
    #first element is number between 0 and 576 or so, second is row, third is "by", fourth is col
    row = int(loc3[1])
    #print("row: ", row)
    col = int(loc3[3])
    #print("col: ", col)
    return row, col

# Function to get patches corresponding to a unique scene ID
def get_patches_for_sceneid(result_root, preds_dir, sceneid):
    path_4preds = os.path.join(result_root, preds_dir)
    files_inside = os.listdir(path_4preds)
    
    related_patches = []
    for file_name in files_inside:
        if sceneid in file_name:
            related_patches.append(file_name)
    
    return related_patches

# Function to remove some of the black/zero padding around the image to exactly match the size of the actual ground truth mask
def unzeropad(in_dest, in_source, largest_col, largest_row):
    #in_dest = complete prediction mask; size is that of the largest scene
    #in_source = ground truth
    #first cut image to actual number of patches (some scenes only have 20x21 patches, others have 24x24 or so)
    #lowest value for row and col is 1
    in_dest = in_dest[0:largest_row*384, 0:largest_col*384]
    #equally remove padding from all 4 sides to exactly match the gt-mask
    ny, nx = in_dest.shape
    nys, nxs = in_source.shape
    deltay = (ny - nys)//2
    deltax = (nx - nxs)//2
    out = in_dest[deltay:deltay+nys, deltax:deltax+nxs]
    return out

# Function to calculate quantitative evaluators
def QE_calcul(predict, gt):
    #Get confusion matrix
    ysize, xsize = predict.shape
    print("beginning confusion matrix evaluation")
    #np.where() returns an array with two vectors that contain the x- and y address in the image. The length of one of these vectors is the amount of points returned by np.where()
    truetrue = np.where(predict & gt)
    truefalse = np.where(np.logical_not(predict) & np.logical_not(gt))
    falsetrue = np.where(predict & np.logical_not(gt))
    falsefalse = np.where(np.logical_not(predict) & gt)
    #calculate jaccard index
    #divide the amount of true predictions (1 in both predict and gt) by the amount of trues in either the prediction or the mask (1 in predict and/or gt)
    smooth = 0.0000001 #needed to not divide by 0 in edge case
    #intersection = len(np.where(predict & gt)[0]) #The same as TrueTrue
    #gtOrPred = len(np.where(predict | gt)[0]) #TrueTrue + FalseTrue + FalseFalse
    #jaccard_index_npwhere = (intersection+smooth)/(gtOrPred+smooth)
    jaccard_index = (len(truetrue[0]) + smooth) / (len(truetrue[0]) + len(falsetrue[0]) + len(falsefalse[0]) + smooth)
    #print("jaccard_index (using TrueTrue): ", jaccard_index_TrueTrue)
    
    conf_matrix = [len(truetrue[0]), len(truefalse[0]), len(falsetrue[0]), len(falsefalse[0])]
    precision = conf_matrix[0]/(conf_matrix[0] + conf_matrix[2]) #precision
    recall = conf_matrix[0]/(conf_matrix[0] + conf_matrix[3]) #recall
    specificity = conf_matrix[1]/(conf_matrix[1] + conf_matrix[2]) #specificity
    accuracy = (conf_matrix[0] + conf_matrix[1])/(sum(conf_matrix)) #accuracy
    return [precision*100, recall*100, specificity*100, jaccard_index*100, accuracy*100]
    

def evaluate_predictions(preds_folder, preds_folder_root, gt_folder_path):
    # Getting unique scene IDs
    all_uniq_sceneid = extract_unique_sceneids(preds_folder_root, preds_folder)

    # Initialize vector for quantitative evaluators
    QE = []

    for n, scene_id in enumerate(all_uniq_sceneid):
        if n == len(all_uniq_sceneid) - 1:
            print(f'Working on sceneID # {n + 1}: {scene_id}\n')
        else:
            print(f'Working on sceneID # {n + 1}: {scene_id} ...\n')
        
        #load ground truth
        gt_path = os.path.join(gt_folder_path, f'edited_corrected_gts_{scene_id}.TIF')
        #print("gt_path: ", gt_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        #extract all patches relating to this scene id
        scid_related_patches = get_patches_for_sceneid(preds_folder_root, preds_folder, scene_id)
        
        #Scenes can be smaller than 24*384 by 24*384, count highest row and col nr and then cut accordingly
        complete_pred_mask = np.zeros([largest_row_nr*pr_patch_size_rows,largest_col_nr*pr_patch_size_cols], dtype=bool)
        #Use these variables to determine scene size
        largest_row = 0
        largest_col = 0
        
        #iterate through all patches that form the scene
        for pcount, patch_name in enumerate(scid_related_patches):
            #print("patch_name: ", patch_name)
            #load patch, convert to numpy-array and extract size
            predicted_patch_path = os.path.join(preds_folder_root, preds_folder, patch_name)
            predicted_patch_tiff = Image.open(predicted_patch_path)
            predicted_patch = np.array(predicted_patch_tiff)
            w, h = predicted_patch.shape
            
            #resize patch if size is wrong
            if w != pr_patch_size_rows or h != pr_patch_size_cols:
                predicted_patch = cv2.resize(predicted_patch, (pr_patch_size_rows, pr_patch_size_cols))
            
            #convert probability values between 0 and 1 to 0 and 1 by logical comparison with threshold value
            predicted_patch = predicted_patch > thresh
            
            #extract position of patch in the complete scene
            patch_row, patch_col = extract_rowcol_each_patch(patch_name)
            if patch_col > largest_col:
                largest_col = patch_col
            if patch_row > largest_row:
                largest_row = patch_row
            #insert patch into predicted scene
            complete_pred_mask[
                (patch_row - 1) * pr_patch_size_rows: patch_row * pr_patch_size_rows,
                (patch_col - 1) * pr_patch_size_cols: patch_col * pr_patch_size_cols
            ] = predicted_patch
        
        #fit size of prediction to actual ground truth
        final_mask = unzeropad(complete_pred_mask, gt, largest_col, largest_row)
        
        complete_folder = f'entire_masks_{preds_folder}'
        os.makedirs(os.path.join(preds_folder_root, complete_folder), exist_ok=True)
        
        baseFileName = f'{scene_id}.TIF'
        path = os.path.join(preds_folder_root, complete_folder, baseFileName)
        cv2.imwrite(path, final_mask.astype(np.uint8) * 255)
        #order of return values:: [precision*100, recall*100, specificity*100, jaccard_index*100, accuracy*100]
        new_QE = QE_calcul(final_mask, gt)
        if QE == []:
            QE = new_QE
        else:
            QE[0] = QE[0] + new_QE[0]
            QE[1] = QE[1] + new_QE[1]
            QE[2] = QE[2] + new_QE[2]
            QE[3] = QE[3] + new_QE[3]
            QE[4] = QE[4] + new_QE[4]

    print(f'average evaluators over {n + 1} scenes are:\n')
    print("Precision (TrueTrue/(TrueTrue+FalseTrue)): ", QE[0]/(n+1))
    print("Recall (TrueTrue/(TrueTrue+FalseFalse)): ", QE[1]/(n+1))
    print("Specificity (TrueFalse/(TrueFalse+FalseTrue)): ", QE[2]/(n+1))
    print("Jaccard: ", QE[3]/(n+1))
    print("accuracy ((TrueTrue+TrueFalse)/(numberOfPredictions)): ", QE[4]/(n+1))
    
    # Saving the average of evaluators in a text file
    txt_baseFileName = f'numerical_results_{complete_folder}.txt'
    txtpath = os.path.join(preds_folder_root, txt_baseFileName)

    with open(txtpath, 'w') as file:
        file.write(f'Threshold=\n{thresh:.3f}\n\n')
        file.write('Precision, Recall, Specificity, Jaccard, Overall Accuracy\n')
        file.write(f'{QE[0]/(n+1):.6f}, {QE[1]/(n+1):.6f}, {QE[2]/(n+1):.6f}, {QE[3]/(n+1):.6f}, {QE[4]/(n+1):.6f}\n')

    print(f'Evaluators saved to {txtpath}')
