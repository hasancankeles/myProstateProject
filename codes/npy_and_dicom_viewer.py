import numpy as np
import os
from PIL import Image
import nrrd
import matplotlib.pyplot as plt
import pydicom
import itertools
import pydicom.data

dcmpaths = []
dcmpathsdx = []
npy_3t_filess =[]
npy_3t_files =[]
npy_dx_filess =[]
npy_dx_files =[]

# Full path of the DICOM file is passed in base
def dicom_path_finder(base = r"/Users/hasancankeles/Prostate/test/manifest-WTWyB8IJ8830296727402453766/Prostate-3T/"):
    fake_dcm_files = [f for f in os.listdir(base) if not f.endswith('.dcm')]
    fake_dcm_files.sort()
    #print(fake_dcm_files)
    for i in fake_dcm_files:
        pth = i  + '/'
        #print(i)
        try:
            dicom_path_finder(base + pth)
        except:
            continue
    
    global dcmpaths
    #print("a")       
    
    for f in os.listdir(base):
        if f.endswith('.dcm'):
            x = base + f
            dcmpaths.append(x)
    #dcm_files = [f for f in os.listdir(base) if  f.endswith('.dcm')]
    #dcm_files.sort()

        
 
# Full path of the DICOM file is passed in base
def dicomdx_path_finder(base = r"/Users/hasancankeles/Prostate/test/manifest-WTWyB8IJ8830296727402453766/PROSTATE-DIAGNOSIS/"):
    fake_dcm_files = [f for f in os.listdir(base) if not f.endswith('.dcm')]
    fake_dcm_files.sort()
    #print(fake_dcm_files)
    for i in fake_dcm_files:
        pth = i  + '/'
        #print(i)
        try:
            dicomdx_path_finder(base + pth)
        except:
            continue
    
    global dcmpathsdx
    #print("a")       
    
    for f in os.listdir(base):
        if f.endswith('.dcm'):
            x = base + f
            dcmpathsdx.append(x)
    #dcm_files = [f for f in os.listdir(base) if  f.endswith('.dcm')]
    #dcm_files.sort()       




######main

input_dir = "/Users/hasancankeles/Prostate/prostate_segmentation-master/data/test/2d/"



def extract_number(filename):
        return int(filename.split("-")[-1].split("_")[0]),int(filename.split("-")[-1].split("_")[1].split(".")[0])


npy_3t_filess = [f for f in os.listdir(input_dir) if f.endswith('.npy') and f.startswith("3T")]
npy_3t_files = sorted(npy_3t_filess, key=extract_number)
npy_dx_filess = [f for f in os.listdir(input_dir) if f.endswith('.npy') and f.startswith("Dx")]
npy_dx_files = sorted(npy_dx_filess, key=extract_number)


dicom_path_finder()
dicomdx_path_finder()

def dcm_sort(filename):
    return int(filename.split("/")[7].split("-")[2]),int(filename.split("/")[10].split("-")[1].split(".")[0])


dcmpaths = sorted(dcmpaths, key=dcm_sort)


for i in dcmpaths:
    print(i.split("/")[7],i.split("/")[10])
    
#######dcmdx

dcmpathsdx = sorted(dcmpathsdx, key=dcm_sort)


for i in dcmpathsdx:
    print(i.split("/")[7],i.split("/")[10])



for i,npy3t in zip(dcmpaths,npy_3t_files):
        

        fig = plt.figure(figsize=(10, 7))

        drc = input_dir + npy3t
        arr = np.load(drc)

        fig.add_subplot(1, 2, 1)
        plt.imshow(arr)
        plt.title(npy3t)
        
        
        ds = pydicom.dcmread(i)
        
        fig.add_subplot(1, 2, 2)
        
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)  # set the color map to bone
        tmp = i.split("/")[7],i.split("/")[10]
        plt.title(tmp)

        plt.show()

'''

for i,npydx in zip(dcmpathsdx,npy_dx_files):
        

        fig = plt.figure(figsize=(10, 7))

        drc = input_dir + npydx
        arr = np.load(drc)

        fig.add_subplot(1, 2, 1)
        plt.imshow(arr)
        plt.title(npydx)
        
        
        ds = pydicom.dcmread(i)
        
        fig.add_subplot(1, 2, 2)
        
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)  # set the color map to bone
        tmp = i.split("/")[7],i.split("/")[10]
        plt.title(tmp)

        plt.show()'''