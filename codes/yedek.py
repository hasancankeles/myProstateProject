import numpy as np
import os
from PIL import Image
import nrrd
import matplotlib.pyplot as plt
import pydicom
import itertools
import pydicom.data
from scipy import ndimage
dcmpathstrain = []
dcmpathsdxtrain = []
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
    for i in fake_dcm_files:
        pth = i  + '/'
        try:
            dicom_path_finder(base + pth)
        except:
            continue
    global dcmpaths
    for f in os.listdir(base):
        if f.endswith('.dcm'):
            x = base + f
            dcmpaths.append(x)
            
def train_dicom_path_finder(base = r"/Users/hasancankeles/Prostate/train/manifest-ZqaK9xEy8795217829022780222/Prostate-3T/"):
    fake_dcm_files = [f for f in os.listdir(base) if not f.endswith('.dcm')]
    fake_dcm_files.sort()
    for i in fake_dcm_files:
        pth = i  + '/'
        try:
            train_dicom_path_finder(base + pth)
        except:
            continue
    global dcmpathstrain
    for f in os.listdir(base):
        if f.endswith('.dcm'):
            x = base + f
            dcmpathstrain.append(x)
    
    

        
 
# Full path of the DICOM file is passed in base
def dicomdx_path_finder(base = r"/Users/hasancankeles/Prostate/test/manifest-WTWyB8IJ8830296727402453766/PROSTATE-DIAGNOSIS/"):
    fake_dcm_files = [f for f in os.listdir(base) if not f.endswith('.dcm')]
    fake_dcm_files.sort()
    for i in fake_dcm_files:
        pth = i  + '/'
        try:
            dicomdx_path_finder(base + pth)
        except:
            continue
    global dcmpathsdx    
    for f in os.listdir(base):
        if f.endswith('.dcm'):
            x = base + f
            dcmpathsdx.append(x)

def train_dicomdx_path_finder(base = r"/Users/hasancankeles/Prostate/train/manifest-ZqaK9xEy8795217829022780222/PROSTATE-DIAGNOSIS/"):
    fake_dcm_files = [f for f in os.listdir(base) if not f.endswith('.dcm')]
    fake_dcm_files.sort()
    for i in fake_dcm_files:
        pth = i  + '/'
        try:
            train_dicomdx_path_finder(base + pth)
        except:
            continue
    global dcmpathsdxtrain    
    for f in os.listdir(base):
        if f.endswith('.dcm'):
            x = base + f
            dcmpathsdxtrain.append(x)
     




####################main

input_dir = "/Users/hasancankeles/Prostate/prostate_segmentation-master/data/test/2d/"



def extract_number(filename):
        return int(filename.split("-")[-1].split("_")[0]),int(filename.split("-")[-1].split("_")[1].split(".")[0])


npy_3t_filess = [f for f in os.listdir(input_dir) if f.endswith('.npy') and f.startswith("3T")]
npy_3t_files = sorted(npy_3t_filess, key=extract_number)
npy_dx_filess = [f for f in os.listdir(input_dir) if f.endswith('.npy') and f.startswith("Dx")]
npy_dx_files = sorted(npy_dx_filess, key=extract_number)


train_dicom_path_finder()
train_dicomdx_path_finder()
dicom_path_finder()
dicomdx_path_finder()


def dcm_sort(filename):
    return int(filename.split("/")[7].split("-")[2]),int(filename.split("/")[10].split("-")[1].split(".")[0])

######sorting
dcmpaths = sorted(dcmpaths, key=dcm_sort)
dcmpathsdxtrain = sorted(dcmpathsdxtrain, key=dcm_sort)
dcmpathstrain = sorted(dcmpathstrain, key=dcm_sort)
dcmpathsdx = sorted(dcmpathsdx, key=dcm_sort)


for i in dcmpathsdxtrain:
    #print(i.split("/")[7],i.split("/")[10])
    pass

for i in dcmpathstrain:
    #print(i.split("/")[7],i.split("/")[10])
    pass


for i in dcmpaths:
    #print(i)
    #print(i.split("/")[7],i.split("/")[10])
    pass
    
#######dcmdx



for i in dcmpathsdx:
    #print(i)
    #print(i.split("/")[7],i.split("/")[10])
    pass





###########         DCIM saving 
    
"""
for i in dcmpathstrain:
    ds = pydicom.dcmread(i)
    pixel_array = ds.pixel_array
    patient = i.split("/")[7]
    dcimno = i.split("/")[10].split(".")[0]
    '''
    path = os.path.join("/Users/hasancankeles/Prostate/dcmdata/train/", patient)
    try:
        os.mkdir(path)
    except:
        pass
    '''
    np.save("/Users/hasancankeles/Prostate/dcmdata/train/" + patient + "_" + dcimno, pixel_array)
 


for i in dcmpaths:
    ds = pydicom.dcmread(i)
    pixel_array = ds.pixel_array
    patient = i.split("/")[7]
    dcimno = i.split("/")[10].split(".")[0]
    '''
    path = os.path.join("/Users/hasancankeles/Prostate/dcmdata/test/", patient)
    try:
        os.mkdir(path)
    except:
        pass
    '''
    np.save("/Users/hasancankeles/Prostate/dcmdata/test/" + patient + "_" + dcimno, pixel_array)

for i in dcmpathsdx:
    ds = pydicom.dcmread(i)
    pixel_array = ds.pixel_array
    patient = i.split("/")[7]
    dcimno = i.split("/")[10].split(".")[0]
    '''
    path = os.path.join("/Users/hasancankeles/Prostate/dcmdata/test/", patient)
    try:
        os.mkdir(path)
    except:
        pass
    '''
    np.save("/Users/hasancankeles/Prostate/dcmdata/test/" + patient + "_" + dcimno, pixel_array)
    
   


for i in dcmpathsdxtrain:
    ds = pydicom.dcmread(i)
    pixel_array = ds.pixel_array
    patient = i.split("/")[7]
    dcimno = i.split("/")[10].split(".")[0]
    '''
    path = os.path.join("/Users/hasancankeles/Prostate/dcmdata/train/", patient)
    try:
        os.mkdir(path)
    except:
        pass
    '''
    np.save("/Users/hasancankeles/Prostate/dcmdata/train/" + patient + "_" + dcimno, pixel_array)
 
"""







#######data viewing

for i,npy3t in zip(dcmpaths,npy_3t_files):
        
        input_dir = "/Users/hasancankeles/Prostate/prostate_segmentation-master/data/test/2d/"

        fig = plt.figure(figsize=(10, 7))

        drc = input_dir + npy3t
        arr = np.load(drc)
        arr = ndimage.rotate(arr, 270)

        fig.add_subplot(1, 2, 2)
        plt.imshow(arr)
        plt.title(npy3t)
        
        
        ds = pydicom.dcmread(i)
        
        fig.add_subplot(1, 2, 1)
        
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)  # set the color map to bone
        tmp = i.split("/")[7],i.split("/")[10]
        plt.title(tmp)

        plt.show()



'''
for i,npydx in zip(dcmpathsdx,npy_dx_files):
        

        fig = plt.figure(figsize=(10, 7))

        drc = input_dir + npydx
        arr = np.load(drc)
        arr = ndimage.rotate(arr, 270)

        fig.add_subplot(1, 2, 2)
        plt.imshow(arr)
        plt.title(npydx)
        
        
        ds = pydicom.dcmread(i)
        
        fig.add_subplot(1, 2, 1)
        
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)  # set the color map to bone
        tmp = i.split("/")[7],i.split("/")[10]
        plt.title(tmp)

        plt.show()


'''

