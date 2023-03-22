import os

folder_path_npy = "/Users/hasancankeles/Downloads/Training/npy"
folder_path_dcm = "/Users/hasancankeles/Prostate/dcmdata/train"
npyfiles = []
missingfiles = []
for i in os.listdir(folder_path_npy):
    npyfiles.append(i)

for i in os.listdir(folder_path_dcm):
    if not i in npyfiles:
        missingfiles.append(i)
        print(i)
  
print(len(missingfiles))