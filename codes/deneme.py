import numpy as np
import os
from PIL import Image
import nrrd
import matplotlib.pyplot as plt
import pydicom
import itertools
import pydicom.data
from scipy import ndimage

x = np.load("/Users/hasancankeles/Prostate/prostate_segmentation-master/data/train/ProstateDx-01-0006_correctedLabels.npy")

plt.imshow(x[:,:,12])
plt.show()