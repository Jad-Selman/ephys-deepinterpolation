import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib
from scipy import ndimage
import scipy
import numpy
file = 'ephys_tiny_continuous.dat2'
dtype = 'int16'
num_channels = 384
with open(file) as f:
    nsamples =(os.fstat(f.fileno()).st_size) // (num_channels * np.dtype(dtype).itemsize)
traces = np.memmap(file, np.dtype(dtype), mode='r', offset=0, shape=(nsamples, num_channels))
#traces= numpy.array(0.195*traces) [This multiplication is important for making the output values in microvolts]
plot= plt.imshow(traces[2030:2430].T, 
                    origin='lower', 
                    vmin=-80, 
                    vmax=80, 
                    cmap='RdGy',
                    aspect='auto')
plt.xlabel('Sample Index')
plt.ylabel('Acquisition Channels')
matplotlib.pyplot.colorbar(shrink=0.25)
plt.title("Original Raw Data")
matplotlib.pyplot.savefig(' Original Sample Data')
