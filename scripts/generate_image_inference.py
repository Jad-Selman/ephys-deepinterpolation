#this script generates the images of the data resulting from inference, whereby the employed models in inference
# are genrated for the purpose of assessing the effect of different batches employed in testing and validation phase on
# the quantitative and qualitative output of the models. 

import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
import h5py
import numpy as np
os.listdir('.')
fn= 'ephys_tiny_continuous_deep_interpolation.h5'
f= h5py.File(fn)
tsurf = f['data']
tsurf
di_traces = tsurf[:,:,0]
di_traces.shape
plot= plt.imshow(di_traces.T, 
                    origin='lower', 
                    vmin=-50, 
                    vmax=50, 
                    cmap='RdGy',
                    aspect='auto')
plt.xlabel('Sample Index')
plt.ylabel('Acquisition Channels')
#plt.ylim(0,386)
#plt.xlim(0,400)
#matplotlib.pyplot.annotate('alpha', xy= (1,1), xytext=(3,3), xycoords='data', textcoords=None, arrowprops=None, annotation_clip=None)
matplotlib.pyplot.colorbar(label= 'μV', shrink=0.25)
plt.title("1000samples_28Epoch")
matplotlib.pyplot.savefig('1000samples_28Epoch')
