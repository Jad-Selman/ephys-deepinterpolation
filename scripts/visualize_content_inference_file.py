import os
import matplotlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy
os.listdir('.')
fn= 'ephys_tiny_continuous_deep_interpolation.h5'
f= h5py.File(fn)
list(f.keys())
tsurf = f['data']
np.squeeze(tsurf)
di_traces = tsurf[:,:,0]
#di_traces=0.195*numpy.array(di_traces)
plot= plt.imshow(di_traces.T, 
                    origin='lower', 
                    vmin=-300, 
                    vmax=300, 
                    cmap='RdGy',
                    aspect='auto')
plt.xlabel('Sample Index')
plt.ylabel('Acquisition Channels')
#plt.ylim(0,386)
#plt.xlim(0,400)
#matplotlib.pyplot.annotate('alpha', xy= (1,1), xytext=(3,3), xycoords='data', textcoords=None, arrowprops=None, annotation_clip=None)
matplotlib.pyplot.colorbar(shrink=0.25)
plt.title("Insert figure title here")
matplotlib.pyplot.savefig('Insert saved figure name here')