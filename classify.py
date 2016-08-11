#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from PIL import Image

plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate

if (int(os.environ.get('CAFFE_CPU_MODE'))):
    print "CPU"
    caffe.set_mode_cpu()
else:
    print "GPU"
    caffe.set_mode_gpu() 

net_root = 'caffe-fcn/fcn-8s'
model_def = net_root + '/deploy.prototxt'
model_weights = net_root + '/fcn-8s-pascalcontext.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)
image = Image.open(sys.argv[1])
## avoiding caffe.io.Transformer, as in https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/infer.py as recommended @ https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/deploy.prototxt#L7
# transformer.set_raw_scale('data', 255) # rescale from [0, 1] to [0, 255]
in_ = np.array(image, dtype=np.float32) # to f32
in_ = in_[:,:,::-1] # RGB -> BGR # transformer.set_channel_swap('data', (2, 1, 0))
in_ -= np.array((104.00698793,116.66876762,122.67891434)) # sub. mean # transformer.set_mean('data', mu)
in_ = in_.transpose((2,0,1)) # HxWxC -> CxHxW  # transformer.set_transpose('data', (2, 0, 1))
## reshape the net input according to input image:
net.blobs['data'].reshape(1, *in_.shape) # py-magic: arg unpacking of tuple consisting of C,H,W : https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists  http://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = in_


print('Running image through net.')
output = net.forward()
print('Done.')

score = output['score'][0]
classed = np.argmax(score, axis=0)
names = dict()
all_labels = ["0: Background"] + open(net_root + '/legend.txt').readlines()
scores = np.unique(classed)
labels = [all_labels[s] for s in scores]
num_scores = len(scores)


def rescore(c):
    """ rescore values from original score values (0-59) to values ranging from
    0 to num_scores-1 """
    return np.where(scores == c)[0][0]

rescore = np.vectorize(rescore)
painted = rescore(classed)

plt.figure(figsize=(10, 10))
plt.imshow(painted)
formatter = plt.FuncFormatter(lambda val, loc: labels[val])
plt.colorbar(ticks=range(0, num_scores), format=formatter)
plt.clim(-0.5, num_scores - 0.5)

plt.savefig(sys.argv[2])
