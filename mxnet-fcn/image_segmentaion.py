# pylint: skip-file
import numpy as np
import mxnet as mx
from PIL import Image
import matplotlib.pyplot as plt
import sys

img = sys.argv[1]
seg = sys.argv[2]
model_previx = "mxnet-fcn/FCN8s_VGG16"
epoch = 19
ctx = mx.gpu(0)

def get_data(img_path):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)
    fcnxs_args["data"] = mx.nd.array(get_data(img), ctx)
    data_shape = fcnxs_args["data"].shape
    label_shape = (1, data_shape[2]*data_shape[3])
    fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
    exector = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_args)
    exector.forward(is_train=False)
    output = exector.outputs[0]
    out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
    classed = out_img

    net_root = 'caffe-fcn/fcn-8s'
    names = dict()
    all_labels = ["0: Background"] + open(net_root + '/legend.txt').readlines() # legent is from PASCAL-Context_59 (http://www.cs.stanford.edu/~roozbeh/pascal-context/), but first 20 labels are identical to VOC2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html), which was used to train the FCN8s_VGG16 used here
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

if __name__ == "__main__":
    main()
