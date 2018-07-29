import numpy as np
import matplotlib.pyplot as plt
import caffe
caffe_root = '/media/xzgz/Ubuntu/Ubuntu/Caffe/SourceCode/'
import os
os.chdir(caffe_root)

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Net(caffe_root + 'examples/mnist/lenet_train_test.prototxt',
                caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel',
                caffe.TRAIN)
for k, v in net.params.items():
    print k, v[0].data.shape

def show_feature(data, padsize = 1, padval = 0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data)
    plt.axis('off')

weight1 = net.params["conv1"][0].data
print weight1.shape
show_feature(weight1[:, 0])
plt.title('conv1')

weight2 = net.params["conv2"][0].data
print weight2.shape
show_feature(weight2.reshape(50 * 20, 5, 5))
plt.title('conv2')

weight2_m = np.mean(weight2, 1)
show_feature(weight2_m)
plt.title('weight2_m')


plt.show()


