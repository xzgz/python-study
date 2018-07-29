import numpy as np
import matplotlib.pyplot as plt
import caffe, cv2
caffe_root = '/media/xzgz/Ubuntu/Ubuntu/Caffe/SourceCode/'
import os
os.chdir(caffe_root)

if not os.path.isfile(caffe_root + 'examples/cifar10/cifar10_quick_iter_4000.caffemodel'):
    print("caffemodel is not exist...")
    
caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick.prototxt',
                caffe_root + 'examples/cifar10/cifar10_quick_iter_4000.caffemodel',
                caffe.TEST)

print net.blobs['data'].data.shape

im = caffe.io.load_image('examples/images/cat_gray.jpg')
# res = caffe.io.resize_image(im, (32, 32))
res = cv2.resize(im, (32, 32), cv2.INTER_CUBIC)
plt.rcParams['image.cmap'] = 'gray'
plt.figure()
plt.imshow(res)
plt.axis('off')
print 'res', res.shape

def convert_mean(bin_mean_file, npy_mean_file):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(bin_mean_file, 'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npy_mean_file, npy_mean)

bin_mean_file = caffe_root + 'examples/cifar10/mean.binaryproto'
npy_mean_file = caffe_root + 'examples/cifar10/mean.npy'
convert_mean(bin_mean_file, npy_mean_file)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(npy_mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)  
transformer.set_channel_swap('data', (2, 1, 0))
net.blobs['data'].data[...] = transformer.preprocess('data', res)
inputData = net.blobs['data'].data

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.title("origin")
# plt.imshow(res)
# plt.axis('off')
# 
# plt.subplot(1, 2, 2)
# plt.imshow(transformer.deprocess('data', inputData[0]))
# plt.title("subtract mean")
# plt.axis('off')

net.forward()
for k, v in net.blobs.items():
    print k, v.data.shape

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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

# show_feature(net.blobs['conv1'].data[0])
# plt.title('conv1')
# show_feature(net.params['conv1'][0].data.reshape(32 * 3, 5, 5))
# plt.title('conv1_params')
print 'net.blobs[conv1]', net.blobs['conv1'].data.shape
print 'net.params[conv1]', net.params['conv1'][0].data.shape

# show_feature(net.blobs['pool1'].data[0])
# plt.title('pool1')
print 'net.blobs[pool1]', net.blobs['pool1'].data.shape
# 
# show_feature(net.blobs['conv2'].data[0], padval = 0.5)
# plt.title('conv2')
# show_feature(net.params['conv2'][0].data.reshape(32 ** 2, 5, 5))
# plt.title('conv2_params')
print net.blobs['conv2'].data.shape
print net.params['conv2'][0].data.shape
# 
# show_feature(net.blobs['conv3'].data[0], padval = 0.5)
# plt.title('conv3')
# show_feature(net.params['conv3'][0].data.reshape(64 * 32, 5, 5)[:1024])
# plt.title('conv3_params')
# print net.blobs['conv3'].data.shape
# print net.params['conv3'][0].data.shape

# show_feature(net.blobs['pool3'].data[0], padval = 0.2)
# plt.title('pool3')
# print net.blobs['pool3'].data.shape
feat = net.blobs['prob'].data[0]
print feat, feat.flat
plt.figure()
plt.plot(feat.flat)

# plt.show()



