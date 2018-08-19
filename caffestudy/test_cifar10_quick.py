import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import struct
# import cv2
caffe_root = '/media/xzgz/ubudata/Ubuntu/Caffe/caffe/'
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
import caffe

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'examples/resnet/resnet-56/trainval.prototxt',
                caffe_root + 'examples/resnet/resnet-56/snapshot/resnet_56_npc_iter_60000.caffemodel',
                caffe.TEST)
print "net.blobs['data'].data.dtype: ", net.blobs['data'].data.dtype
print "net.blobs['conv1'].data.dtype: ", net.blobs['conv1'].data.dtype
print "net.params['conv1'][0].data.dtype: ", net.params['conv1'][0].data.dtype


def get_image_label(filename, read_count, start_location):
    f1 = open(filename, 'rb')
    buf1 = f1.read()

    image = np.zeros((read_count, 32, 32, 3)).astype('uint8')
    label = np.zeros(read_count).astype('uint8')
    image_offset = 3073 * start_location
    for i in range(read_count):
        im_la = []
        temp = struct.unpack_from('B', buf1, image_offset)
        label[i] = np.array(temp)
        image_offset += 1
        temp = struct.unpack_from('3072B', buf1, image_offset)
        image[i] = np.reshape(temp, (3, 32, 32)).transpose(1, 2, 0).astype('uint8')
        image_offset += 3072
    return image, label


def convert_mean(binMean, npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_mean = arr[0]
    np.save(npyMean, npy_mean)


binMean = caffe_root + 'examples/cifar10/mean.binaryproto'
npyMean = caffe_root + 'examples/cifar10/mean.npy'
convert_mean(binMean, npyMean)

read_count = 40
start_location = 234
image_label = get_image_label('data/cifar10/test_batch.bin', read_count, start_location)
# label_name = open('data/cifar10/batches.meta.txt').readline()
# label_name = open('data/cifar10/batches.meta.txt').readlines()
# label_name = open('data/cifar10/batches.meta.txt').read().replace('\n', ' ').split()
label_name = open('data/cifar10/batches.meta.txt').read().split()
print 'label_name: ', label_name
print 'len(label_name): ', len(label_name)

# plt.imshow(image_label[0][6], cmap = 'gray')
# plt.imshow(image_label[0][8])
# plt.imshow(cv2.cvtColor(image_label[0][8], cv2.COLOR_BGR2RGB))
# for i in range(read_count):
#     plt.imshow(image_label[0][i])
#     plt.show()

print 'type(image_label[0]): ', type(image_label[0])
print 'type(image_label[1]): ', type(image_label[1])
print 'image_label[0].shape: ', image_label[0].shape
print 'image_label[0].dtype: ', image_label[0].dtype