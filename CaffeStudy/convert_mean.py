import numpy as np
import matplotlib.pyplot as plt
import os, caffe

caffe_root = '/media/xzgz/Ubuntu/Ubuntu/Caffe/SourceCode'
os.chdir(caffe_root)

# if len(sys.argv)!=3:
#     print "Usage: python convert_mean.py mean.binaryproto mean.npy"
#     sys.exit()
# 
# blob = caffe.proto.caffe_pb2.BlobProto()
# bin_mean = open( sys.argv[1] , 'rb' ).read()
# blob.ParseFromString(bin_mean)
# arr = np.array( caffe.io.blobproto_to_array(blob) )
# npy_mean = arr[0]
# np.save( sys.argv[2] , npy_mean )

plt.rcParams['image.cmap'] = 'gray'
blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open('examples/mnist/mean.binaryproto', 'rb').read()
blob.ParseFromString(bin_mean)
arr = np.array(caffe.io.blobproto_to_array(blob))
plt.imshow(arr[0, 0, :])
plt.axis('off')
plt.title('mean picture')
print arr.shape

npy_mean = arr[0]
np.save('mean.npy', npy_mean)


# plt.show()

