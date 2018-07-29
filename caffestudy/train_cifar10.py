import numpy as np
import matplotlib.pyplot as plt
import sys, os, caffe

caffe_root = '/home/heyanguang/caffecode/caffe/'
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('examples/cifar10/cifar10_quick_solver.prototxt')

niter = 4000
test_interval = 200
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')
    
    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...', 'accuracy:', acc
        test_acc[it/test_interval] = acc

with open('../caffestudy/result.txt', 'wb') as f:
        f.write('train_loss ')
        for k in train_loss:
            f.write(str(k) + ' ')
        f.write('\n')
with open('../caffestudy/result.txt', 'ab') as f:
        f.write('test_acc ')
        for k in test_acc:
            f.write(str(k) + ' ')
        f.write('\n')
# np.savetxt("result.txt", train_loss);
# np.savetxt("result.txt", test_acc);


