from __future__ import division
import sys

caffe_root = '/home/mengxin1/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(3)

# caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver_finetune.prototxt')
solver.net.copy_from('/home/mengxin1/HumanPoseDetect/fcn_joint_predict/bvlc_reference_caffenet.caffemodel')

niter = 30000
train_loss = np.zeros(niter)

f = open('log.txt', 'w')

for it in range(niter): 
    solver.step(1)
    scoreMap_shape=solver.net.blobs['score'].shape
    scoreMap_size = scoreMap_shape[2]*scoreMap_shape[3] # height*width 
    train_loss[it] = solver.net.blobs['loss'].data
    f.write('{0: f}\n'.format(train_loss[it]/scoreMap_size))

f.close()

# solver.step(80000)


