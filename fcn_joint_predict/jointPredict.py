from __future__ import division
import sys

caffe_root = '/home/mengxin1/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(5)

net = caffe.Net('/home/mengxin1/HumanPoseDetect/fcn_joint_predict/deploy_finetune.prototxt', '/home/mengxin1/HumanPoseDetect/fcn_joint_predict/models/model_iter_20000.caffemodel', caffe.TEST)

#create transformer for the input called 'data'
mu = np.array([104, 116, 122]);
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # move image channels to outermost
transformer.set_mean('data', mu)           # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)     # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR

# load image and transform
imageDirectory = '/home/mengxin1/mpii_human_pose_v1_images_resized/'
imageFile = '032078006.jpg'
image = caffe.io.load_image(imageDirectory+imageFile);
transformed_image = transformer.preprocess('data', image);

# forward propagate
net.blobs['data'].data[0] = transformed_image;
out = net.forward()
output_scoreMap = out['score'][0]
num_joints = output_scoreMap.shape[0]
for i in range(num_joints):
   predict_joint_idx = np.argmax(output_scoreMap[i])
   predict_joint_sub = np.unravel_index(predict_joint_idx, output_scoreMap[i].shape)
   print 'predicted joint position ', i, ': '
   print  predict_joint_sub[1], predict_joint_sub[0] 
