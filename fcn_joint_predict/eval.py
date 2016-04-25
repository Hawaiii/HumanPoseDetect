from __future__ import division
import sys

caffe_root = '/home/mengxin1/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(5)

net = caffe.Net('/home/mengxin1/HumanPoseDetect/fcn_joint_predict/deploy_finetune.prototxt', '/home/mengxin1/HumanPoseDetect/fcn_joint_predict/models/model_iter_10000.caffemodel', caffe.TEST)

#create transformer for the input called 'data'
mu = np.array([104, 116, 122]);
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # move image channels to outermost
transformer.set_mean('data', mu)           # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)     # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR

num_joings = 16
# Read ground truth labels
import csv
gt_file = open('/home/mengxin1/HumanPoseDetect/preprocess/val_label_resized.txt','rb')
gt_reader = csv.reader(gt_file, delimiter=' ')
gt_list = list(gt_reader)
for gtl in gt_list:
	gtl[1:] = [int(elem) for elem in gtl[1:]]

# Read resized bounding box sizes
bbox_file = open('/home/mengxin1/HumanPoseDetect/preprocess/val_bbox.txt','rb')
bbox_reader = csv.reader(bbox_file, delimiter=' ')
bbox_list = list(gt_reader)
for bboxl in bbox_list:
	bboxl[1:] = [float(elem) for elem in bboxl[1:]]
#next(itertools.islice(csv.reader(f), N, None))

# Predict for each joint
correct_prediction = np.zeros(1,num_joints)
tot_prediction = np.zeros(1,num_joints)
imageDirectory = '/home/mengxin1/mpii_human_pose_v1_images_resized/'

import math
alpha = 0.1
# For each image in val
for i, gtl in gt_list:
	# load image and transform
	imageFile = gtl[0]
	image = caffe.io.load_image(imageDirectory+imageFile);
	transformed_image = transformer.preprocess('data', image);

	# forward propagate
	net.blobs['data'].data[0] = transformed_image;
	out = net.forward()
	output_scoreMap = out['score'][0]
	# num_joints = output_scoreMap.shape[0]
	for j in range(num_joints):
		predict_joint_idx = np.argmax(output_scoreMap[j])
		predict_joint_sub = np.unravel_index(predict_joint_idx, output_scoreMap[j].shape)
		# print 'predicted joint position ', j, ': '
		# print  predict_joint_sub[1], predict_joint_sub[0], output_scoreMap[j][predict_joint_sub[0]][predict_joint_sub[1]] 
		xdiff = predict_joint_sub[1] - gtl[1+3*j]
		ydiff = predict_joint_sub[0] - gtl[1+3*j+1]
		tot_diff = math.sqrt(xdiff**2 + ydiff**2)
		thres = alpha * max(bbox_list[i][1], bbox_list[i][2])
		if tot_diff < thres:
			correct_prediction[1,j] += 1
		tot_prediction[1,j] += 1

	#debug
	if i == 0:
		print xdiff, ydiff, tot_diff, thres
		break	

# Calculate PCK
for j in range(num_joints):
	print "PCK of joint",j,":", 1.0*correct_prediction[1,j]/ tot_prediction[1,j]
