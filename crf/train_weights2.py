import h5py
import random
import numpy as np
from pystruct.learners import OneSlackSSVM
from pystruct.models import BinaryClf
import loadCNNfeature as ldf
import copy

class Pt:
	x = None
	y = None

	def __init__(self, x, y):
		self.x = x
		self.y = y

	@staticmethod
	def dist(pt0, pt1):
		# returns numpy array of dimension (4,): [dx, dy, dx^2, dy^2]
		dx = pt0.x - pt1.x
		dy = pt0.y - pt1.y
		return np.array([dx, dy, dx*dx, dy*dy])

def get_pose(joint_gt, n_joint):
	# joint_gt.shape == (3*n_joint,)
	p = []
	for i in xrange(n_joint):
		p.append(Pt(joint_gt[3*i], joint_gt[3*i+1]))
	return tuple(p)

def random_pose_pos(joint_gt, n_joint, wdw_half):
	# Returns a tuple of 16 Pt
	rp = []
	for i in xrange(n_joint):
		x = joint_gt[3*i]
		y = joint_gt[3*i+1]
		x_j = random.randint(max(0, x-wdw_half), min(511, x+wdw_half))
		y_j = random.randint(max(0, y-wdw_half), min(511, y+wdw_half))
		rp.append(Pt(x_j, y_j))
	return tuple(rp)

def random_pose_neg(n_cls_config, joint_gt, n_joint, wdw_half, pts_1d, score):
	# Returns a tuple of 16 Pt
	# Selected with probability proportional to the score
	# Points too close to ground truth are rejected
	rp = []
	for i in xrange(n_cls_config):
		rp.append([])
	for i in xrange(n_joint):
		score_dup = score[i].copy()
		x_gt = int(joint_gt[3*i])
		y_gt = int(joint_gt[3*i+1])
		min_score = score_dup.min()
		#print x_gt, y_gt, wdw_half, max(0, y_gt-wdw_half), min(512, y_gt+wdw_half+1)
		for y in xrange(max(0, y_gt-wdw_half), min(512, y_gt+wdw_half+1)):
			for x in xrange(max(0, x_gt-wdw_half), min(512, x_gt+wdw_half+1)):
				score_dup[y,x] = min_score
#		print score_dup[max(0, y_gt-wdw_half), max(0, x_gt-wdw_half)], score_dup.min(), score_dup.max(), pts_1d[y_gt*512+x_gt]
		score_1d = make_score1d(score_dup)
		p = np.random.choice(pts_1d, n_cls_config, replace=False, p=score_1d)

#		while too_close(p, joint_gt[3*i], joint_gt[3*i+1], wdw_half):
#			print p, "too_close to (", joint_gt[3*i], ",", joint_gt[3*i+1],")"
#			print p, "prob", score_1d[p[1]*512+p[0]], "max",score_1d.max()
#			p = np.random.choice(pts_1d, p=score_1d)
		for j in xrange(n_cls_config):
			rp[j].append(Pt(p[j][0], p[j][1]))
	for i in xrange(n_cls_config):
		rp[i] = tuple(rp[i])
	return rp

def too_close(point, x, y, wdw_half):
	if abs(point[0]-x) <= wdw_half and  abs(point[1]-y) <= wdw_half:
		return True
	return False

def calc_feature(pose, score, feature_len, n_joint):
	assert(len(pose) == 16)
	# Returns 1xfeature_len numpy array
	f = np.zeros((1, feature_len))

	cnt = 0
	# Add Unary features from score
	for i in xrange(n_joint):
		f[0, cnt] = score[i, pose[i].y, pose[i].x]
		cnt += 1
	# Add Binary features
	for i in xrange(n_joint):
		for j in xrange(n_joint):
			if i == j:
				continue
			f[0, cnt:cnt+4] = Pt.dist(pose[i], pose[j])
			cnt += 4
	assert(feature_len == cnt) #
	return f

def format_weights(weight_arr, n_joint):
	# weight_arr is (feature_len,) numpy array
	# Returns a dictionary, where the weights could be looked up by checking
	# weight[i], weight[(i,j)]
	
	weight = {}
	# Unary
	for i in xrange(n_joint):
		weight[i] = weight_arr[i]
	# Binary
	cnt = n_joint
	for i in xrange(n_joint):
		for j in xrange( n_joint):
			if i == j:
				continue
			weight[(i,j)] = weight_arr[cnt]
			cnt += 1
	return weight

def make_pts1d():
	# Return 1 x (512*512) numpy array containing position tuples
	pos_arr = np.empty((512*512,),dtype=object)
	cnt = 0
	for y in xrange(512):
		for x in range(512):
			pos_arr[cnt] = (x,y)
			cnt += 1
	print "pts1d shape", pos_arr.shape
	return pos_arr

def make_score1d(score):
	# Assuming score1d has only positive entries
	score1d = score.reshape((-1,))
	#if score1d.min() < 0:
	score1d = -score1d.min()*np.ones(score1d.shape) + score1d
	tot = np.sum(score1d)
	if tot != 0:
		score1d = score1d / tot
	assert(score1d.min() == 0)
	assert(score1d.max() <= 1)
	return score1d


# load training features and goundtruth
n_joint = 16
feature_len = n_joint + 16*15*4
pos_wdw_half = 5
neg_wdw_half = 12
pts1d = make_pts1d()

n_cls_config = 5000
X_train = np.array([])
Y_train = np.array([])

feature_file_path = "../fcn_joint_predict/train_score_fullHuman.h5"
feature_file = h5py.File(feature_file_path)
gp_names = feature_file.keys()
num_batches = len(gp_names) / 2
jointLocation = np.array([])
for i in range(num_batches):
#for i in range(1):
   print "batch: ", i
   joint_loc = feature_file['/'+gp_names[i]]
   joint_loc = np.array(joint_loc)
   batch_size = joint_loc.shape[0]
   joint_loc = joint_loc.reshape((batch_size, 48))
   score = feature_file['/'+gp_names[i+num_batches]]
   score = np.array(score)
   for image in range(batch_size):
     print "image: ", image
     pos_config = []
     pos_config.append(get_pose(joint_loc[image, :], n_joint))
     for j in xrange(n_cls_config-1):
        pos_config.append(random_pose_pos(joint_loc[image, :], n_joint, pos_wdw_half))
     neg_config = []
     #for j in xrange(n_cls_config):
     #   neg_config.append(random_pose_neg(joint_loc[image, :], n_joint, neg_wdw_half, pts1d, score[image]))
     neg_config = random_pose_neg(n_cls_config, joint_loc[image,:], n_joint, neg_wdw_half, pts1d, score[image])
     for p in pos_config:
        x_train = calc_feature(p, score[image], feature_len, n_joint)
        if len(X_train) == 0:
          X_train = x_train
          Y_train = np.ones((1))
        else:
          X_train = np.concatenate((X_train, x_train), axis = 0)
          Y_train = np.concatenate((Y_train, np.ones((1))), axis = 0)
     for p in neg_config:
        x_train = calc_feature(p, score[image], feature_len, n_joint)
        if len(X_train) == 0:
          X_train = x_train
          Y_train = np.ones((1))*-1
        else:
          X_train = np.concatenate((X_train, x_train), axis = 0)
          Y_train = np.concatenate((Y_train, np.ones((1))*-1), axis = 0)

del joint_loc
del score
del x_train
del pos_config
del neg_config
# Train SVM
smodel = BinaryClf(feature_len)
svm = OneSlackSSVM(smodel)
print("fitting svm...")
print X_train.shape, Y_train.shape
svm.fit(X_train, Y_train)

# Write weights
weights = format_weights(svm.w, n_joint)

del X_train
del Y_train

# inference
print "inference..."
val_feature_file_path = "../fcn_joint_predict/val_score_fullHuman.h5"
val_feature_file = h5py.File(val_feature_file_path)
gp_names = val_feature_file.keys()
num_batches = len(gp_names) / 2
jointLocation_gt = np.array([])
jointLocation_pd = np.array([])
max_itr = 4
for i in range(1):
   joint_location_gt = val_feature_file['/'+gp_names[i]]
   joint_location_gt = np.array(joint_location_gt)
   batch_size = joint_location_gt.shape[0]
   joint_location_gt = joint_location_gt.reshape((batch_size,48))  # joint_location_gt.shape=[batch_size, 48]
   if len(jointLocation_gt) == 0:
     jointLocation_gt = joint_location_gt
   else:
     jointLocation_gt = np.vstack((jointLocation_gt,joint_location_gt))  # jointLocation_gt.shape = [num_images, 48]
   scores = val_feature_file['/'+gp_names[i+num_batches]]
   scores = np.array(scores)   # scores.shape = [batch_size,16,512,512]
   print jointLocation_gt[1]
   #for image in range(batch_size):
   for image in range(8):
      score = scores[image]   # score.shape = [16,512,512]
      joint_location_pd = np.zeros((16, 2)) # each row is (x,y)
      for joint in range(16):
         joint_pd_idx = np.argmax(score[joint])
         joint_pd_sub = np.unravel_index(joint_pd_idx,score[joint].shape)
         joint_location_pd[joint][0] = joint_pd_sub[1]
         joint_location_pd[joint][1] = joint_pd_sub[0]
      print joint_location_pd
      for itr in range(max_itr):
         print "iteration: ", itr
         for joint in range(16):
         #for joint in range(1):
            print "joint: ", joint
            joints = copy.deepcopy(joint_location_pd)
            Energy = np.zeros((512,512))
            for x in range(512):
               joints[joint][0] = x
               for y in range(512):
                  joints[joint][1] = y
                  pose = []
                  for i in xrange(16):
                     pose.append(Pt(joints[i][0], joints[i][1]))
                  pose = tuple(pose)
                  feature = calc_feature(pose, score, feature_len, 16)
                  Energy[x][y] = np.inner(svm.w, feature)
            update_joint_idx = np.argmax(Energy)
            update_joint_sub = np.unravel_index(update_joint_idx, Energy.shape)
            joint_location_pd[joint][0] = update_joint_sub[0]
            joint_location_pd[joint][1] = update_joint_sub[1]  
      joint_location_pd = joint_location_pd.reshape(1, 16*2)
      if len(jointLocation_pd) == 0:
        jointLocation_pd = joint_location_pd
      else:
        jointLocation_pd = np.vstack((jointLocation_pd,joint_location_pd))   # jointLocation_pd.shape = [num_images, 32], [x,y,x,y...]

#print(jointLocation_pd[0])
# evaluate

np.save("predicted_joints_loc", jointLocation_pd)
 

                   
      

   
   
