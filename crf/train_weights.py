import random
import numpy as np
from pystruct.learners import OneSlackSSVM
from pystruct.models import BinaryClf
import loadCNNfeature as ldf

class Pt:
	x = None
	y = None

	def __init__(self, x, y):
		self.x = x
		self.y = y

	@staticmethod
	def dist(pt0, pt1):
		# returns numpy array of dimension (4,): [dx, dy, dx^2, dy^2]
		dx = abs(pt0.x - pt1.x)
		dy = abs(pt0.y - pt1.y)
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
		y_j = random.randit(max(0, y-wdw_half), min(511, y+wdw_half))
		rp.append(Pt(x_j, y_j))
	return tuple(rp)

def random_pose_neg(joint_gt, n_joint, wdw_half, pts_1d, score_1d):
	# Returns a tuple of 16 Pt
	# Selected with probability proportional to the score
	# Points too close to ground truth are rejected
	rp = []
	for i in xrange(n_joint):
		p = np.random.choice(pts_1d, p=score_1d)
		while too_close(p, joint_gt[3*i], joint_gt[3*i+1], wdw_half):
			p = np.random.choice(pts_1d, p=score_1d)
			print p, "too_close to (", joint_gt[3*i], "," joint_gt[3*i+1],")"
		rp.append(p)
	return tuple(rp)

def too_close(point, x, y, wdw_half):
	if abs(point[0]-x) <= wdw_half:
		return True
	if abs(point[1]-y) <= wdw_half:
		return True
	return False

def calc_feature(pose, score, feature_len, n_joint):
	assert(len(pose) == 16)
	# Returns 1xfeature_len numpy array
	f = np.zeros((1, feature_len))

	cnt = 0
	# Add Unary features from score
	for i in xrange(n_joint):
		f[0, cnt] = score[pose[i].x, pose[i].y]
		cnt += 1
	# Add Binary features
	for i in xrange(n_joint):
		for j in xrange(i+1, n_joint):
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
		for j in xrange(i+1, n_joint):
			weight[(i,j)] = weight_arr[cnt]
			weight[(j,i)] - weight[(i,j)]
			cnt += 1
	return weight

def make_pts1d():
	# Return 1 x (512*512) numpy array containing position tuples
	pos_list = []
	for y in xrange(512):
		for x in range(512):
			pos_list.append((y,x))
	return np.array(pos_list).reshape(1, -1)

def make_score1d(score):
	# Assuming score1d has only positive entries
	score1d = score.reshape((1,-1))
	tot = abs(np.sum(score))
	if tot != 0:
		score1d = score1d / tot
	return score1d


# Load the ground truth
score, joint_location = ldf.load_full_person_data("../fcn_joint_predict/train_score_fullHuman.h5")
n_imgs = score.shape[0]
n_joint = 16
feature_len = n_joint + 120 #16 choose 2 binary
pos_wdw_half = 5
neg_wdw_half = 12
score1d = make_score1d()
pts1d = make_pts1d()

# Make features
X_train = np.zeros((2*n_cls_config*n_imgs, feature_len))
Y_train = np.ones((2*n_cls_config*n_imgs, 1))
# n_cls_config = 43046721 #3^16
n_cls_config = 40000
cnt = 0
for i in xrange(n_imgs):
	# Construct positive configurations
	pos_config = [] # list of tuples, each tuple contains 16 Pt, representing a pose
	# add the precise pose
	pos_config.append(get_pose(joint_location[i, :], n_joint))
	# add other possible joint configurations
	for j in xrange(n_cls_config-1):
		pos_config.append(random_pose_pos(joint_location[i,:], n_joint, pos_wdw_half))

	# Construct negative configurations
	neg_config = []
	for j in xrange(n_cls_config):
		neg_config.append(random_pose_neg(joint_location[i,:], n_joint, neg_wdw_half, pts1d, score1d))

	# Construct features for each configuration
	for p in pos_config:
		X_train[cnt,:] = calc_feature(p, score, feature_len, n_joint)
		cnt += 1
	for p in neg_config
		X_train[cnt,:] = calc_feature(p, score, feature_len, n_joint)
		Y_train[cnt,:] = -1
		cnt += 1

# Train SVM
smodel = BinaryClf(feature_len)
svm = OneSlackSSVM(smodel)
print("fitting svm...")
svm.fit(X_train, Y_train)

# Write weights
weights = format_weights(svm.w, n_joint)
