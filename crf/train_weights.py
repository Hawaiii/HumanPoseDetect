import numpy as np
from pystruct.learners import OneSlackSSVM

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

def random_pose_pos(joint_gt, n_joint):
	# Returns a tuple of 16 Pt
	# @TODO

def random_pose_neg(joint_gt, n_joint):
	# Returns a tuple of 16 Pt
	# @TODO

def calc_feature(pose, score):
	# Returns 1xfeature_len numpy array
	# @TODO

# Load the ground truth
score, joint_location = load_full_person_data() #@TODO
n_imgs = score.shape[0]
n_joint = 16
feature_len = n_joint + 120 #16 choose 2 binary

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
		pos_config.append(random_pose_pos(joint_location[i,:], n_joint))

	# Construct negative configurations
	neg_config = []
	for j in xrange(n_cls_config):
		neg_config.append(random_pose_neg(joint_location[i,:], n_joint))

	# Construct features for each configuration
	for p in pos_config:
		X_train[cnt,:] = calc_feature(p, score)
		cnt += 1
	for p in neg_config
		X_train[cnt,:] = calc_feature(p, score)
		Y_train[cnt,:] = -1
		cnt += 1

# Train SVM
# @TODO

# Write weights
# @TODO