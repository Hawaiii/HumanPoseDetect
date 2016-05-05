import numpy as np
import h5py

def  load_full_person_data(feature_file_path): 
  feature_file = h5py.File(feature_file_path)
  gp_names = feature_file.keys()
  num_batches = len(gp_names) / 2
  CNNscore = np.array([])
  jointLocation = np.array([])
  for i in range(num_batches):
    joint_location = feature_file['/'+gp_names[i]]
    joint_location = np.array(joint_location)
    batch_size = joint_location.shape[0]
    joint_location = joint_location.reshape((batch_size,48))
    if len(jointLocation) == 0:
      jointLocation = joint_location
    else:
      jointLocation = np.vstack((jointLocation,joint_location))
    score = feature_file['/'+gp_names[i+num_batches]]
    score = np.array(score)
    if len(CNNscore) == 0:
     CNNscore = score
    else:
     CNNscore = np.concatenate((CNNscore,score), axis = 0)
  return [CNNscore, jointLocation]


#feature_file_path = "train_score_fullHuman.h5"
#[CNNscore, jointLocation] = load_full_person_data(feature_file_path)
#print CNNscore.shape
#print jointLocation.shape
   

