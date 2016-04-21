train_label_file = 'train_label.txt';
val_label_file = 'val_label.txt';
new_train_label_file = 'train_label_resized.txt';
new_val_label_file = 'val_label_resized.txt';
new_height = 512;
new_width = 512;
image_folder = '~/mpii_human_pose_v1_images/';
new_image_folder = '~/mpii_human_pose_v1_images_resized/';

resize_images(train_label_file, new_height, new_width, new_train_label_file, image_folder, new_image_folder);
resize_images(val_label_file, new_height, new_width, new_val_label_file, image_folder, new_image_folder);
