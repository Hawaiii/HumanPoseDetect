new_width = 512;
new_height = 512;
image_folder = '~/mpii_human_pose_v1_images/';

label_file = fopen('val_label.txt','rb');
new_label_file = fopen('val_bbox.txt','w');
load('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat');

format_spec = '%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n';
write_spec = '%s %d %d\n';

tline = fgetl(label_file);
index = 1;
while ischar(tline) && index < size(RELEASE.annolist, 2)
    label = textscan(tline,format_spec);
    imageName = label{1}{1};
    
    while ~strcmp(RELEASE.annolist(index).image.name, imageName),
        index = index + 1;
    end
    
    if isfield(RELEASE.annolist(index).annorect(1), 'x1') == 0,
        fprintf('bad!\n');
    end
    
    x1 = RELEASE.annolist(index).annorect.x1;
    x2 = RELEASE.annolist(index).annorect.x2;
    y1 = RELEASE.annolist(index).annorect.y1;
    y2 = RELEASE.annolist(index).annorect.y2;
    
    image = imread(strcat(image_folder,imageName));
    imsize = double(size(image)); %y, x
%     imsize = [512, 512];

    m_size = [abs(x2-x1)*new_width/imsize(2), abs(y2-y1)*new_height/imsize(1)];
    fprintf(new_label_file, write_spec,imageName, m_size(1), m_size(2));
    
    tline = fgetl(label_file);
end
fclose(label_file);
fclose(new_label_file);