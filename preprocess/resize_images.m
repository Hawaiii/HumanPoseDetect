
function []=resize_images(label_file_path, new_height, new_width, new_label_file_path, image_folder, new_image_folder)

format_spec = '%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n';

label_file = fopen(label_file_path,'r');
new_label_file = fopen(new_label_file_path,'w');
joints=int16(zeros(1,48));
tline = fgetl(label_file);
num_lines=1;
while ischar(tline)
    label = textscan(tline,format_spec);
    imageName = label{1};
    display(num_lines);
    num_lines=num_lines+1;
    for i = 1:48
        joints(i) = label{i+1};
    end
    image = imread(strcat(image_folder,imageName{1}));
    old_size = double(size(image));
    scale = [new_height, new_width]./old_size(:,1:2);
    new_image = imresize(image, [new_height, new_width]);  
    imwrite(new_image,strcat(new_image_folder, imageName{1}));
    for i=1:16
        if joints(3*i)~=-1
            joints(3*i-2) = min(max(1,joints(3*i-2)*scale(2)),new_width-1);
            joints(3*i-1) = min(max(1,joints(3*i-1)*scale(1)),new_height-1);
        end
    end
    fprintf(new_label_file, format_spec,imageName{1},joints);
    tline=fgetl(label_file);
end
fclose(label_file);
fclose(new_label_file);
