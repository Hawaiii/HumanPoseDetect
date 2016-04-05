img_fn = '059241457.jpg';
%load('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
im = imread(img_fn);

img_names = struct2cell([RELEASE.annolist.image]);
img_idx = find( cellfun(@(x)isequal(x,img_fn),img_names) )

joint_struct = RELEASE.annolist(img_idx).annorect.annopoints(1).point;

figure()
imshow(im);
hold on
for i = 1:size(joint_struct, 2)
    if joint_struct(i).is_visible
        text(joint_struct(i).x,joint_struct(i).y,num2str(joint_struct(i).id),'Color','r','FontSize',30)
    else
        if joint_struct(i).is_visible == 0
            text(joint_struct(i).x,joint_struct(i).y,num2str(joint_struct(i).id),'Color','g','FontSize',30)
        else
            text(joint_struct(i).x,joint_struct(i).y,num2str(joint_struct(i).id),'Color','b','FontSize',30)
        end
    end
    joint_struct(i)
% if joint_struct(i).is_visible
%         text(joint_struct(i).y,joint_struct(i).x,num2str(joint_struct(i).id),'Color','r','FontSize',30)
%     else
%         if joint_struct(i).is_visible == 0
%             text(joint_struct(i).y,joint_struct(i).x,num2str(joint_struct(i).id),'Color','g','FontSize',30)
%         else
%             text(joint_struct(i).y,joint_struct(i).x,num2str(joint_struct(i).id),'Color','b','FontSize',30)
%         end
%     end
end