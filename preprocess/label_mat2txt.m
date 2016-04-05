% Currently only using images with single person inside

load('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')

train_label_file = fopen('train_label.txt','w');
val_label_file = fopen('val_label.txt','w');
%test_label_file = fopen('test_label.txt', 'w');

VAL_RATIO = 0.2
format_spec = '%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n';
invalid_pos = [-1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0];

make_val = rand(size(RELEASE.img_train,2),1); %if < VAL_RATIO, add to validation set
train_count = 0;
val_count = 0;

multi_ppl_count = 0;
bad_count = 0;
for i = 1:size(RELEASE.annolist, 2)
%     if isfield(RELEASE.annolist(i), 'annorect') == 0
%         continue
%     end
%     for j = 1:size(RELEASE.annolist(i).annorect, 1) %person number
%         if isfield(RELEASE.annolist(i).annorect(j), 'annopoints') == 0,
%             continue
%         end
    if RELEASE.img_train(i) == 0
        continue
    end
    if size(RELEASE.annolist(i).annorect,2) > 1,
        multi_ppl_count = multi_ppl_count + 1;
        continue % skip images containing multiple people
    end
    if size(RELEASE.annolist(i).annorect,1) == 0, % zero person in picture
        if make_val(i) < VAL_RATIO % add to validation
            fprintf(val_label_file, format_spec, RELEASE.annolist(i).image.name, invalid_pos);
            val_count = val_count + 1;
        else % add to training
            fprintf(train_label_file, format_spec, RELEASE.annolist(i).image.name, invalid_pos);
            train_count = train_count + 1;
        end
        continue;
    end
    if isfield(RELEASE.annolist(i).annorect(1), 'annopoints') == 0,
        bad_count = bad_count + 1;
        continue;
    end
    
    tmp = orderfields(RELEASE.annolist(i).annorect(1).annopoints.point);
    format_annot = -ones(16,3);
    tmpcell = struct2cell(tmp);
    tmpcell = reshape(tmpcell, size(tmpcell, 1), []);
    tmpcell = tmpcell';
    tmpcell = sortrows(tmpcell,1);
    for j = 1:size(tmpcell, 1)
        format_annot(tmpcell{j,1}+1, 1) = tmpcell{j,3};
        format_annot(tmpcell{j,1}+1, 2) = tmpcell{j,4};
        if tmpcell{j,2} == 1,
            format_annot(tmpcell{j,1}+1, 3) = 1;
        else
            if (tmpcell{j,1} == 8 || tmpcell{j,1} == 9) && (size(tmpcell{j,2},1) == 0)
                format_annot(tmpcell{j,1}+1, 3) = 1;
                format_annot(tmpcell{j,1}+1, 1) = round(tmpcell{j,3});
                format_annot(tmpcell{j,1}+1, 2) = round(tmpcell{j,4});
            else
                format_annot(tmpcell{j,1}+1, 3) = 0;
            end
        end
    end
    if make_val(i) < VAL_RATIO % add to validation
        fprintf(val_label_file, format_spec, RELEASE.annolist(i).image.name, format_annot');
        val_count = val_count + 1;
    else % add to training
        fprintf(train_label_file, format_spec, RELEASE.annolist(i).image.name, format_annot');
        train_count = train_count + 1;
    end
    
end
train_count
val_count

multi_ppl_count
bad_count