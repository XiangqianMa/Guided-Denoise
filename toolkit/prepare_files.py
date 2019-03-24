import numpy as np
import os
import pandas

data_train_class_flag = False
val_txt_flag = False
calculate_class_number = True

if data_train_class_flag:
    # 准备dataset_train_class.npy文件，该文件中存放用于训练的类别名
    imagenet_path = '/media/mxq/数据/DataSets/Tianchi/IJCAI_2019_AAAC_train'
    subdirs = os.listdir(imagenet_path)
    subdirs = np.sort(subdirs)

    np.save("utils/dataset_train_class.npy", subdirs)

    print(subdirs)

if val_txt_flag:
    # 准备val.txt文件，该文件中存放用于测试的样本
    # 文件格式为：image_name label_id
    val_txt_name = "/home/mxq/Project/Adversial_Attack/Guided-Denoise/utils/val.txt"
    val_images_path = "/media/mxq/数据/DataSets/Tianchi/dev_data"
    label_file_name = "dev.csv"

    label_file = os.path.join(val_images_path, label_file_name)

    label = pandas.read_csv(label_file)
    file_names = label['filename']
    true_labels = label['trueLabel']

    with open(val_txt_name, 'w') as txt_file:
        for file_name, true_label in zip(file_names, true_labels):
            print(file_name, true_label)
            txt_file.writelines(file_name + ' ' + str(true_label) + '\n')

if calculate_class_number:
    imagenet_path = '/media/mxq/数据/DataSets/Tianchi/IJCAI_2019_AAAC_train'
    class_folders = os.listdir(imagenet_path)

    class_num = len(class_folders)
    print("----There are {} classes.----".format(class_num))
    
    # 挑出包含样本数目最少的类别
    class_number_min = 100000
    class_name_min = " "
    for class_folder in class_folders:
        image_files = os.listdir(os.path.join(imagenet_path, class_folder))
        image_num = len(image_files)
        print("----Class {} has {} images.----".format(class_folder, image_num))

        if(image_num < class_number_min):
            class_number_min = image_num
            class_nam_min = class_folder
        else:
            continue
    
    print("---Class {} has min images num: {}".format(class_name_min, class_number_min))


