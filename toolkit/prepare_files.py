import numpy as np
import os
import pandas

data_train_class_flag = False
val_txt_flag = True

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


