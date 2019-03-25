# coding: utf-8

import os
import shutil
import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy.misc import imresize,imsave
from multiprocessing import Pool
from functools import partial
import cv2


class_num = 110
train_ratio = 0.99

def resize_all(id,namelist,path2):
    im = cv2.imread(namelist[id])
    print(os.path.join(path2,os.path.basename(namelist[id])))
    resizeim = cv2.resize(im, (299, 299))
    if len(resizeim.shape) == 2:
        resizeim = np.array([resizeim]*3).transpose([1,2,0])
    cv2.imwrite(os.path.join(path2,os.path.basename(namelist[id]).split('.')[0]+'.jpg'),resizeim)
    

# # generate the training set, set your imagenet_path
# 
# dataset1: the default dataset
# 
# dataset2用于评估HGD的迁移能力
# 
# dataset2: only 750 classes in the training set, and 250 classes in the testset, designed to evaluate the transferability
# 
# the training and validation sets are extracted from the training set of the imagenet

# 原始数据集的存放路径，该路径下每一个子文件夹分别代表一类
imagenet_path = '/media/mxq/数据/DataSets/Tianchi/IJCAI_2019_AAAC_train'
# 提取出的数据的存放路径
path2 = './Originset/'
# 每一类抽取的类别数目
n_per_class = 4 # train
n_per_class_test = [10,40] # test

# 在imagenet_path的目录下，每一个文件夹代表一类样本
n_train = int(n_per_class*0.75 ) # 从用于训练的类别中抽取用于训练的样本数目，剩余的用于验证集
subdirs = os.listdir(imagenet_path)
subdirs = np.sort(subdirs)
label_mapping={}
example = pandas.read_csv('./sample_dev_dataset.csv')
print(example.columns)

# 将从0计数映射为从1计数
for id,name in enumerate(subdirs):
    label_mapping[name] = id+1


# 加载dataset2中用于训练，测试的类别
class1 = np.load('utils/dataset_train_class.npy')
class2 = np.load('utils/dataset_train_class.npy')
class1 = [label_mapping[name] for name in class1]
class2 = [label_mapping[name] for name in class2]

# n_repeat表示从每一类中抽取的样本数目，乘以类别总数表示训练集中的样本总数，12表示.csv文件中总共有多少列
n_repeat = n_per_class
# info_list = np.zeros([n_repeat*class_num, 12]).astype('str')
info_list = np.zeros([1, 12])
trainset_d1 = np.array([])
valset_d1 = np.array([])
trainset_d2 = np.array([])
valset_d2 = np.array([])
namelist = np.array([])

for i_dir, dir in enumerate(subdirs):
    fullpath = os.path.join(imagenet_path,dir)
    filelist = os.listdir(fullpath)
    image_num = len(filelist)
    train_num = int(image_num * train_ratio)

    print("Class {} has {} images, in which {} images are chosen as train_sets, {} are chosen as val_sets."
            .format(dir, image_num, train_num, image_num-train_num))

    # 抽取所有的样本
    randid = np.random.permutation(len(filelist))
    chosen_im = np.array(filelist)[randid]
    rename_im = np.array([n.split('.')[0]+'jpg' for n in chosen_im])
    # 从抽取的样本中抽取train_num个用于训练，剩余的用于验证
    trainset_d1 = np.concatenate([trainset_d1,rename_im[:train_num]])
    valset_d1 = np.concatenate([valset_d1,rename_im[train_num:]])
    
    # 被选中的样本的绝对路径
    fullimpath = [os.path.join(fullpath,f) for f in chosen_im]
    namelist = np.concatenate([namelist, fullimpath])

    labels = label_mapping[dir]
    if labels in class1:
        trainset_d2 = np.concatenate([trainset_d2,rename_im[:train_num]])
        valset_d2 = np.concatenate([valset_d2,rename_im[train_num:]])
    
    for i in range(image_num):
        target_class = labels
        while target_class==labels:
            target_class = np.random.randint(class_num)

        new_column = np.array([[chosen_im[i].split('.')[0], 0, 0, 0, 1, 1, labels, target_class, 0, 0, 0, 0]])
        if i == 0 and i_dir == 0:
            info_list = new_column
        else:
            info_list = np.concatenate((info_list, new_column), axis=0)

newpd = pandas.DataFrame(info_list)
newpd.columns = example.columns
newpd.to_csv('dev_dataset.csv', index=False)


pool = Pool()
resize_partial = partial(resize_all,namelist=namelist,path2=path2)
_ = pool.map(resize_partial,range(len(namelist)))


# 用于产生对抗样本的训练集
np.save('./utils/dataset1_train_split.npy',trainset_d1)
np.save('./utils/dataset1_val_split.npy',valset_d1)

# 用于验证HGD迁移行的训练集
np.save('./utils/dataset2_train_split.npy',trainset_d2)
np.save('./utils/dataset2_val_split.npy',valset_d2)


# ---------------------------------------------------生成测试集-----------------------------------------------

imagenet_path = '/media/mxq/数据/DataSets/Tianchi/dev_data'
path2 = './Originset_test/'


with open('/home/mxq/Project/Adversial_Attack/Guided-Denoise/utils/val.txt') as f:
    tmp = f.readlines()
label_val = {}
for line in tmp:
    label_val[line.split(' ')[0]] = int(line.split(' ')[1].split('\n')[0])+1


example = pandas.read_csv('dev_dataset.csv')
keys = np.array(label_val.keys())
values = np.array(label_val.values())


i_cum = 0
info_list = np.zeros([len(label_val), 12]).astype('str')
namelist = np.array([])

for key in label_val:
    fullimpath = [os.path.join(imagenet_path, key)]
    labels = int(label_val[key])

    target_class = labels
    while target_class == labels:
        target_class = np.random.randint(class_num)
    info_list[i_cum, :] = np.array([key, 0, 0, 0, 1, 1, labels, target_class, 0, 0, 0, 0])

    # namelist = np.append(namelist, np.array(fullimpath), axis=0)
    namelist = np.concatenate([namelist, fullimpath])
    i_cum += 1


newpd = pandas.DataFrame(info_list)
newpd.columns = example.columns
newpd.to_csv('dev_dataset_test.csv', index=False)


label1 = pandas.read_csv('dev_dataset.csv')
label1 = np.array([label1['ImageId'],label1['TrueLabel']])
label2 = pandas.read_csv('dev_dataset_test.csv')
label2 = np.array([label2['ImageId'],label2['TrueLabel']])

tmp = np.concatenate([label1,label2],1).T
labels = {}
for key,value in tmp:
    labels[key] = value
np.save('utils/labels.npy',labels)


names = label2[0]
values = label2[1]
allnames = []
for i in range(1,111):
    class_names = names[values==i][:1]
    allnames.append(class_names)
allnames = np.concatenate(allnames)
np.save('utils/dataset1_test_split.npy',allnames)

allnames = []
class2 = np.load('utils/dataset_train_class.npy')
class2 = [label_mapping[name] for name in class2]
for i in class2:
    class_names = names[values==i]
    allnames.append(class_names)
allnames = np.concatenate(allnames)
np.save('utils/dataset2_test_split.npy',allnames)


resize_partial = partial(resize_all,namelist=namelist,path2=path2)
_ = pool.map(resize_partial,range(len(namelist)))
