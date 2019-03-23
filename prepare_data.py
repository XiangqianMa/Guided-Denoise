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
# get_ipython().run_line_magic('matplotlib', 'inline')


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


imagenet_path = '/media/mxq/数据/DataSets/Tianchi/IJCAI_2019_AAAC_train'
path2 = './Originset/'
n_per_class = 4 # train
n_per_class_test = [10,40] # test

# 在imagenet_path的目录下，每一个文件夹代表一类样本
n_train = int(n_per_class*0.75 ) # 从用于训练的类别中抽取用于训练的类别，剩余的用于验证集
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


n_repeat = n_per_class
info_list = np.zeros([n_repeat*1000,12]).astype('str')
trainset_d1 = np.array([])
valset_d1 = np.array([])
trainset_d2 = np.array([])
valset_d2 = np.array([])
namelist = np.array([])

i_cum =0

for i_dir,dir in enumerate(subdirs):
    fullpath = os.path.join(imagenet_path,dir)
    filelist = os.listdir(fullpath)
    # 从filelist中随机选择n_repeat个样本
    randid = np.random.permutation(len(filelist))[:n_repeat]
    chosen_im = np.array(filelist)[randid]
    rename_im = np.array([n.split('.')[0]+'jpg' for n in chosen_im])
    # 从抽取的n_repeat个样本中抽取n_train个用于训练，剩余的用于验证
    trainset_d1 = np.concatenate([trainset_d1,rename_im[:n_train]])
    valset_d1 = np.concatenate([valset_d1,rename_im[n_train:]])
    
    # 被选中的样本的绝对路径
    fullimpath = [os.path.join(fullpath,f) for f in chosen_im]
    namelist = np.concatenate([namelist,fullimpath])

    labels = label_mapping[dir]
    if labels in class1:
        trainset_d2 = np.concatenate([trainset_d2,rename_im])
        valset_d2 = np.concatenate([valset_d2,rename_im])
    
    # 
    for i in range(n_repeat):
        target_class = labels
        while target_class==labels:
            target_class = np.random.randint(1000)
#         info_list[i].append([chosen_im[i].split('.')[0],0,0,0,1,1,labels,target_class,0,0,0,0])
        info_list[i_cum] = np.array([chosen_im[i].split('.')[0],0,0,0,1,1,labels,target_class,0,0,0,0])
        
        i_cum += 1
newpd = pandas.DataFrame(info_list)
newpd.columns = example.columns
newpd.to_csv('dev_dataset.csv', index=False)


pool = Pool()
resize_partial = partial(resize_all,namelist=namelist,path2=path2)
_ = pool.map(resize_partial,range(len(namelist)))


np.save('./utils/dataset1_train_split.npy',trainset_d1)
np.save('./utils/dataset1_val_split.npy',valset_d1)
np.save('./utils/dataset2_train_split.npy',trainset_d2)
np.save('./utils/dataset2_val_split.npy',valset_d2)


# 生成测试集
# 测试集来自于imagenet的验证集


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
        target_class = np.random.randint(110)
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

