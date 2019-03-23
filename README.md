# Guided-Denoise
The winning submission for NIPS 2017: Defense Against Adversarial Attack of team TSAIL

# Paper 
[Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser](https://arxiv.org/abs/1712.02976)

# File Description

* prepare_data.ipynb: generate dataset

* Originset, Originset_test: the folder for original image

* toolkit: the program running the attack in batch

* Attackset: the attacks

* Advset: the adversarial images

* checkpoints: the models checkpoint used, download [here](https://pan.baidu.com/s/1kVzP9nL)

* Exps: the defense model

* GD_train, PD_train: train the defense model using guided denoise or pixel denoise

# How to use
the attacks are stored in folder Attackset 
the script is in the toolkit folder. 
in the run_attacks.sh file:
modify models to the attacks you want to generate, separate by comma, or use "all" to include all attacks in Attackset.
use the command to run:

   `bash run_attacks.sh $gpuids`
   
where gpuids is the id of the gpus you want to use, they are number separated by comma. It will generate the training set.
Then change the line `DATASET_DIR="${parentdir}/Originset"` to `DATASET_DIR="${parentdir}/Originset_test"`, and run the command    `bash run_attacks.sh $gpuids` again.

Then specify a model you want to use, the models are stored in Exp folder, there is a sample folder, it refers to a model named "sample", let's use it. Then go to GD_train if you want to use guided denoiser, 
run 

`python main --exp sample ` 

The program will load Exp/sample/model.py as a model to train. and also you can specify other parameters defined in the GD_train/main.py

# 修改

## 在运行attack_iter.py脚本时会报错，需要修改以下几处：

1. 注意加载的图片的格式，原始的图片格式为.png，需要将其修改为.jpg．同时修改tf的图片加载函数:
```
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_ = tf.image.resize_images(image, [299, 299])
```

2. 原始脚本中使用的``image.set_shape((299, 299, 3))``不起作用，将其替换为以下几句：
```
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image_ = tf.image.resize_images(image, [299, 299])
```

3. 将图片保存函数修改为以下格式：
```
  def save_images(arg):
      image,filename,output_dir = arg
      imsave(os.path.join(output_dir, str(filename, 'utf-8')), (image + 1.0) * 0.5)
```

4. 类别数目
将
```
def graph(x, y, i, x_max, x_min, grad, eps_inside):
```
中的
```
  num_classes = 1001
```
修改为自己的类别总数．

## prepare_data的修改

`target_class = np.random.randint(1000)`中的`1000`修改为自己的训练集的类别数