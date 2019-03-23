import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

from functools import partial
from multiprocessing import Pool
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

input_dir = "/home/mxq/Project/Adversial_Attack/Guided-Denoise/Originset"
output_dir = "/home/mxq/Project/Adversial_Attack/Guided-Denoise/Advset/attacks_output/Iter2_v4_random"

batch_size = 3
use_existing = 0


flists = set([f for f in os.listdir(input_dir) if '.jpg' in f])
if use_existing == 1:
    flists_existing = set([f for f in os.listdir(output_dir) if '.jpg' in f ])
    newfiles = list(flists.difference(flists_existing))
    newfiles = [os.path.join(input_dir,f) for f in newfiles]
else:
    newfiles = [os.path.join(input_dir,f) for f in flists]
print('creating %s new files'%(len(newfiles)))
if len(newfiles) == 0:
    pass

filename_queue = tf.train.string_input_producer(newfiles, shuffle = False)
image_reader = tf.WholeFileReader()
filename, image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file,channels=3)
# image.set_shape((299, 299, 3))
image = tf.image.convert_image_dtype(image, dtype=tf.float32)

image_ = tf.image.resize_images(image, [299, 299])

images, filenames = tf.train.shuffle_batch([image_, filename], batch_size=batch_size, capacity=3*batch_size+10, min_after_dequeue=3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    img = sess.run(image_)
    print(np.shape(img))
    cv2.imshow("win", img)
    cv2.waitKey(0)
    coord.request_stop()
    coord.join(threads)