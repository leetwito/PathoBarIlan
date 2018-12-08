
# coding: utf-8

# In[1]:


import scipy.io 
import matplotlib.pyplot as plt
import cv2
import keras
from glob import glob
import numpy as np
from tqdm import tqdm
import os

from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# In[2]:


normal_paths = glob('E:/Datasets/PathoBarIlan/Case8/Normal*.mat')
cancer_paths = glob('E:/Datasets/PathoBarIlan/Case8/Cancer*.mat')
mixed_paths = glob('E:/Datasets/PathoBarIlan/Case8/Mixed*.mat')

data_dir = 'E:/Datasets/PathoBarIlan/Case8'

pos_name_init = 'Cancer'
neg_name_init = 'Normal'

use_rgb = True # True=rgb, False=spectral

window_size = (200, 200)
shift = (100, 100)


# In[3]:


def read_slide(path):
    mat = scipy.io.loadmat(path)
    spectral = mat["Spec"]
    rgb = mat["Section"]
    shape = rgb.shape
    
    return spectral, rgb


# In[4]:


def create_batch_of_crops_from_slide(img, window_size, shift, vis_flag=False):
    crops = []

    n_iter_x = (img.shape[1]-window_size[0])//shift[0] + 1

    n_iter_y = (img.shape[0]-window_size[1])//shift[1] + 1

#     n_iter_x, n_iter_y

    for i in range(n_iter_x):
        for j in range(n_iter_y):
            init_y = i*shift[0]
            init_x = j*shift[1]
        
            crops.append(img[init_x:init_x+window_size[0], init_y:init_y+window_size[1], :])
    if vis_flag:
        visualize_batch_of_crops(crops, n_iter_y, n_iter_x)
    return crops


# In[5]:


def visualize_batch_of_crops(crops, n_iter_y, n_iter_x):
    fig, axes = plt.subplots(n_iter_y, n_iter_x, figsize=(5, 5), gridspec_kw = {'wspace':0, 'hspace':0})

    for i in range(n_iter_x):
        for j in range(n_iter_y):
            axes[j, i].imshow(crops[i*n_iter_y + j])
            axes[j, i].axis('off')
            axes[j, i].set_aspect('equal')
    plt.show()


# In[6]:


def create_crops_from_fileslist(fileslist, window_size, shift):
    rgb_crops = []
    spectral_crops = []
    labels = []

    for file in tqdm(fileslist):
#         print(file)
        spectral, rgb = read_slide(file)
        spectral_crops += create_batch_of_crops_from_slide(spectral, window_size=window_size, shift=shift)
        added_rgb_crops = create_batch_of_crops_from_slide(rgb, window_size=window_size, shift=shift)
        rgb_crops += added_rgb_crops
        file_name = os.path.basename(file)
        if pos_name_init in file_name:
            labels += [True]*len(added_rgb_crops)
        elif neg_name_init in file_name:
            labels += [False]*len(added_rgb_crops)
        else:
            raise ValueError('File {} is not in the right format ({}-pos, {}-neg)'.format(file_name, pos_name_init, neg_name_init))
#         print(labels)
    out_labels = to_categorical(labels)
    out_sepc = np.stack(spectral_crops)
    out_rgb = np.stack(rgb_crops)
    
    
    return out_sepc, out_rgb, out_labels


# In[7]:


def create_crops_from_dir(dir_path, window_size, shift):
    fileslist = glob(dir_path + '/*')
    spectral_crops, rgb_crops, labels = create_crops_from_fileslist(fileslist, window_size, shift)
    return spectral_crops, rgb_crops, labels


# ## test and vis
path = mixed_paths[0]
spectral, rgb = read_slide(path)
img = rgb

crops = create_batch_of_crops_from_slide(img, window_size=window_size, shift=shift, vis_flag=True)plt.imshow(img)
# ## prepare data

# In[8]:


train_spectral, train_rgb, train_labels = create_crops_from_dir(data_dir+'/Train', window_size=window_size, shift=shift)


# In[9]:


test_spectral, test_rgb, test_labels = create_crops_from_dir(data_dir+'/Test', window_size=window_size, shift=shift)
eval_spectral, eval_rgb, eval_labels = create_crops_from_dir(data_dir+'/Eval', window_size=window_size, shift=shift)


# In[10]:


print(eval_labels, train_labels, test_labels)


# ##### old prepare data
train_pos_spectral_crops, train_pos_rgb_crops = create_crops_from_fileslist(cancer_paths[:-3], window_size=window_size, shift=shift)
test_pos_spectral_crops, test_pos_rgb_crops = create_crops_from_fileslist(cancer_paths[-3:-1])
val_pos_spectral_crops, val_pos_rgb_crops = create_crops_from_fileslist(cancer_paths[-1:])train_neg_spectral_crops, train_neg_rgb_crops = create_crops_from_fileslist(normal_paths[:-2])
test_neg_spectral_crops, test_neg_rgb_crops = create_crops_from_fileslist(normal_paths[-2:-1])
val_neg_spectral_crops, val_neg_rgb_crops = create_crops_from_fileslist(normal_paths[-1:])y_pos_train = [True]*len(train_pos_rgb_crops)
y_neg_train = [False]*len(train_neg_rgb_crops)

y_pos_test = [True]*len(test_pos_rgb_crops)
y_neg_test = [False]*len(test_neg_rgb_crops)

y_pos_val = [True]*len(val_pos_rgb_crops)
y_neg_val = [False]*len(val_neg_rgb_crops)len(test_pos_rgb_crops)# train_X_spectral = np.stack(train_pos_spectral_crops + train_neg_spectral_crops)
train_X_rgb = np.stack(train_pos_rgb_crops + train_neg_rgb_crops)
y_train = np.array(y_pos_train + y_neg_train)

# test_X_spectral = np.stack(test_pos_spectral_crops + test_neg_spectral_crops)
test_X_rgb = np.stack(test_pos_rgb_crops + test_neg_rgb_crops)
y_test = np.array(y_pos_test + y_neg_test)

# val_X_spectral = np.stack(val_pos_spectral_crops + val_neg_spectral_crops)
val_X_rgb = np.stack(val_pos_rgb_crops + val_neg_rgb_crops)
y_val = np.array(y_pos_val + y_neg_val)# print(len(y_train))
# print(len(train_X_rgb))
# ## build and train model

# In[11]:


if use_rgb:
    x_train = train_rgb
    x_eval = eval_rgb
    x_test = test_rgb
else:
    x_train = train_spectral
    x_eval = eval_spectral
    x_test = test_spectral
y_train = train_labels
y_eval = eval_labels
y_test = test_labels


# In[13]:


print('Data:')
print(y_train.shape)
train_pos = y_train[:,0].sum()
eval_pos = y_eval[:,0].sum()
test_pos = y_test[:,0].sum()

print('Train: {}/{} (pos/neg)'.format(train_pos, len(y_train)-train_pos ))
print('Test: {}/{} (pos/neg)'.format(test_pos, len(y_test)-test_pos ))
print('Eval: {}/{} (pos/neg)'.format(eval_pos, len(y_eval)-eval_pos ))


# In[14]:


mobilenet_model = mobilenet.MobileNet(include_top=True, weights=None, input_shape=x_train[0].shape, classes=2, dropout=0.2)


# In[15]:


mobilenet_model.summary()


# In[16]:


optimizer = Adam(lr=1e-3)
mobilenet_model.compile(loss="binary_crossentropy", optimizer=optimizer)
lrReduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, min_lr=1e-6)
chkpnt = ModelCheckpoint("my_models/model_spec", save_best_only=True)


# In[17]:


print(x_train.shape, y_train.shape)
print(x_eval.shape, y_eval.shape)


# In[18]:


mobilenet_model.fit(x=x_train, y=y_train, epochs=1000, validation_data=(x_eval, y_eval), batch_size=64, verbose=2, callbacks=[chkpnt, lrReduce], shuffle=True)


# In[19]:


y_pred = mobilenet_model.predict(test_rgb)


# In[20]:


y_pred = y_pred[:, 0]>y_pred[:, 1]


# In[21]:


y_test = y_test[:, 0]>y_test[:, 1]


# In[22]:


(y_test.T[0] != y_pred).sum()


# In[23]:


len(y_test)

