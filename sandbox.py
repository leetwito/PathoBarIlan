
# coding: utf-8

# In[10]:


import scipy.io 
import matplotlib.pyplot as plt
import cv2
import keras
from glob import glob
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import pandas as pd

from keras.applications import mobilenet #, vgg16, inception_v3, resnet50, 
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# ## Params

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


# ## utils

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
#         file_name = os.path.basename(file)
#         print('Saving crops for file {} ...'.format(file_name))
#         print(file)
        spectral, rgb = read_slide(file)
        spectral_crops = create_batch_of_crops_from_slide(spectral, window_size=window_size, shift=shift)
        rgb_crops = create_batch_of_crops_from_slide(rgb, window_size=window_size, shift=shift)
#         if pos_name_init in file_name:
#             labels += [True]*len(added_rgb_crops)
#         elif neg_name_init in file_name:
#             labels += [False]*len(added_rgb_crops)
#         else:
#             raise ValueError('File {} is not in the right format ({}-pos, {}-neg)'.format(file_name, pos_name_init, neg_name_init))
#         print(labels)
        save_dir = file.replace('.mat', '_win{}-{}_shift{}-{}'.format(window_size[0], window_size[1], shift[0], shift[1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
#         np.save(save_dir+'/Spectral_crops.npy', spectral_crops)
#         np.save(file.replace('.mat', '_RGB_win{}-{}_shift{}-{}.npy'.format(window_size[0], window_size[1], shift[0], shift[1])), rgb_crops)
        for idx, (im_np, spec_np) in enumerate(zip(rgb_crops, spectral_crops)):
            im = Image.fromarray(im_np)
            im.save(os.path.join(save_dir, '{:05}.png'.format(idx)))
            np.save(os.path.join(save_dir, '{:05}.npy'.format(idx)), spec_np)
    
#     out_labels = to_categorical(labels)
#     out_sepc = np.stack(spectral_crops)
#     out_rgb = np.stack(rgb_crops)
    
    
#     return out_sepc, out_rgb, out_labels


# In[7]:


def create_crops_from_dir(dir_path, window_size, shift):
    print('Saving crops for slides in dir: {}'.format(dir_path))
    fileslist = glob(dir_path + '/*.mat')
    create_crops_from_fileslist(fileslist, window_size, shift)
#     return spectral_crops, rgb_crops, labels


# In[35]:


def create_csv_for_folder(data_dir, ext):
    if ext[0] == '.':
        ext = ext[1:]
    data_df = pd.DataFrame(columns=['filename', 'label'])
    files = glob(os.path.join(data_dir,'*', '*.{}'.format(ext)))
#     print(data_dir+'/*/*.{}'.format(ext))
    
    init_len = len(data_dir)
    files = [file[init_len:] for file in files]
#     print(files)
    labels = [1 if pos_name_init in file else 0 for file in files]
#     print(labels)
    data_df['filename'] = files
    data_df['label'] = labels
    data_df.to_csv(os.path.join(data_dir, os.path.basename(data_dir)+'.csv'), index=False)
    print('Created CSV successfully for folder {}'.format(data_dir))


# ## test and vis
path = mixed_paths[0]
spectral, rgb = read_slide(path)
img = rgb

crops = create_batch_of_crops_from_slide(img, window_size=window_size, shift=shift, vis_flag=True)plt.imshow(img)
# ## prepare data
train_spectral, train_rgb, train_labels = create_crops_from_dir(data_dir+'/Train', window_size=window_size, shift=shift)test_spectral, test_rgb, test_labels = create_crops_from_dir(data_dir+'/Test', window_size=window_size, shift=shift)
eval_spectral, eval_rgb, eval_labels = create_crops_from_dir(data_dir+'/Eval', window_size=window_size, shift=shift)create_crops_from_dir(data_dir, window_size=window_size, shift=shift)
# In[37]:


ext = 'png'
split_data_dir = 'E:\Datasets\PathoBarIlan'
create_csv_for_folder(split_data_dir+'/Train', ext)

print('Data:')
train_pos = train_labels[:,1].sum()
eval_pos = eval_labels[:,1].sum()
test_pos = test_labels[:,1].sum()

print('Train: {}/{} (pos/neg)'.format(train_pos, len(train_labels)-train_pos ))
print('Test: {}/{} (pos/neg)'.format(test_pos, len(test_labels)-test_pos ))
print('Eval: {}/{} (pos/neg)'.format(eval_pos, len(eval_labels)-eval_pos ))
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

# In[ ]:


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


# In[ ]:


mobilenet_model = mobilenet.MobileNet(include_top=True, weights=None, input_shape=x_train[0].shape, classes=2, dropout=0.2)


# In[ ]:


mobilenet_model.summary()


# In[ ]:


optimizer = Adam(lr=1e-3)
mobilenet_model.compile(loss="binary_crossentropy", optimizer=optimizer)
lrReduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, min_lr=1e-6)
chkpnt = ModelCheckpoint("my_models/model_spec", save_best_only=True)


# In[ ]:


print(x_train.shape, y_train.shape)
print(x_eval.shape, y_eval.shape)


# In[ ]:


mobilenet_model.fit(x=x_train, y=y_train, epochs=1000, validation_data=(x_eval, y_eval), batch_size=64, verbose=2, callbacks=[chkpnt, lrReduce], shuffle=True)


# In[ ]:


y_pred = mobilenet_model.predict(test_rgb)


# In[ ]:


y_pred = y_pred[:, 0]>y_pred[:, 1]


# In[ ]:


y_test = y_test[:, 0]>y_test[:, 1]


# In[ ]:


(y_test.T[0] != y_pred).sum()


# In[ ]:


len(y_test)

