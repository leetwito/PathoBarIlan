
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.applications import mobilenet #, vgg16, inception_v3, resnet50, 
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# #### prepare data

# In[2]:


data_dir = 'E:/Datasets/PathoBarIlan/Case8'


# In[3]:


def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    imgs = []
    labels = []
    for row in tqdm(df.iterrows()):
        row = row[1]
#         print(row)
        file_path = row.loc["filepath"]
        if file_path.endswith('.png'):
            imgs.append(plt.imread(file_path))
        else:
            imgs.append(np.load(file_path))
        labels.append(row.loc["label"])
    return np.array(imgs), to_categorical(np.array(labels))


# In[4]:


x_train, y_train = load_data_from_csv(data_dir+"/train.csv")
x_test, y_test = load_data_from_csv(data_dir+"/test.csv")
x_eval, y_eval = load_data_from_csv(data_dir+"/eval.csv")


# In[5]:


print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
print(len(x_eval), len(y_eval))


# #### build/train model

# In[6]:


mobilenet_model = mobilenet.MobileNet(include_top=True, weights=None, input_shape=x_train[0].shape, classes=2, dropout=0.2)


# In[7]:


mobilenet_model.summary()


# In[8]:


optimizer = Adam(lr=1e-3)
mobilenet_model.compile(loss="binary_crossentropy", optimizer=optimizer)
lrReduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, min_lr=1e-6)
chkpnt = ModelCheckpoint("my_models/model_spec", save_best_only=True)


# In[9]:


mobilenet_model.fit(x=x_train, y=y_train, epochs=1000, validation_data=(x_eval, y_eval), batch_size=1, verbose=2, callbacks=[chkpnt, lrReduce], shuffle=True)


# #### evaluate

# In[10]:


y_pred = mobilenet_model.predict(x_test)
y_pred_bool = y_pred.argmax(axis=1)
y_pred_bool


# In[11]:


y_test_bool = y_test.argmax(axis=1)
y_test_bool


# In[12]:


(y_test_bool != y_pred_bool).mean()


# In[18]:


from keras.utils.vis_utils import plot_model
plot_model(mobilenet_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[17]:


get_ipython().system(' pip install pydot')

