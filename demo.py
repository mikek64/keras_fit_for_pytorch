# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:46:36 2018
Demo of the keraspytorch functionality using the fashion MNIST data

@author: mike_
"""
import keras # only used to get the fashion mnist data set
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import pickle
import random
import os

from keraspytorch import CompiledModel

#%% download and save data if not already downloaded
SAVE_FILE = './fashion_mnist_data.pkl'

if not os.path.exists(SAVE_FILE):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump((train_images, train_labels, test_images, test_labels),f)

#%%

# load saved files from disk
with open(SAVE_FILE,'rb') as f: 
    train_images, train_labels, test_images, test_labels = pickle.load(f)
fashion_mnist_data = (train_images, train_labels, test_images, test_labels)


#%%
data_sets = ['train_images', 'train_labels', 'test_images', 'test_labels']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  
#%% explore the data
        
def explore_data(data, label = ''):
    print(label,
          type(data),
          data.shape,
          data.dtype)
    
    # display random entry
    d = len(data)
    r = random.randint(0,d)
    if type(data[r]) == np.ndarray:
        plt.imshow(data[r], cmap = 'gray')
        plt.show()
    else:
        print(data[r], class_names[data[r]])

for i, d in enumerate(fashion_mnist_data):
    explore_data(d, data_sets[i])

#%% random training images
    
d = train_images.shape[0]
r = random.randint(0,d)
print('Image {} label {} {}'.
      format(r, 
      train_labels[r], 
      class_names[train_labels[r]]))
plt.imshow(train_images[r], cmap = plt.cm.binary)
plt.colorbar()
plt.show()

#%%  Preprocess data

train_images_p = (train_images / 255.0 * 2.0 - 1.0).astype(np.float32)
test_images_p = (test_images / 255.0 * 2.0 - 1.0).astype(np.float32)
train_labels = train_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)

#%%

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


#%%

def new_model():
    model = Model()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    lossfn = nn.CrossEntropyLoss()
    metrics = ['accuracy']
    predictfn = nn.Softmax(dim = 1)
    
    compiled_model = CompiledModel(model, optimizer, lossfn, metrics,
                                   predictfn = predictfn)
    return compiled_model

compiled_model = new_model()
#%% train
hist = compiled_model.fit(train_images_p, train_labels, epochs = 5, 
                          batch_size = 128, validation_split = 0.2)

print('Training history')
for x in hist.items():
    print(x)


#%% save model
    
compiled_model.save('./testmodel.pkl')

#%% Plot training

n_epochs = len(hist['acc'])
epochs = list(range(n_epochs))

plt.plot(epochs, hist['acc'], 'b', label = 'train')
plt.plot(epochs, hist['val_acc'], 'r', label = 'val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training')
plt.show()


#%% evaluate

test_loss, test_accuracy = compiled_model.evaluate(test_images_p, test_labels, batch_size = 128)
print('Test accuracy: {:.4f}'.format(test_accuracy))

#%% Make predictions

predictions = compiled_model.predict(test_images_p) 
print(type(predictions), predictions.shape) # np array 10000, 10


#%% individual random predictions
s = test_images.shape[0]
r = random.randint(0, s)

p = predictions[r].argmax()
prob = predictions[r].max()
a = test_labels[r]
print('Item ', r, ' predict ', p, class_names[p], ' probability ', prob,
      ' actual ', a, class_names[a] )
plt.imshow(test_images[r], cmap = plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

#%% new model to demonstrate low accuracy of untrained model

compiled_model2 = new_model()
test_loss, test_accuracy = compiled_model2.evaluate(test_images_p, test_labels, batch_size = 128)
print('Test accuracy: {:.4f}'.format(test_accuracy))

#%% load model to demonstrate loading works

compiled_model2.load('./testmodel.pkl')
test_loss, test_accuracy = compiled_model2.evaluate(test_images_p, test_labels, batch_size = 128)
print('Test accuracy: {:.4f}'.format(test_accuracy))