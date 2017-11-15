
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# In[2]:


import utils


# In[3]:


batch_size = 200
train_iter, val_iter, test_iter, text_field = utils.load_imdb(batch_size=batch_size)


# In[4]:


V = len(text_field.vocab)


# In[88]:


N_epochs = 10


# In[93]:


model = torch.nn.Sequential(
    torch.nn.Linear(V, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.BCELoss(size_average=True)

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

for i in range(N_epochs):
    print "epoch ", i
    train_iter.init_epoch()
    loss_batches = np.zeros(24000)
    
    for j, batch in enumerate(train_iter):
        x = utils.bag_of_words(batch, text_field)
        y = batch.label - 1
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y.float().view(batch_size, 1))
        model.zero_grad()
        loss.backward()
        
        if j%20 == 0:
            print i, j, loss.data[0]
            
        optimizer.step()


# In[95]:


# ACCURACY

val_iter.init_epoch()
indexes_vocabulary = np.arange(V)

accuracy = 0.
total_number = 0.

for batch in val_iter:
    x = utils.bag_of_words(batch, text_field)
    y = batch.label - 1
    y_int = y.data.cpu().numpy()[0]
    
    y_pred = model(x)
    y_aux = y_pred - y.float().view(batch_size, 1)
    
    aux_acc = torch.lt(y_aux.abs(), 0.5).sum()
    accuracy += float(aux_acc.data.numpy()[0])
    
    total_number += batch_size

accuracy = accuracy/float(total_number)
print accuracy

