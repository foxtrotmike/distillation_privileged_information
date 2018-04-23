#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:17:33 2018

@author: fminhas
"""

from keras.layers.core import Dense, Dropout, Activation
from keras.objectives import categorical_crossentropy
#from keras.models import Sequential, Graph
from keras.models import Sequential, Model#, Graph
from scipy.misc import imresize
import numpy as np
import theano
import sys
from keras.layers import Input, Dense
from keras.utils import to_categorical

def downsample(x,p_down):
    size = len(imresize(x[0].reshape(28,28),p_down,mode='F').ravel())
    s_tr = np.zeros((x.shape[0], size))
    for i in xrange(x.shape[0]):
      img = x[i].reshape(28,28)
      s_tr[i] = imresize(img,p_down,mode='F').ravel()
    return s_tr

def MLP(d,m,q):
    model = Sequential()
    model.add(Dense(m, input_dim=d, activation="relu"))
    model.add(Dense(m, activation="relu"))
    model.add(Dense(q))
    model.add(Activation('softmax'))
    model.compile('rmsprop','categorical_crossentropy')
    return model

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]

def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function

def distillation(d,m,q,t,l):
    
    inputs = Input(shape=(d,), name = 'x')
    w1 = Dense(q)(inputs)
    z1 = Activation('relu', name = 'z1')(w1)
    w2 = Dense(q)(z1)
    z2 = Activation('relu', name = 'z2')(w2)
    w3 = Dense(q)(z2)
    hard_softmax = Activation('softmax', name = 'hard')(w3)
    soft_softmax = Activation('softmax', name = 'soft')(w3)
    model = Model(inputs=inputs, outputs=(hard_softmax,soft_softmax))
    loss_hard = weighted_loss(categorical_crossentropy,1.-l)
    loss_soft = weighted_loss(categorical_crossentropy,t*t*l)
    model.compile(optimizer = 'rmsprop', loss = {'hard':loss_hard, 'soft':loss_soft})
    
#    graph = Graph()
#    graph.add_input(name='x', input_shape=(d,))
#    graph.add_node(Dense(m), name='w1', input='x')
#    graph.add_node(Activation('relu'), name='z1', input='w1')
#    graph.add_node(Dense(m), name='w2', input='z1')
#    graph.add_node(Activation('relu'), name='z2', input='w2')
#    graph.add_node(Dense(q), name='w3', input='z2')
#    
#    
#    graph.add_node(Activation('softmax'), name='hard_softmax', input='w3')
#    graph.add_node(Activation('softmax'), name='soft_softmax', input='w3')
#    graph.add_output(name='hard', input='hard_softmax')
#    graph.add_output(name='soft', input='soft_softmax')
#    loss_hard = weighted_loss(categorical_crossentropy,1.-l)
#    loss_soft = weighted_loss(categorical_crossentropy,t*t*l)
#    graph.compile('rmsprop', {'hard':loss_hard, 'soft':loss_soft})
    return model

def load_data(dataset):
  d = np.load(dataset + '.npz','r')
  x_tr = d['train'].astype(np.float32).T#[:,::200]
  x_te = d['test'].astype(np.float32).T
  y_tr = d['train_labels'].astype(np.float32).T#[:,::200]
  y_te = d['test_labels'].astype(np.float32).T
  y_tr = to_categorical(y_tr)
  y_te = to_categorical(y_te)
  return x_tr, y_tr, x_te, y_te

np.random.seed(0)

ax_tr, ay_tr, x_te, y_te = load_data('mnist')
p_downsample = 25
N = 500#int(sys.argv[1])
M = 20

outfile = open('result_mnist_' + str(N), 'w')

xs_te = downsample(x_te,p_downsample)
x_te  = x_te/255.0
xs_te = xs_te/255.0

print xs_te.shape

for rep in xrange(5):
  # random training split
  i     = np.random.permutation(ax_tr.shape[0])[0:N]
  x_tr  = ax_tr[i]
  y_tr  = ay_tr[i]
  xs_tr = downsample(x_tr,p_downsample)
  x_tr  = x_tr/255.0
  xs_tr = xs_tr/255.0
  
  # big mlp
  mlp_big = MLP(x_tr.shape[1],M,y_tr.shape[1])
  mlp_big.fit(x_tr, y_tr, nb_epoch=50, verbose=0)
  err_big = np.mean(mlp_big.predict_classes(x_te,verbose=0)==np.argmax(y_te,1))
  
  # student mlp
  for t in [1,2,5,10,20,50]:
    for L in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:        
#        mlp_priv.layers[0].trainable = False
#    soften = theano.function([mlp_priv.layers[0].input], mlp_priv.layers[0].output)
      mlp_big.layers[0].trainable = False
      mlp_big.layers[2].trainable = False
      soften = theano.function([mlp_big.layers[0].input], mlp_big.layers[2].output)
      ys_tr  = softmax(soften(x_tr),t)
      
      mlp_student = distillation(xs_tr.shape[1],M,ys_tr.shape[1],t,L)
      mlp_student.fit(xs_tr,{'hard':y_tr, 'soft':ys_tr}, nb_epoch=50, verbose=0)
      
      err_student = np.mean(np.argmax(mlp_student.predict(xs_te)[0],1)==np.argmax(y_te,1))
      print t,L,err_student
      line = [N, p_downsample, round(err_big,3), t, L, round(err_student,3)]
      outfile.write(str(line)+'\n')

outfile.close()
