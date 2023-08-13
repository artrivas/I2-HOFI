# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:44:09 2018

@author: beheraa

from custom_validate_callback import TestCallback

callbacks = [TestCallback(test_datagen, 10)] # the model evaluate every 10 epochs
"""
import keras
import keras.backend as K
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import tensorflow as tf

import wandb
#from os.path import dirname, realpath


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name, wandb_log = False):
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.model_name = model_name
        self.wandb_log = wandb_log

    def on_epoch_end(self, epoch, logs={}):

        if tf.executing_eagerly():
            lr = self.model.optimizer.lr.numpy()
        else:
            lr = keras.backend.get_value(self.model.optimizer.lr)
        print(' - lr : ', lr)

        if self.wandb_log:
            # Log epoch, training accuracy and loss
            wandb.log({'epoch' : epoch})
            wandb.log({'loss': logs['loss'], 'acc': logs['acc']})
            wandb.log({'lr': lr})
            

        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            
            try:
                loss, acc = self.model.evaluate(self.test_generator) # change to model.evaluate()
            except:
                #model is regression, so must approximate accuracy
                loss = self.model.evaluate(self.test_generator) # change to model.evaluate()
                acc = validateRegression(self.test_generator, self.model)
               
            # Log validation accuracy and loss
            if self.wandb_log:
                wandb.log({'val_loss': loss, 'val_acc': acc})
                
                
                
def validateRegression(val_dg, model):
    predsAcc=[]
    trues=[]
    
    for b in range(len(val_dg)):
        x, y_true = val_dg.__getitem__(b)
        pred = model.predict(x)
        
        batch_size = pred.size
        
        
        #put trues in column
        y_true = np.reshape(y_true, (batch_size,1))
        
        for i in range(0, batch_size):
            
            y_true[i] = int(y_true[i] * 90)
            pred[i] = int(pred[i] * 90)
            
            
            if y_true[i] - 22.5 <= pred[i] <= y_true[i] + 22.5:
                pred[i] = y_true[i]
            else:
                pred[i] = int(-10) #rogue value
        
        predsAcc.append(pred[i])
        trues.append(y_true[i])
    
    #print(predsAcc)
    #print(trues)
    return accuracy_score(trues,predsAcc)

#writes validation metrics to csv file
def writeValToCSV(self, epoch, loss, acc):
    
    #get root directory
    #filepath = realpath(__file__)
    #metrics_dir = dirname(dirname(filepath)) + '/Metrics/'
    metrics_dir = 'Metrics/'
    
    
    with open(self.model_name + '(Validation).csv', 'a', newline='') as csvFile:
        metricWriter = csv.writer(csvFile)
        metricWriter.writerow([epoch, loss, acc])