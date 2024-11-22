# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:05:19 2024

@author: david

Compare the different hyper parameter tests.
Make a figure showing the comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

#Create a figure with the required number of subplots.
fig = plt.figure(figsize=(8, 5.33),dpi=300)
axs = fig.subplots(nrows=2,ncols=3)
plt.subplots_adjust(wspace=0.3,hspace=0.2)

#Plot the training and validation loss for the original model.
folder = '../Neural_Network_Method'
history_original = np.load(folder+'/DetectAreasAndAxisModel_history.npy',allow_pickle=True).item()
epochs = np.arange(len(history_original['loss']))
axs[0,0].plot(epochs,history_original['loss'],'C1-',epochs,history_original['val_loss'],'C0-')
axs[0,0].set_ylim(0.1,0.6)
# axs[0,0].set_xlabel('Epoch')
axs[0,0].set_ylabel('Loss')
axs[0,0].text(10,0.55,'A')
axs[0,0].legend(['Training','Validation'])

#Define a function to compare three histories.
def PlotHistories(histories,names,title,ax,xlabel,ylabel,letter):
    colors = ['C0','C1','C2']
    for i,history in enumerate(histories):
        epochs = np.arange(len(history['loss']))
        ax.plot(epochs,history['loss'],colors[i]+'-')
        # ax.plot(epochs,history['val_loss'],colors[i]+'-')
        ax.set_ylim(0.1,0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.text(10,0.55,letter)
    ax.legend(names,title=title)

#Learning rate.
history1 = np.load(folder+'/HyperParameterTest/LR1e-3_history.npy',allow_pickle=True).item()
history2 = np.load(folder+'/HyperParameterTest/LR1e-5_history.npy',allow_pickle=True).item()
PlotHistories([history1,history_original,history2],['1e-3','1e-4','1e-5'],'Learning Rate',axs[0,1],None,None,'B')
print(history1['loss'][-1],history_original['loss'][-1],history2['loss'][-1])

#Batch Size.
history1 = np.load(folder+'/HyperParameterTest/4Batch_history.npy',allow_pickle=True).item()
history2 = np.load(folder+'/HyperParameterTest/16Batch_history.npy',allow_pickle=True).item()
PlotHistories([history1,history_original,history2],['4','8','16'],'Batch Size',axs[0,2],None,None,'C')
print(history1['loss'][-1],history_original['loss'][-1],history2['loss'][-1])

#UNET Layers
history1 = np.load(folder+'/HyperParameterTest/4Steps_history.npy',allow_pickle=True).item()
history2 = np.load(folder+'/HyperParameterTest/6Steps_history.npy',allow_pickle=True).item()
PlotHistories([history1,history_original,history2],['4','5','6'],'UNET Levels',axs[1,0],'Epoch','Loss','D')
print(history1['loss'][-1],history_original['loss'][-1],history2['loss'][-1])

#UNET Features (1st Level)
history1 = np.load(folder+'/HyperParameterTest/32Filters_history.npy',allow_pickle=True).item()
history2 = np.load(folder+'/HyperParameterTest/128Filters_history.npy',allow_pickle=True).item()
PlotHistories([history1,history_original,history2],['32','64','128'],'UNET Features',axs[1,1],'Epoch',None,'E')
print(history1['loss'][-1],history_original['loss'][-1],history2['loss'][-1])

#Activation Function
history1 = np.load(folder+'/HyperParameterTest/elu_history.npy',allow_pickle=True).item()
PlotHistories([history1,history_original],['ELU','ReLU'],'Activation',axs[1,2],'Epoch',None,'F')
print(history1['loss'][-1],history_original['loss'][-1])

#Save the figure.
fig.savefig('Fig4.pdf',bbox_inches='tight')
fig.savefig('Fig4.png',bbox_inches='tight')