#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:53:47 2023

@author: kunwang
"""

from river.datasets import synth
import pandas as pd
import numpy as np

    
    
#------------------------
# STAGGER datasets
#------------------------

def STAGGER_data():
    
    for i in range(15):
        
        dataset = synth.STAGGER(classification_function = 0, seed = i,
                          balance_classes = False)
    
        df_x = pd.DataFrame(columns = ['size','color','shape'])
        df_y = pd.DataFrame(columns = ['label'])
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/data/kwang3/work8/river data/STAGGER/'+'STAGGER'+str(i)+'.arff', index = False)
        
        
        
def Mixed_data():
    
    for i in range(15):
        
        dataset = synth.Mixed(seed = i, classification_function=1, balance_classes = False)
    
        df_x = pd.DataFrame(columns = [0,1,2,3])
        df_y = pd.DataFrame(columns = ['label'])
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/data/kwang3/work8/river data/Mixed/'+'Mixed'+str(i)+'.arff', index = False)
    
    
    
def AnomalySine_data():
    
    for i in range(15):
        
        dataset = synth.AnomalySine(seed=i,n_samples=100000,n_anomalies=25,contextual=True,n_contextual=10)
                                        
        df_x = pd.DataFrame(columns = ['sine', 'cosine'])
        df_y = pd.DataFrame(columns = ['label'])
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/data/kwang3/work8/river data/AnomalySine/'+'AnomalySine'+str(i)+'.arff', index = False)
    

        
        
        
def Hyperplane_data():
    
    for i in range(15):
        
        dataset = synth.Hyperplane(seed=42, n_features=2)
        
        df_x = pd.DataFrame(columns = [0, 1])
        df_y = pd.DataFrame(columns = ['label'])
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/data/kwang3/work8/river data/Hyperplane/'+'Hyperplane'+str(i)+'.arff', index = False)
        
        
        

def LEDDrift_data():
    
    for i in range(15):
        
        dataset = synth.LEDDrift(seed = i, noise_percentage = 0.28,
                         irrelevant_features= True, n_drift_features=4)
        
        df_x = pd.DataFrame(columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
        df_y = pd.DataFrame(columns = ['label'])
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/data/kwang3/work8/river data/LEDDrift/'+'LEDDrift'+str(i)+'.arff', index = False)
    
    

    
def LED_data():
    
    for i in range(15):
        
        dataset = synth.LED(seed = 112, noise_percentage = 0.28, irrelevant_features= False)

        df_x = pd.DataFrame(columns = [0,1,2,3,4,5,6])
        df_y = pd.DataFrame(columns = ['label'])
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/data/kwang3/work8/river data/LED/'+'LED'+str(i)+'.arff', index = False)
    
    

'''
#------------------------
# Sine datasets
#------------------------

def Sine_data():
    
    for i in range(15):
        dataset = synth.Sine(classification_function = 2, seed = i,
                              balance_classes = True, has_noise = True)
        
        df_x = pd.DataFrame(columns = [0,1,2,3])
        df_y = pd.DataFrame(columns = ['label'])
        
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/home/kunwang/Data/Work4/data/synthetic data/Sine/'+'Sine'+str(i)+'.arff', index = False)
    
    
    
#------------------------
# Waveform datasets
#------------------------

def Waveform_data():
    
    for i in range(15):
        
        dataset = synth.Waveform(seed=i, has_noise=True)
        
        df_x = pd.DataFrame(columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
        df_y = pd.DataFrame(columns = ['label'])
        
        # collect the generated data to Dataframe
        for x, y in dataset.take(100000):
            
            xf = pd.DataFrame.from_dict(x, orient='index').transpose()
            df_x.loc[len(df_x)] = xf.loc[0]
            df_y.loc[len(df_y)] = y
            
        # Merge x, y dataframe
        data = pd.concat([df_x, df_y], axis = 1)
        
        # Save dataframe to noteboosk with delimete
        data.to_csv('/home/kunwang/Data/Work4/data/synthetic data/Waveform/'+'Waveform'+str(i)+'.arff', index = False)
    
'''    
    
#------------------------
# generate datasets:
#------------------------

# STAGGER_data()
# Mixed_data()
# Sine_data()
# Waveform_data()
# AnomalySine_data()
# Hyperplane_data()
# LEDDrift_data()
LED_data()
    

