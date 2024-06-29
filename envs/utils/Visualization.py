# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:26:21 2022

@author: Yx
"""



import matplotlib.pyplot as plt
import numpy as np

def Visual_Two_flocking(r):
    fig = plt.figure(figsize=(12, 6), facecolor='w')
    ax = fig.gca(projection='3d')
    ax.plot(r[0], r[1], r[2], '-^',label='Leader Trace')
    ax.plot(r[3], r[4], r[5],'-o',label='Follower 1 Trace')
    ax.plot(r[6], r[7], r[8],'-o',color='orange',label='Follower 2 Trace')
    plt.legend()