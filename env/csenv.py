# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 09:35:31 2022

@author: Yx
"""
import gym
from gym import spaces
import numpy as np
import os
import math
import sys
import random
ROOT_DIR = os.path.abspath('./env')
sys.path.append(ROOT_DIR)
from utils.action_space import MultiAgentActionSpace
from utils.observation_space import MultiAgentObservationSpace


def CSreward(state,num):
    # Traditional Cucker-Smale Flocking Reward
    beta = 0.5
    sigma = 10
    reward = -np.sqrt((state[num][4])**2-2*np.sqrt((state[num][4])**2*(state[0][4])**2)*np.cos(state[num][2]-state[0][2])+(state[0][4])**2)/np.sqrt(sigma**2+(state[num][0]-state[0][0])**2+(state[num][1]-state[0][1])**2)**beta+(state[num][0]-state[0][0])**2+(state[num][1]-state[0][1])**2
    return reward

def get_F(state_i,state_j):
    beta = 0.5
    sigma = 10
    c1 = 1
    c2 = 0
    delta_v = np.sqrt((state_i[4]) ** 2 - 2 * (state_i[4]) * (state_j[4]) * np.cos(state_i[2] - state_j[2]) + (state_j[4]) ** 2)
    delta_r = np.sqrt((state_i[0] - state_j[0]) ** 2 + (state_i[1] - state_j[1]) ** 2)
    f = (delta_v+c2) / np.sqrt(sigma ** 2 + delta_r ** 2) + c1 * delta_r
    return f


def L_G_CSreward(state,num):
    #Leader-guided Cucker-Smale Flocking Reward
    f_1 = 0
    F = []
    alpha = []
    theta = 1
    for i in range(0,len(state)):
        f = get_F(state[num],state[i])
        F.append(f)
        if i != 0:
            g=get_F(state[i],state[0])
        else:
            g=0
        alpha.append(g)
    sms=sum([np.exp(-theta*alpha[i]) for i in range(len(alpha))])
    for j in range(len(state)-1):
        alpha_j=np.exp(-theta*alpha[j])/sms
        f_1+=alpha_j*F[j]
    reward = -f_1**2
    return reward

def Q_flocking_reward(state,num):
    d1 = 40
    d2 = 65
    omega = 0.05
    rho = np.sqrt((state[0][0]-state[num][0])**2+(state[0][1]-state[num][1])**2)
    d = np.max([d1-rho,rho-d2,0])
    reward = -np.max([d,d1*(state[num][2]-state[0][2])/(np.pi*(1+omega*d))])
    return reward

def crash_jud(state):
    # Collision Judgement
    done=[False for i in range(9)]
    Dis=np.zeros((len(state),len(state)))
    for i in range(len(state)):
        for j in range(i,len(state)):
            if i==j:
                Dis[i,j]=10
            else:
                Dis[i,j]=np.sqrt((state[i][0]-state[j][0])**2+(state[i][1]-state[j][1])**2)
            if Dis[i,j]<=3:
                done[i]=True
                done[j]=True
                return done
    return done
    
# Leader-Follower gym Environment
    

class CS_flocking():
    metadata = {'render.modes': ['human']}
    def __init__(self,n_agents):
        
        '''
        state:
           [x,
            y,
            psi_f-l,
            roll_f,
            roll_l,
            v_f,
            v_l]
        '''

        #Input fixed-wing UAV's physical parameters
        self.vmax=3
        self.rollmax=np.pi/18
        self.alpha_g=9.8
        self.n=n_agents
        #Initialize flocking environment
        self.action_space = MultiAgentActionSpace(
            [spaces.Box(low = -1,high = 1,shape=(2,)) for _ in range(self.n)])
        self.state_space = MultiAgentActionSpace(
            [spaces.Box(low=np.array([-15000,-15000,-4*np.pi,-np.pi/12,
                                      -np.pi/6,12,12]),
                        high=np.array([15000,15000,4*np.pi,np.pi/12,
                                       np.pi/6,18,18]), shape=(7,)) for _ in range(self.n)])
        self.observation_space = MultiAgentActionSpace(
            [spaces.Box(low=np.array([-15000,-15000,-4*np.pi,-np.pi/12,
                                      -np.pi/12,12,12]),
                        high=np.array([15000,15000,4*np.pi,np.pi/12,
                                       np.pi/12,18,18]), shape=(7,)) for _ in range(self.n)])
        self.reset()
        
    def generator(self):
        '''
        output:
        [x,
         y,
         psi,
         roll,
         v]
        '''
        x_grid=235
        y_grid=235
        x=[]
        y=[]
        k=0
        for i in range(int(np.sqrt(self.n))+1):
            for j in range(int(np.sqrt(self.n))+1):
                x_grid = x_grid + i * 40
                y_grid = y_grid + j * 40
                x.append(x_grid)#+np.random.rand())
                y.append(y_grid)#np.random.rand())
                k+=1
                if k==self.n+1:
                    break
            if k==self.n+1:
                break
        psi=[np.random.rand() for i in range(len(x))]
        roll=[np.random.rand() for i in range(len(x))]        
        v=[12 for i in range(len(x))]
        self.state = [[] for i in range(len(x))]
        for j in range(len(x)):
            self.state[j].append(x[j])
            self.state[j].append(y[j])
            self.state[j].append(psi[j])
            self.state[j].append(roll[j])
            self.state[j].append(v[j])
        return self.state
    
    def reset(self):
        '''
        output:
        [x,  (relative coordinate)
         y,  (relative coordinate)
         psi_f-psi_l,
         roll_f,
         roll_l,
         v_l,
         v_f]
        '''
        state = self.generator()
        Rot=np.mat([[np.cos(state[0][2]),np.sin(state[0][2])],
             [-np.sin(state[0][2]),np.cos(state[0][2])]])

        '''
        rotation matrix of leader
        R = [ cos(psi_l),sin(psi_l)  ]
            [ -sin(psi_l),cos(psi_f) ]
        '''

        self.Rstate=[[] for i in range(len(state)-1)]
        for i in range(1,len(state)):
            s12=Rot*np.mat([[state[i][0]-state[0][0]],
               [state[i][1]-state[0][1]]])

            self.Rstate[i-1].append(float(s12[0]))
            self.Rstate[i-1].append(float(s12[1]))
            self.Rstate[i-1].append(state[i][2]-state[0][2])
            self.Rstate[i-1].append(state[0][3])
            self.Rstate[i-1].append(state[i][3])
            self.Rstate[i-1].append(state[0][4])
            self.Rstate[i-1].append(state[i][4])
        return self.Rstate
    
    def Convert(self):
        #Covert UAVs' state to leader-relative state
        state=self.state
        Rot=[[np.cos(state[0][2]),np.sin(state[0][2])],
             [-np.sin(state[0][2]),np.cos(state[0][2])]]
        Rot=np.mat(Rot)
        self.Rstate=[[] for i in range(len(state)-1)]
        for i in range(1,len(state)):
            r=[[state[i][0]-state[0][0]],
               [state[i][1]-state[0][1]]]
            r=np.mat(r)
            s12=Rot*r
            self.Rstate[i-1].append(float(s12[0]))
            self.Rstate[i-1].append(float(s12[1]))
            s3=state[i][2]-state[0][2]
            self.Rstate[i-1].append(s3)
            self.Rstate[i-1].append(state[0][3])
            self.Rstate[i-1].append(state[i][3])
            self.Rstate[i-1].append(state[0][4])
            self.Rstate[i-1].append(state[i][4])
            self.Rstate[i-1]=np.array(self.Rstate[i-1])
        #self.Rstate=np.array(self.Rstate)
        return self.Rstate

    def step(self, action: float):
        #action=[action[i]/10e2 for i in range(len(action))]
        reward = [0 for i in range(self.n)]
        done = False#[False for i in range(8)]
        action_l=[random.uniform(-1,1),random.uniform(-1,1)]
        
        #Update state
        state=self.state
        state[0][3] += action_l[1]*self.rollmax
        state[0][3] = np.clip(state[0][3],-np.pi/12,np.pi/12)
        state[0][4] += action_l[0]*self.vmax
        state[0][4] = np.clip(state[0][4],12,18)
        
        #update v & roll
        for j in range(1,len(state)):
            state[j][3] += action[j-1][1]*self.rollmax
            state[j][4] += action[j-1][0]*self.vmax
            state[j][3] = np.clip(state[j][3],-np.pi/12,np.pi/12)
            state[j][4] = np.clip(state[j][4],12,18)
            if sum(np.isnan(action[j-1]))>0:
                print('Action Boom')
                
            #if sum(np.isnan(state[j]))>0:
                #print('State Boom')
                #print(state[j])
        #update x,y,psi
        for k in range(len(state)):
            state[k][2] += -(self.alpha_g/(state[k][4]+0.0001))*np.tan(state[k][3])
            while abs(state[k][2])>2*np.pi:
                if state[k][2]>0:
                    state[k][2]-=2*np.pi
                if state[k][2]<0:
                    state[k][2]+=2*np.pi
            state[k][0] += state[k][4]*np.cos(state[k][2])*0.5
            state[k][1] += state[k][4]*np.sin(state[k][2])*0.5
            
        for i in range(self.n):
            rwd = Q_flocking_reward(state,i+1) # Turn on LG-CS Reward
            #rwd=L_G_CSreward(state,i+1) # Turn on Q-flocking Reward
            #rwd = CSreward(state,i+1) # Turn on C-S Reward
            if np.isnan(float(rwd))==True:
                rwd=-200
                #reward[i]=-2000
                done=True
            reward[i]+=rwd

        #convert state to Rstate
        self.Rstate = self.Convert()
        self.state=state
        info = []
        for i in range(len(state)):
            info.append(state[i])
        don=crash_jud(state)
        
        if any(don)==True:
            #done=True
            for i in range(self.n+1):
                if don[i]==True:
                    reward[i-1]-=200
            
        return self.Rstate,reward, done, info
    
    
    
    

    




if __name__ == '__main__':
    env=CS_flocking()
    rst=env.reset()
    buffer=[]
    done=False
    i=0
    while done==False:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        info=np.array(info)
        buffer.append(i)
        buffer[i]=reward
        i+=1
        if i==200:
            done=True
        #buffer[i]=info
        #info=[]
    
    