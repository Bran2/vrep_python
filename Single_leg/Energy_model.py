import numpy as np
from scipy.interpolate import interp1d
from sympy import *
from matplotlib import pyplot as plt
import CPG_single


#####################  hyper parameters  ####################

Ra  = 6.33          # resistance
Km  = 339           # constant torque
Eta = 0.904*0.77    # efficiency
Ng  = 200           # transmission ratio
CPG = CPG_single.CPG_Single()
N = CPG.N
class Energy:

    def __init__(self,t0,t1,tau,theta_differ):
        self.t0 = t0                      #下限
        self.t1 = t1                      #上限
        self.t = np.linspace(0,t1,num=t1) # 0 - time_out  num = size(len(tau)) self.t = np.linspace(0,20,num=20)
        self.tau = tau                    #力矩
        self.theta_differ = theta_differ  #角速度

        
    def function(self):
        P = np.zeros((1,len(self.t))).ravel()  # 函数长度
        Pa = 0
        for i in range(len(self.t)):
            if self.theta_differ[i] > 0:
                P[i] = (Ra*self.tau[i]**2)/((Km*Eta*Ng)**2) + (self.theta_differ[i]*self.tau[i])/(Eta*Ng)
                print("瞬时功率为：",0.05*P[i])
            else:
                P[i] = (Ra*self.tau[i]**2)/((Km*Eta*Ng)**2)
                print("瞬时功率为：",0.05*P[i])

            Pa += 0.05*P[i]
        return Pa
    
