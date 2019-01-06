# setting
from __future__ import division
import time
import numpy as np
import math
import vrep
import CPG_single
import Energy_model_lisan

RAD2EDG = 180 / math.pi   # 常数，度数转弧度
tstep = 0.000000005             # 定义仿真步长
#####################  configuration data stream  #################### 
jointNum = 3
forceName = 'Force_sensor'
baseName = 'body1_vision'
jointName = 'body1_joint'


############ init
print('Program started')
# 关闭潜在的连接
vrep.simxFinish(-1)
# 每隔0.2s检测一次，直到连接上V-rep
while True:
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID > -1:
        break
    else:
        time.sleep(0.2)
        print("Failed connecting to remote API server!")
print("Connection success!")


# 设置仿真步长，为了保持API端与V-rep端相同步长
vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step, tstep, vrep.simx_opmode_oneshot)
# 然后打开同步模式
vrep.simxSynchronous(clientID, True) 
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

# 然后读取Base和Joint的句柄
jointHandle = np.zeros((jointNum,), dtype=np.int)  # 注意是整型
for i in range(jointNum):
    _, returnHandle = vrep.simxGetObjectHandle(clientID, jointName + str(i+1), vrep.simx_opmode_blocking)
    _, forceHandle  = vrep.simxGetObjectHandle(clientID, forceName, vrep.simx_opmode_blocking)
    jointHandle[i] = returnHandle

_, baseHandle = vrep.simxGetObjectHandle(clientID, baseName, vrep.simx_opmode_blocking)

print('Handles available!')

#####################  hyper parameters  ####################

CPG = CPG_single.CPG_Single()
N = CPG.N
T = np.zeros((1,N+1)).ravel()
X = np.zeros((1,N+1)).ravel()
Y = np.zeros((1,N+1)).ravel()
X[0] = -0.6
Y[0] = -0.5

F = np.zeros((1,N+1)).ravel()
V = np.zeros((1,N+1)).ravel()


h = CPG.h
while vrep.simxGetConnectionId(clientID) != -1:
    for i in range(N):
        k1 = CPG.f(T[i], X[i], Y[i])  # k1=feval(f,T(i),X(i),Y(i));
        l1 = CPG.g(T[i], X[i], Y[i])  # l1=feval(g,T(i),X(i),Y(i));
        k2 = CPG.f( T[i]+h/2, X[i]+(h/2)*k1, Y[i]+(h/2)*l1)  # k2=feval(f,T(i)+h/2,X(i)+(h/2)*k1,Y(i)+(h/2)*l1);
        l2 = CPG.g( T[i]+h/2, X[i]+(h/2)*k1, Y[i]+(h/2)*l1)  # l2=feval(g,T(i)+h/2,X(i)+(h/2)*k1,Y(i)+(h/2)*l1);
        k3 = CPG.f( T[i]+h/2, X[i]+(h/2)*k2, Y[i]+(h/2)*l2)  # k3=feval(f,T(i)+h/2,X(i)+(h/2)*k2,Y(i)+(h/2)*l2);
        l3 = CPG.g( T[i]+h/2, X[i]+(h/2)*k2, Y[i]+(h/2)*l2)  # l3=feval(g,T(i)+h/2,X(i)+(h/2)*k2,Y(i)+(h/2)*l2);
        k4 = CPG.f( T[i]+h, X[i]+h*k3, Y[i]+h*l3)  #k4=feval(f,T(i)+h,X(i)+h*k3,Y(i)+h*l3);
        l4 = CPG.g( T[i]+h, X[i]+h*k3, Y[i]+h*l3)  #l4=feval(g,T(i)+h,X(i)+h*k3,Y(i)+h*l3);
        X[i+1] = X[i] + (h/6)*(k1+2*k2+2*k3+k4)
        Y[i+1] = Y[i] + (h/6)*(l1+2*l2+2*l3+l4)
        x = X[i+1]                    #位置不能乱变
        num = i                       # 用在 F V 的记录上
        
        # 然后首次读取关节的初始值，以streaming的形式
        jointConfig = np.zeros((jointNum,))
        for k in range(jointNum):
             _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[k], vrep.simx_opmode_streaming)
             jointConfig[k] = jpos             
                 
        #simulation
        lastCmdTime=vrep.simxGetLastCmdTime(clientID)  # 记录当前时间
        vrep.simxSynchronousTrigger(clientID)  # 让仿真走一步    
        # start simulation    
        currCmdTime=vrep.simxGetLastCmdTime(clientID)  # 记录当前时间
        dt = currCmdTime - lastCmdTime                 # 记录时间间隔，用于控制


        ##读取当前的状态值，之后都用buffer形式读取 vrep.simx_opmode_streaming  vrep.simx_opmode_oneshot vrep.simx_opmode_buffer
        for i in range(jointNum):
            _,jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_buffer)
            _,f = vrep.simxGetJointForce(clientID,jointHandle[1],vrep.simx_opmode_oneshot)        # 关节力矩
            _,_v,v = vrep.simxGetObjectVelocity(clientID,jointHandle[1],vrep.simx_opmode_oneshot) # 关节转速
            #_,state,f,t = vrep.simxReadForceSensor(clientID,forceHandle,vrep.simx_opmode_oneshot) # 力传感器度数
            # _,name,f_,f,_f = vrep.simxGetObjectGroupData(clientID,sim.object_forcesensor_type,sim.object_graph_type,vrep.simx_opmode_buffer)#待定
            # print(round(jpos * RAD2EDG, 2))
            # print(f)
            jointConfig[i] = jpos
            
            F[num] = f
            V[num] = v[2]
              
        ##控制命令需要同时方式，故暂停通信，用于存储所有控制命令一起发送    
        vrep.simxPauseCommunication(clientID, True)
        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(clientID, jointHandle[i], x, vrep.simx_opmode_oneshot) # 120/RAD2EDG
        vrep.simxPauseCommunication(clientID, False)      
        # print(x)
        
        lastCmdTime = currCmdTime              # 记录当前时间
        vrep.simxSynchronousTrigger(clientID)  # 进行下一步
        vrep.simxGetPingTime(clientID)         # 使得该仿真步走完
    clientID = -2


#########################  calculate energy  ########################
print("获得的关节力矩为：",F)
t0 = 0                        #时间为零
t1 = N                        #时间步 步数
t  = np.linspace(0,N,num=101) #10个值
tau = F                       #101个值
theta_differ = V              #101个值

ene = Energy_model_lisan.Energy(t0,t1,tau,theta_differ)
E = ene.function()
print("最终能耗为：",E)

