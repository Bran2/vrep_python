import numpy as np
import math
import matplotlib.pyplot as plt

## r=x^2+y^2;
## w=(((1-beta)*ws/beta)/(exp(-a*y)+1)+ws/(exp(a*y)+1))

class CPG_Single: 
    def __init__(self):    
        self.u1 = 0                 ## alpha  用于控制振荡收敛到极限环的速度
        self.u2 = 0                 ## ws = input('输入摆动相频率ws=')
        self.u = 1                  ## u      决定振荡器的幅值
        self.beta = 0.5             ## u1,u2 外部反馈项
        self.ws = 1*math.pi         ## w     振荡器的频率
                                    ## C，D  自变量的取值范围
        self.alpha=5                ## N     迭代次数
        self.a=50                   ## ws    摆动相频率
        self.w=2*np.pi              ## beta  负载因子（占空比）
        self.C=0                    ## syms x y t u  u1 u2 w beta ws 
        self.D=10                   ## u1 = input('输入外部反馈项u1=')
        self.N=200                  ## u2 = input('输入外部反馈项u2=')
                                    ## u = input('输入参数u=')
                                    ## beta = input('输入负载因子beta=')    

        self.h = (self.D-self.C)/self.N

        
    def f(self,t,x,y):
        k = self.alpha*(self.u-x**2-y**2)*(x-self.u1)-(((1-self.beta)*self.ws/self.beta)/(np.exp(-self.a*y)+1)+self.ws/(np.exp(self.a*y)+1))*(y-self.u2)
        return k
    def g(self,t,x,y):
        l = self.alpha*(self.u-x**2-y**2)*(y-self.u2)+(((1-self.beta)*self.ws/self.beta)/(np.exp(-self.a*y)+1)+self.ws/(np.exp(self.a*y)+1))*(x-self.u1)
        return l
##      f=@(t,x,y) alpha*(u-x^2-y^2)*(x-u1)-(((1-beta)*ws/beta)/(exp(-a*y)+1)+ws/(exp(a*y)+1))*(y-u2);
##      g=@(t,x,y) alpha*(u-x^2-y^2)*(y-u2)+(((1-beta)*ws/beta)/(exp(-a*y)+1)+ws/(exp(a*y)+1))*(x-u1);
        
##      T=C:h:D;

    def runge_kutta(self):
        T = np.zeros((1,self.N+1)).ravel()
        X = np.zeros((1,self.N+1)).ravel()
        Y = np.zeros((1,self.N+1)).ravel()
        X[0] = -0.6
        Y[0] = -0.5
        
        for i in np.arange(self.N):
            k1 = self.f( T[i], X[i], Y[i])  # k1=feval(f,T(i),X(i),Y(i));
            l1 = self.g( T[i], X[i], Y[i])  # l1=feval(g,T(i),X(i),Y(i));
            k2 = self.f( T[i]+self.h/2, X[i]+(self.h/2)*k1, Y[i]+(self.h/2)*l1)  # k2=feval(f,T(i)+h/2,X(i)+(h/2)*k1,Y(i)+(h/2)*l1);
            l2 = self.g( T[i]+self.h/2, X[i]+(self.h/2)*k1, Y[i]+(self.h/2)*l1)  # l2=feval(g,T(i)+h/2,X(i)+(h/2)*k1,Y(i)+(h/2)*l1);
            k3 = self.f( T[i]+self.h/2, X[i]+(self.h/2)*k2, Y[i]+(self.h/2)*l2)  # k3=feval(f,T(i)+h/2,X(i)+(h/2)*k2,Y(i)+(h/2)*l2);
            l3 = self.g( T[i]+self.h/2, X[i]+(self.h/2)*k2, Y[i]+(self.h/2)*l2)  # l3=feval(g,T(i)+h/2,X(i)+(h/2)*k2,Y(i)+(h/2)*l2);
            k4 = self.f( T[i]+self.h, X[i]+self.h*k3, Y[i]+self.h*l3)  #k4=feval(f,T(i)+h,X(i)+h*k3,Y(i)+h*l3);
            l4 = self.g( T[i]+self.h, X[i]+self.h*k3, Y[i]+self.h*l3)  #l4=feval(g,T(i)+h,X(i)+h*k3,Y(i)+h*l3);
            X[i+1] = X[i] + (self.h/6)*(k1+2*k2+2*k3+k4)
            Y[i+1] = Y[i] + (self.h/6)*(l1+2*l2+2*l3+l4)
        return X,Y
     
if __name__ == '__main__':
    CPG = CPG_Single()
    #print(CPG.runge_kutta())
    a = np.arange(0,7.505,0.005)
    x,y = CPG.runge_kutta()
    plt.figure()
    plt.subplot(121)
    plt.plot(a, x)
    plt.plot(a, y,'r')
    plt.subplot(122)
    plt.plot(x, y)
    plt.show()
