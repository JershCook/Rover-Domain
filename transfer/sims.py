import numpy as np

class predpred:

    def __init__(self,npred,nprey,speed):
        self.npred=npred
        self.nprey=nprey
        self.speedprey=speed
        self.speedpred=1.0
        self.init_pred=np.random.normal(0,0.1,(2,npred))
        self.init_pred_dir=np.random.random(npred)
        self.init_prey=np.random.normal(2.0,0.1,(2,nprey))

        self.sight=5.0

    def reset(self):
        self.pred_loc=self.init_pred.copy()
        self.prey_loc=self.init_prey_dir.copy()
        self.pred_dir=self.init_pred.copy()
        self.prey_dir=self.init_prey_dir.copy()

    def closest(self,loc):
        d=1e9
        close=None
        x,y=loc
        for i in range(self.npred):
            X,Y=self.pred_loc[i]
            D=(X-x)**2+(Y-y)**2
            if D<d:
                d=D:
                close=[X,Y,D]
        return close

    def step(self,action):
        for i in range(self.npred):
            act=action[i]
            dt,speed=act
            speed=max(0,speed)
            self.pred_dir[i]+=dt*0.5
            self.pred_loc[i][0]+=speed*np.cos(self.pred_dir[i]*2*np.pi)
            self.pred_loc[i][1]+=speed*np.sin(self.pred_dir[i]*2*np.pi)
        self.pred_dir=np.remainder(self.pred_dir,1.0)
        for i in range(self.nprey):




