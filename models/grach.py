

from arch import arch_model





class BaseModel():

    def __init__(self,p,q,mean,vol,dist,n):
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.n = n

        self.model = arch_model(df['spread'], p=1, q=1, mean='constant', vol='GARCH', dist='normal') 

        
        

    def fit(self):
        pass 
