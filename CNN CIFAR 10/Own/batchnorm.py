import numpy as np
from adam_optimizer import Adam


class BatchNormalization:
    def __init__(self,eps=1e-5,momentum=0.9):
        
        self.eps = eps
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        
        
        self.trainable_params = False
        
        self.Beta_opt = None
        self.Gamma_opt = None
       
    
    def forward(self,X,mode="training"):
        
        if self.gamma is None:
            self.gamma = np.zeros(X.shape[1:])
        if self.beta is None:
            self.beta = np.zeros(X.shape[1:])
        if self.running_mean is None:
            self.running_mean = np.zeros(X.shape[1:])
        if self.running_var is None:
            self.running_var = np.zeros(X.shape[1:])
            
        eps = self.eps
        momentum = self.momentum
        running_mean = self.running_mean
        running_var = self.running_var
        gamma = self.gamma
        beta = self.beta
        
        if mode == "training":
            mu = 1/float(X.shape[0]) * np.sum(X,axis=0)
            xmu = X - mu
            carre = xmu**2
            var = 1 / float(X.shape[0]) * np.sum(carre, axis=0)
            sqrtvar = np.sqrt(var + eps)
            invvar = 1. / sqrtvar
            va2 = xmu * invvar
            va3 = gamma * va2
            out = va3 + beta
            running_mean = momentum * running_mean + (1.0 - momentum) * mu
            running_var = momentum * running_var + (1.0 - momentum) * var
            
            self.gamma = gamma
            self.beta = beta
            self.running_mean = running_mean
            self.running_var = running_var
            self.momentum = momentum
            self.eps = eps
            
            self.mu = mu
            self.xmu = xmu
            self.carre = carre
            self.var = var
            self.sqrtvar = sqrtvar
            self.invvar = invvar
            self.va2 = va2
            self.va3 = va3
            self.X = X
            
            
        else:
            mu = running_mean
            var = running_var
            xhat = (X - mu) / np.sqrt(var + eps)
            out = gamma * xhat + beta
            
            self.gamma = gamma
            self.beta = beta
            self.running_mean = running_mean
            self.running_var = running_var
            self.momentum = momentum
            self.eps = eps
            self.mu = mu
            self.var = var
            
        return out
            
            
    def backward(self,dout,mode="training"):
        dx, dgamma, dbeta = None, None, None
        mu = self.mu
        xmu = self.xmu
        carre = self.carre
        var = self.var
        sqrtvar = self.sqrtvar
        invvar = self.invvar
        va2 = self.va2
        va3 = self.va3
        gamma = self.gamma
        beta = self.beta
        X = self.X
        eps = self.eps
        momentum = self.momentum
        running_mean = self.running_mean
        running_var = self.running_var
        
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum((X - mu) * (var + eps)**(-1. / 2.) * dout, axis=0)
        dx = (1. / X.shape[0]) * gamma * (var + eps)**(-1. / 2.) * (X.shape[0] * dout - np.sum(dout, axis=0)
                                                       - (X - mu) * (var + eps)**(-1.0) * np.sum(dout * (X - mu), axis=0))
        
        if self.Beta_opt is None:
            self.Beta_opt = Adam(dbeta)
        if self.Gamma_opt is None:
            self.Gamma_opt = Adam(dgamma)
            
        self.gamma = self.Gamma_opt.update(self.gamma,dgamma)
        self.beta = self.Beta_opt.update(self.beta,dbeta)
       
        return dx
        
   
