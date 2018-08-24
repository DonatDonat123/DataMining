# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 14:41:27 2017

@author: Lisa Tostrams
Class for building a MultiLayer Perceptron with one hidden layer, that mimimizes an objective function

Basic call for the xor data:
    import scipy.io
    mat = scipy.io.loadmat('Data/xor.mat')
    X = mat['X']
    y = mat['y']
    perc = MLP(X,y)
    w1,w2,c = perc.learn_weights(nhidden=2)
    perc.plot_boundaries(w1,w2,X)

"""


import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, X,y,binary=True):
        """MLP(X,y), builds 3 layer MLP on data X using labels y, minimizing the error function:
            MSE(X,y,W) = 1/N * sum_i s(W_o*s(W_h*X_i)) - y)^2
            by moving along the gradient 
        optional argument:
            binary -> whether or not output should be binary instead of continuous"""
        # add bias term
        self.X = X.T
        self.X = np.vstack([np.ones([1,self.X.shape[1]]),self.X])
        self.y=y.T
        self.binary=binary
        self.b = 0.5 #initialise decision boundary
        
    def sigmoid(self,x):
        """ Sigmoid(x), returns value and gradient; only implemented sigmoid activation due to convenient grad"""
        s = 1.0 / (1 + np.exp(-x))
        grad_s = s * (1 - s)
        return s, grad_s
    
    def class_error(self,y,y_hat):
        """ class_error(y,y_hat), returns the misclassification rate using output y_hat to label data points compared to y """
        return 1-np.mean((y==y_hat).flatten())
    
    def forwardprop(self,W_h, W_o,**kwargs):
        """forwardprop(Wh,Wo,X), forward propagation of the data through hidden layer (W_h) and output layer (W_o)
        optional input:
            X -> when an instance of X is given, add bias terms and feed it through
        else:
            h -> output of hidden units
            o -> output of output units
            grad_h -> gradient of the hidden units wrt weights 
            grad_o -> gradient of the output units wrt weights  
            y_hat -> output of network, when labels are not binary y_hat == o
            """
        if('X' in kwargs.keys()):
            tmp = kwargs['X']
            tmp = tmp.T
            X = np.vstack([np.ones([1,tmp.shape[1]]),tmp])
        else:
            X = self.X    
        activation_h = np.dot(W_h, X)
        h, grad_h = self.sigmoid(activation_h) 
        activation_o = np.dot(W_o, h)    
        o, grad_o = self.sigmoid(activation_o)
        y_hat = o
        if(self.binary):
            y_hat = o > self.b
        return h, o, grad_h, grad_o, y_hat
    
    def backprop(self,h,o, grad_h, grad_o,W_o,X):
        """ backprop(h,o,grad_h,grad_o,W_o,X), propagates error of output o compared to labels y back through layers,
                    computes gradient in objective function MSE(X,y,W) wrt weights in both layers
        output:
            grad_E_h -> gradient of MSE(X,y,W) wrt W_h
            grad_E_o -> gradient of MSE(X,y,W) wrt W_o   """
        error_output = (o - self.y) * grad_o     
        error_hidden = grad_h *(np.dot(W_o.T, error_output))
        grad_E_o = np.dot(error_output, h.T)     
        grad_E_h = np.dot(error_hidden, X.T)
        return grad_E_h, grad_E_o
    
    def learn_weights(self,nhidden=1,nepochs=8000,eta=0.1,verbose=True):
        """ learn_weights(), learns the weights in each layer
        optional arguments:
            nhidden -> number of units in hidden layer
            nepochs -> number of learning steps
            eta -> learning rate 
            verbose -> whether to print progress
        """  
        ninput = self.X.shape[0]
        noutput = self.y.shape[0] 
        W_h = np.random.uniform(0, 1, [nhidden,ninput])
        W_h[:,0]*=-1
        W_o = np.random.uniform(-1, 1, [noutput,nhidden])
        for epoch in xrange(0,nepochs):
            """ Forward propagation step: compute the activations and the activiation gradients wrt the weights
            """
            h,o,grad_h,grad_o,y_hat = self.forwardprop(W_h, W_o)
            self.b = np.mean(o) #seperate output into 2 classes 
            
            """ Backward propagation step: use the activation gradients to figure 
                out in which directions the error gradients points wrt the weights
            """
            grad_E_h, grad_E_o = self.backprop(h, o, grad_h, grad_o,W_o,self.X)
            if(epoch%500==0 and verbose):
                print('Iteration: {} / {} ; misclassication rate: {:.4f}'.format(epoch,nepochs,self.class_error(self.y,y_hat))) 
                
            
            """ Update the weights
            """
            W_h = W_h - eta * grad_E_h                        
            W_o = W_o - eta * grad_E_o

        y_hat=self.forwardprop(W_h, W_o)[4]                                                                      
        self.b = np.mean(o)
        error = self.class_error(self.y,y_hat)
        if(verbose):
            print('Final misclassification rate: {:.4f}'.format(error))
        return W_h,W_o,error
        
    
    def plot_boundaries(self,W1,W2,Data):
        """ plot_boundries(Wh,Wo,X), plots the decision boundaries of the MLP with weights Wh and Wo and where data X is placed
            kind of a weird function to have user input instead of using internal variables
            but want to leave something to do for users"""
        y_hat = self.forwardprop(W1,W2,X=Data)[4]
        x0 = np.arange(min(self.X[1,:])-0.2, max(self.X[1,:])+0.2, 0.1)
        x1 = np.arange(min(self.X[2,:])-0.2, max(self.X[2,:])+0.2, 0.1)
        xx, yy = np.meshgrid(x0, x1, sparse=False)
        space = np.asarray([xx.flatten(),yy.flatten()]).T
        z = self.forwardprop(W1,W2,X=space)[1]
        tmp = Data.T
        plt.scatter(tmp[0,y_hat.flatten()],tmp[1,y_hat.flatten()],label='1')
        plt.scatter(tmp[0,(y_hat==False).flatten()],tmp[1,(y_hat==False).flatten()],label='0')
        h = plt.contourf(x0,x1,np.reshape(z,[len(x0),len(x0)]),levels=[0,self.b,1],colors=('orange','b'), alpha=0.1)
        plt.title('Decision boundaries resulting from given weights')
        plt.legend()
        plt.show()