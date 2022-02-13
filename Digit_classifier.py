import numpy as np 
import pandas as pd 
import random

#training dataset
dataset=pd.read_csv(r'mnist_train.csv')
data=np.array(dataset)

#test dataset
dataset_test=pd.read_csv(r'mnist_test.csv')
data_test=np.array(dataset_test)

#dimensions of train dataset
m,n=data.shape
alpha=0.01

#training input and expected output
data=data.T
Y=data[0]
X=data[1:m]
X=X/255

#test input and expected output
data_test=data_test.T
Y_test=data_test[0]
X_test=data_test[1:m]
X_test=X_test/255

#convert output labels of training data to onehot matrix
one_hot=np.zeros((Y.size,Y.max()+1))
one_hot[np.arange(Y.size),Y]=1
Y=one_hot.T

#convert output labels of test data to onehot matrix
one_hot_test=np.zeros((Y_test.size,Y_test.max()+1))
one_hot_test[np.arange(Y_test.size),Y_test]=1
Y_test=one_hot_test.T

#initialise weights and biases
W1=np.random.rand(10,784)-0.5
B1=np.random.rand(10,1)-0.5
W2=np.random.rand(10,10)-0.5
B2=np.random.rand(10,1)-0.5

#softmax activation function
def softmax(layer):
    return np.exp(layer)/sum(np.exp(layer))

#relu activation function
def relu(layer):
    return np.maximum(layer,0)

#derivative of relu activation function
def del_relu(layer):
    return np.where(layer>0,1,0)

#forward pass
def forward_prop(W1,W2,B1,B2,layer1):
    layer2=relu(np.dot(W1,layer1)+B1)
    layer3=softmax(np.dot(W2,layer2)+B2)
    return layer2, layer3

#backward pass to calculate derivations and delta
def backward_prop(W1,W2,B1,B2,A1,A2,layer,Y):
    del_Z2=A2-Y
    del_B2=(1/np.size(del_Z2))*(np.sum(del_Z2))
    del_W2=np.dot(del_Z2,A1.T)
    del_A1=np.dot(del_Z2.T,W2).T
    del_Z1=del_A1*del_relu(A1)
    del_B1=(1/np.size(del_Z1))*(np.sum(del_Z1))
    del_W1=np.dot(del_Z1,layer.T)
    return del_W1, del_B1, del_W2, del_B2

#update weights according to the delta calculated
def update(W1, B1, W2, B2, del_W1, del_B1, del_W2, del_B2, alpha):
    W1 = W1 - alpha * del_W1
    B1 = B1 - alpha * del_B1    
    W2 = W2 - alpha * del_W2  
    B2 = B2 - alpha * del_B2    
    return W1, B1, W2, B2

#training run
count=0
for i in range(50000):
    index=random.randint(0,m-1)
    layer1=X[:,[index]]
    A1,A2=forward_prop(W1,W2,B1,B2,layer1)
    del_W1,del_B1,del_W2,del_B2=backward_prop(W1,W2,B1,B2,A1,A2,layer1,Y[:,[index]])
    W1,B1,W2,B2=update(W1,B1,W2,B2,del_W1,del_B1,del_W2,del_B2,alpha)
    if np.where(A2==np.amax(A2)) == np.where(Y[:,[index]]==np.amax(Y[:,[index]])):
        count+=1
    if i%10000==0:
        print("Accuracy at ",i,"epochs is :",count/100,"%")
        count=0

#test run
count=0
for i in range(10000):
    layer1=X_test[:,[i]]
    A1,A2=forward_prop(W1,W2,B1,B2,layer1)
    if np.where(A2==np.amax(A2)) == np.where(Y_test[:,[i]]==np.amax(Y_test[:,[i]])):
        count+=1
print("Test dataset accuracy :",count/100,"%")