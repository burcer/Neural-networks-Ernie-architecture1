##Sukru Burc Eryilmaz
##Trains a 4-layer FC neural network with several options in terms of activation quantization, noisy ensembles, 
##noisy ensembles, and splitting the FC layers into a number of (2-8) locally connected independent blocks (towers)
##This particular code uses MNIST dataset, but CIFAR10 is also available by changing the data input size in model 
##and get data command to CIFAR10 case.


import numpy as np
import matplotlib.pyplot as plt
##This particular code implements splitted FC layers (FC layers splitted to independent 2-8 locally connected layers). If
##full FC layer is desired, use 'from nn.classifiers.fc_net_nosplit3 import *' instead of 'from nn.classifiers.fc_net_split3_3 import *'
from nn.classifiers.fc_net_split3_3 import *
##
from nn.data_utils import get_CIFAR10_data
from nn.data_utils_mnist import *
from nn.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

### for auto-reloading external modules
### see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
##%load_ext autoreload
##%autoreload 2

##load MNIST data.  xxx_trinary=1 indicates that the corresponding portion of data (i.e. test data for test_trinary=1)
##is quantized with no_of_levels levels. Here all data is quantized to 3 levels (ternary). You can quantize data to any
##number of levels, provided it is a positive integer larger than 1.

data = get_MNIST_data(test_trinary=1,train_trinary=1,val_trinary=1,noise_amplitude=0.0,no_of_levels=3)
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

X=data['X_train']
print X.shape
#hyperparameter search
##In FourLayerNet class:
##weight_scale: defines the standard deviation for initialization. 
##reg argument: weight decay coefficient
##scores_activation: defines the activation function used for top score layer. Options are linear, relu, tanh, sigmoid,
##noisy tanh quantization with the number of levels equal to 'levels' argument
##activation: same as above, but for hidden layers
##external_load: determines if the bottom N-1 layers are externally loaded from a pretrained network or all the N layers are
##trained. When it is 1, weight_scale is ignored
##levels: number of quantization levels in activation
##see the class definitions for FourLayerNet and Solver to understand the details. 
for i in range(100):

    regu=10**np.random.uniform(-4,-1)
    learning_rate = 10**np.random.uniform(-4,-2)
    weight_scale = 10**np.random.uniform(-3,-1)
    dec = np.random.uniform(0.95,1)

    model = FourLayerNet(input_dim=28*28, 
              weight_scale=weight_scale, reg=regu, activation=3, scores_activation=3,external_load=1,levels=3)
    solver = Solver(model,data,
                print_every=10, num_epochs=60, batch_size=100,
                update_rule='sgd_momentum',lr_decay=dec,verbose = False,
                optim_config={
                  'learning_rate': learning_rate
                }
         )
    
    solver.train()

    print 'lr=%f, std=%f, regu=%f ,dec= %f train accuracy is : %f val acc:  %f' %(learning_rate, weight_scale, regu, dec,solver.train_acc_history[-1], solver.val_acc_history[-1])
 

    model = FourLayerNet(input_dim=28*28,  
              weight_scale=weight_scale, reg=regu,activation=3, scores_activation=3,external_load=1,levels=3)
    solver = Solver(model,data,
                print_every=4000, num_epochs=60, batch_size=100, 
                update_rule='sgd_momentum',lr_decay=1.0,verbose =False, 
                optim_config={
                  'learning_rate': learning_rate
                }
         )
    solver.train()
    print ' no decay lr=%f, std=%f,regu=%f train accuracy is : %f val acc:  %f' %(learning_rate, weight_scale,regu, solver.train_acc_history[-1], solver.val_acc_history[-1])
 


#solver.train()

plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')


#optimum onfiguration found: lr=0.001246, std=0.007923, regu=0.003754 ,dec= 0.985792 train accuracy is : 0.969000 val acc:  0.959800
#retrain with optimum configuration
regu=0.003754
learning_rate = 0.001246
weight_scale = 0.007923
dec = 0.985792 

model = FourLayerNet(input_dim=28*28, ##hidden_dim=256,
              weight_scale=weight_scale, reg=regu, activation=3, scores_activation=3,external_load=1,levels=3)
solver = Solver(model,data,
                print_every=10, num_epochs=60, batch_size=100,
                update_rule='sgd_momentum',lr_decay=dec,verbose = False,
                optim_config={
                  'learning_rate': learning_rate
                }
         )
    
solver.train()

print 'lr=%f, std=%f, regu=%f ,dec= %f train accuracy is : %f val acc:  %f' %(learning_rate, weight_scale, regu, dec,solver.train_acc_history[-1], solver.val_acc_history[-1])
 
#this piece of code experiments with classification accuracy by quantizing activations, 
##introducing noise, using noisy ensembles at the class layer to mitigate degradation due to quantization
##noise =1 indicates noise is introduced during quantizing activations. noise2=1 indicates that noise is introduced 
##at the softmax layer during quantization, and a number of ensembles are averaged with this noise at the softmax layer 
##to find the final class scores. the number of ensembles averaged at softmax layer are given by the argument 
##parallel_samples_output. Note in the results that more ensembles result in better accuracy.

y_test_pred = np.argmax(model.loss(data['X_test'],noise=0,test=1), axis=1)
y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1), axis=1)
print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()

par=1
aa=0.0
for i in range(0,20):
  y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1,noise2=1, parallel_samples_output=par), axis=1)
  #print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()
  aa=aa+ (y_test_pred == data['y_test']).mean()
print aa/20

par=2
aa=0.0
for i in range(0,20):
  y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1,noise2=1, parallel_samples_output=par), axis=1)
  #print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()
  aa=aa+ (y_test_pred == data['y_test']).mean()
print aa/20


par=3
aa=0.0
for i in range(0,20):
  y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1,noise2=1, parallel_samples_output=par), axis=1)
  #print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()
  aa=aa+ (y_test_pred == data['y_test']).mean()
print aa/20

par=4
aa=0.0
for i in range(0,20):
  y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1,noise2=1, parallel_samples_output=par), axis=1)
  #print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()
  aa=aa+ (y_test_pred == data['y_test']).mean()
print aa/20

par=8
aa=0.0
for i in range(0,20):
  y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1,noise2=1, parallel_samples_output=par), axis=1)
  #print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()
  aa=aa+ (y_test_pred == data['y_test']).mean()
print aa/20

par=16
aa=0.0
for i in range(0,20):
  y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1,noise2=1, parallel_samples_output=par), axis=1)
  #print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()
  aa=aa+ (y_test_pred == data['y_test']).mean()
print aa/20

par=24
aa=0.0
for i in range(0,20):
  y_test_pred = np.argmax(model.loss(data['X_test'],noise=1,test=1,noise2=1, parallel_samples_output=par), axis=1)
  #print 'Noisy test set accuracy: ', (y_test_pred == data['y_test']).mean()
  aa=aa+ (y_test_pred == data['y_test']).mean()
print aa/20

##toggle the comment for the following piece of code to save the model variables.

#np.save('W1_1ext784-256-256-10-3level', model.params['W1_1'])
#np.save('b1_1ext784-256-256-10-3level', model.params['b1_1'])
#np.save('W1_2ext784-256-256-10-3level', model.params['W1_2'])
#np.save('b1_2ext784-256-256-10-3level', model.params['b1_2'])
#np.save('W1_3ext784-256-256-10-3level', model.params['W1_3'])
#np.save('b1_3ext784-256-256-10-3level', model.params['b1_3'])
#np.save('W1_4ext784-256-256-10-3level', model.params['W1_4'])
#np.save('b1_4ext784-256-256-10-3level', model.params['b1_4'])
#np.save('W2ext784-256-256-10-3level', model.params['W2'])
#np.save('b2ext784-256-256-10-3level', model.params['b2'])


