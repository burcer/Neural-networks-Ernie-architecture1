import numpy as np,h5py 
import scipy.io
from nn.layers import *
from nn.layer_utils import *
from nn.layer_utils import affine_sigmoid_forward
from nn.layer_utils import affine_pwlsig2_backward


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,conv_size=256,
               weight_scale=1e-3, reg=0.0,scores_activation=0, activation=0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.scores_activation=scores_activation
    self.activation=activation
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params = {}
    self.params['W1_1'] = weight_scale * np.random.randn(conv_size, hidden_dim/4)
    self.params['W1_2'] = weight_scale * np.random.randn(conv_size, hidden_dim/4)
    self.params['W1_3'] = weight_scale * np.random.randn(conv_size, hidden_dim/4)
    self.params['W1_4'] = weight_scale * np.random.randn(conv_size, hidden_dim/4)

    self.params['b1_1'] = np.zeros(hidden_dim/4)
    self.params['b1_2'] = np.zeros(hidden_dim/4)
    self.params['b1_3'] = np.zeros(hidden_dim/4)
    self.params['b1_4'] = np.zeros(hidden_dim/4)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None,noise=0,noise2=0,test=0,parallel_samples_output=1):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None

    W1_1 = self.params['W1_1']
    W1_2 = self.params['W1_2']
    W1_3 = self.params['W1_3']
    W1_4 = self.params['W1_4']

    W2 = self.params['W2']
    #hidden, hidden_cache = affine_sigmoid_forward(X, W1, self.params['b1'],noise=noise)

    X1=X[:,0:16,0:16]
    X2=X[:,12:28,0:16]
    X3=X[:,0:16,12:28]
    X4=X[:,12:28,12:28]

    hidden_cache1 = []
    hidden_cache2 = []
    hidden_cache3 = []
    hidden_cache4 = []
    out1, tmp1 = affine_forward(X1, self.params['W1_1'], self.params['b1_1'])
    out2, tmp2 = affine_forward(X2, self.params['W1_2'], self.params['b1_2'])
    out3, tmp3 = affine_forward(X3, self.params['W1_3'], self.params['b1_3'])
    out4, tmp4 = affine_forward(X4, self.params['W1_4'], self.params['b1_4'])

    out=np.hstack([out1,out2,out3,out4])

    hidden_cache1.append(tmp1)
    hidden_cache2.append(tmp2)
    hidden_cache3.append(tmp3)
    hidden_cache4.append(tmp4)


    if noise ==0:
      if self.activation ==0:
        hidden, tmp = relu_forward(out) #change to 
      if self.activation ==1:
        hidden, tmp = sigmoid_forward(out) #change to 
      if self.activation ==2:
        hidden, tmp = pwlsig_forward(out) #change tos 
      if self.activation ==3:
        hidden1, tmp1 = tanh_forward(out1) #change to 
        hidden2, tmp2 = tanh_forward(out2) #change to 
        hidden3, tmp3 = tanh_forward(out3) #change to 
        hidden4, tmp4 = tanh_forward(out4) #change to 

      if self.activation ==4:
        hidden, tmp = pwlsig2_forward(out) #change to 

      hidden=np.hstack([hidden1,hidden2,hidden3,hidden4])

      hidden_cache1.append(tmp1)
      hidden_cache2.append(tmp2)
      hidden_cache3.append(tmp3)
      hidden_cache4.append(tmp4)

    if noise ==1:
      #mat=np.array(scipy.io.loadmat('/home/burc/MATLAB/noise.mat'))
      f = h5py.File('/home/burc/MATLAB/myfilesample1fullresetnonoisevth06delta05.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample2fullresetnonoisevth06delta12.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample3fullresetnonoisevth06delta13.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample4fullresetnonoisevth06delta13.h5','r') 
      data = f.get('dataset1') 
      data = np.transpose(np.array(data)) 

      indices=np.array(np.clip(np.round(50*out)+250,1,500))-1
      randsel=10000*np.random.rand(10000,256)+1
      randint=np.array(np.floor(randsel))-1
      #print data.shape
      #print indices.astype(int).shape
      #print randint.astype(int).shape
      #print out[45,24:29]
      hidden = data[indices.astype(int),randint.astype(int)]
      #print hidden[45,24:29]
      #print  2*sigmoid(out[45,24:29])-1
    scores, score_cache = affine_forward(hidden, W2, self.params['b2'])
    if self.scores_activation >0 and noise2==0:
	if self.scores_activation ==1:
	  scores, score_cache2 = sigmoid_forward(scores) #changed
	if self.scores_activation ==2:
	  scores, score_cache2 = pwlsig_forward(scores) #changed
	if self.scores_activation ==3:
	  scores, score_cache2 = tanh_forward(scores) #changed

    if noise2 ==1:
      #f = h5py.File('/home/burc/MATLAB/myfilesample1fullresetnonoiseVth06.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample2fullresetnoise06Vth06.h5','r') 
      f = h5py.File('/home/burc/MATLAB/myfilesample1fullresetnoise06vth06delta05.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample2fullresetnoise04vth06delta12.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample3fullresetnoise04vth06delta13.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample4fullresetnoise05vth06delta12.h5','r') 
      data = f.get('dataset1') 
      data = np.transpose(np.array(data)) 
      indices=np.array(np.clip(np.round(50*scores)+250,1,500))-1
      randsel=10000*np.random.rand(10000,10)+1
      randint=np.array(np.floor(randsel))-1
      #print data.shape
      #print indices.astype(int).shape
      #print randint.astype(int).shape
      #print scores[45,0:9]
      scores2 = data[indices.astype(int),randint.astype(int)]
       ##following implements parallel sampling under presence of noise:
     
      if parallel_samples_output >1:
	for k in range(0,parallel_samples_output-1):
          indices=np.array(np.clip(np.round(50*scores)+250,1,500))-1
          randsel=10000*np.random.rand(10000,10)+1
          randint=np.array(np.floor(randsel))-1
          scores2 += data[indices.astype(int),randint.astype(int)]
      scores2 /= parallel_samples_output
      scores=scores2
      ##
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None or test==1:
      #if test == 1:
        #print scores[3,:]
        
        return scores
    
    loss, grads = 0, {}

    loss, dscores = softmax_loss(scores, y)
    if self.scores_activation==1:
	dscores = sigmoid_backward(dscores, score_cache2) #changed
    if self.scores_activation==2:
	dscores = pwlsig_backward(dscores, score_cache2) #changed	
    if self.scores_activation==3:
	dscores = tanh_backward(dscores, score_cache2) #changed

    dhidden, dW2, db2 = affine_backward(dscores, score_cache)


    if self.activation ==0:
      dx, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)   #change to affine_sigmoid later
    if self.activation ==1:
      dx, dW1, db1 = affine_sigmoid_backward(dhidden, hidden_cache)   #change to affine_sigmoid later
    if self.activation ==2:
      dx, dW1, db1 = affine_pwlsig_backward(dhidden, hidden_cache)   #change to affine_sigmoid later
    if self.activation ==3:
      dx1, dW1_1, db1_1 = affine_tanh_backward(dhidden[:,0:64], hidden_cache1)   #change to affine_sigmoid later
      dx2, dW1_2, db1_2 = affine_tanh_backward(dhidden[:,64:128], hidden_cache2)  
      dx3, dW1_3, db1_3 = affine_tanh_backward(dhidden[:,128:192], hidden_cache3)  
      dx4, dW1_4, db1_4 = affine_tanh_backward(dhidden[:,192:256], hidden_cache4)        

    if self.activation ==4:
      dx, dW1, db1 = affine_pwlsig2_backward(dhidden, hidden_cache)   #change to affine_sigmoid later
    


    

    reg = self.reg
    loss += 0.5 * reg * (np.sum(W1_1*W1_1) + np.sum(W2*W2) +np.sum(W1_2*W1_2)+np.sum(W1_3*W1_3)+np.sum(W1_4*W1_4) )
    dW1_1 += reg * W1_1
    dW1_2 += reg * W1_1
    dW1_3 += reg * W1_1
    dW1_4 += reg * W1_1
    dW2 += reg * W2
    
    grads['W1_1'] = dW1_1
    grads['W1_2'] = dW1_2
    grads['W1_3'] = dW1_3
    grads['W1_4'] = dW1_4

    grads['W2'] = dW2
    grads['b1_1'] = db1_1
    grads['b1_2'] = db1_2
    grads['b1_3'] = db1_3
    grads['b1_4'] = db1_4
    grads['b2'] = db2


    return loss, grads



class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,activation=0,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None,external_load=0,scores_activation=0):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.hidden_dims=hidden_dims
    self.external_load=external_load
    self.scores_activation=scores_activation
    self.activation=activation

    if self.external_load==0:
    	self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
    	self.params['b1'] = np.zeros(hidden_dims[0])
    if self.external_load==1:
	self.params['W1']=np.load('W1ext.npy')
	self.params['b1']=np.load('b1ext.npy')

    #self.params=np.load()
    if self.use_batchnorm:
        self.params['gamma1'] = np.ones([hidden_dims[0]])
        self.params['beta1'] = np.zeros([hidden_dims[0]])
    for i in range(2,self.num_layers):
        self.params['W'+ str(i)] = weight_scale * (np.random.randn(hidden_dims[i-2], hidden_dims[i-1]))
        self.params['b' + str(i)] = np.zeros(hidden_dims[i-1])
        if self.use_batchnorm:
            self.params['gamma' + str(i)] = np.ones([hidden_dims[i-1]])
            self.params['beta' + str(i)] = np.zeros([hidden_dims[i-1]])
    self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    self.params['b' + str(self.num_layers)] = np.zeros(num_classes)


    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None,noise=0, noise2=0,test=0,parallel_samples_output=1):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None


    cache = []
    out, tmp = affine_forward(X, self.params['W1'], self.params['b1'])
    cache.append(tmp)
    for i in range(2,self.num_layers+1):
        if self.use_batchnorm:
            out, tmp = batchnorm_forward(out, self.params['gamma' + str(i-1)],
                                         self.params['beta' + str(i-1)], self.bn_params[i-2])
            cache.append(tmp)

        if noise ==0:
	  if self.activation ==0:
            out, tmp = relu_forward(out) #changed
	  if self.activation ==1:
            out, tmp = sigmoid_forward(out) #changed
	  if self.activation ==2:
            out, tmp = pwlsig_forward(out) #changed
          cache.append(tmp)

        if noise ==1:
          f = h5py.File('/home/burc/MATLAB/myfilesample2fullresetnonoisevth06delta12.h5','r')
          #f = h5py.File('/home/burc/MATLAB/myfilesample1fullresetnonoisevth06delta05.h5','r') 
          data = f.get('dataset1') 
          data = np.transpose(np.array(data)) 
          print data.shape
          indices=np.array(np.clip(np.round(50*out)+250,1,500))-1
          randsel=10000*np.random.rand(10000,self.hidden_dims[i-2])+1
          randint=np.array(np.floor(randsel))-1
          out = data[indices.astype(int),randint.astype(int)]

        if self.use_dropout:
            out, tmp = dropout_forward(out, self.dropout_param)
            cache.append(tmp)
        out, tmp = affine_forward(out, self.params['W'+str(i)], self.params['b'+str(i)])
        cache.append(tmp)


    if self.scores_activation > 0 and noise2==0:
        if self.scores_activation==1:
	  out, tmp = sigmoid_forward(out) #changed
        if self.scores_activation==2:
	  out, tmp = pwlsig_forward(out) #changed
        cache.append(tmp)

    if noise2 ==1:
      f = h5py.File('/home/burc/MATLAB/myfilesample2fullresetnoise04vth06delta12.h5','r') 
      #f = h5py.File('/home/burc/MATLAB/myfilesample1fullresetnoise06vth06delta05.h5','r') 
      data = f.get('dataset1') 
      data = np.transpose(np.array(data)) 
      indices=np.array(np.clip(np.round(50*out)+250,1,500))-1
      randsel=10000*np.random.rand(10000,10)+1
      randint=np.array(np.floor(randsel))-1
      #print data.shape
      #print indices.astype(int).shape
      #print randint.astype(int).shape
      #print scores[45,0:9]
      scores2 = data[indices.astype(int),randint.astype(int)]
       ##following implements parallel sampling under presence of noise:
     
      if parallel_samples_output >1:
	for k in range(0,parallel_samples_output-1):
          indices=np.array(np.clip(np.round(50*scores)+250,1,500))-1
          randsel=10000*np.random.rand(10000,10)+1
          randint=np.array(np.floor(randsel))-1
          scores2 += data[indices.astype(int),randint.astype(int)]
      scores2 /= parallel_samples_output
      out=scores2

    scores = out

    


    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    reg = self.reg
    loss, dscores = softmax_loss(scores, y)
    if self.scores_activation > 0 and noise2==0:
	if self.scores_activation == 1:
    	  dx = sigmoid_backward(dscores, cache.pop()) #changed
	if self.scores_activation == 2:
    	  dx = pwlsig_backward(dscores, cache.pop()) #changed
	dx, dW, db = affine_backward(dx, cache.pop())
    if self.scores_activation == 0:
	dx, dW, db = affine_backward(dscores, cache.pop())
    W = self.params['W'+str(self.num_layers)]
    dW += reg * W
    loss += 0.5 * reg * np.sum(W*W)
    grads['W'+str(self.num_layers)] = dW
    grads['b'+str(self.num_layers)] = db
    


    for i in range(self.num_layers-1,0,-1):
        if self.use_dropout:
            dx = dropout_backward(dx, cache.pop())
	if self.activation ==0:        
	  dx = relu_backward(dx, cache.pop()) #changed
	if self.activation ==1:        
	  dx = sigmoid_backward(dx, cache.pop()) #changed
	if self.activation ==2:        
	  dx = pwlsig_backward(dx, cache.pop()) #changed
        if self.use_batchnorm:
            dx, dgamma, dbeta = batchnorm_backward(dx, cache.pop())
            grads['gamma' + str(i)] = dgamma
            grads['beta' + str(i)] = dbeta
        dx, dW, db = affine_backward(dx, cache.pop())
        W = self.params['W'+str(i)]
        dW += reg * W
        loss += 0.5 * reg * np.sum(W*W)
        grads['W'+str(i)] = dW
        grads['b'+str(i)] = db


    return loss, grads
