wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
tar -xzvf train-images-idx3-ubyte.gz
tar -xzvf train-labels-idx1-ubyte.gz
tar -xzvf t10k-images-idx3-ubyte.gz
tar -xzvf t10k-labels-idx1-ubyte.gz

rm train-images-idx3-ubyte.gz
rm train-labels-idx1-ubyte.gz
rm t10k-images-idx3-ubyte.gz
rm t10k-labels-idx1-ubyte.gz
