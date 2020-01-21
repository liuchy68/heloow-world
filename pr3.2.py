#### 加载各种库
# 标准库
import pickle
import gzip

# 第三方库
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

# 定义各种激活函数
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### 是否使用GPU跑模型
GPU = False
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True.")

#### 加载 MNIST 数据
def load_data_shared(filename="mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### 用来创建用于训练和预测的网络结构
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """
        输入：
        layers是一个列表，用来保存各种网络的实例引用
        mini_batch_size 批次的大小
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        #将各层参数集合起来捆绑到一个列表中
        self.params = [param for layer in self.layers for param in layer.params]

        #定义了Theano 符号变量x 和y。这些会⽤来表⽰输⼊和⽹络得到的输出
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        ##下面开始定义前向传播过程
        init_layer = self.layers[0]
        #设置初始层（第0层）输入
        #注意输⼊是以每次⼀个mini-batch 的⽅式进⾏的
        #且存在两个self.x 第二个self.x 是为drop_out操作引入的
        #这是因为我们我们可能会以两种⽅式（有dropout 和⽆dropout）使⽤⽹络
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        #遍历所有层
        for j in range(1, len(self.layers)): 
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            #当前层的输入等于上一层的输出
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        
        #layers[-1]表示索引到最后一层
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    #下面开始定义训练过程
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """ 运用小批量SGD训练网络
            参数说明： eta 学习率  lmbda 正侧化因子             
        """

        #将各个数据集分解成：特征数据x 和 标签数据y
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # 计算各个数据集的小批量个数
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        #符号化地定义： 正侧化的对数似然损失函数, 梯度, 更新公式
        #定义权值范数的平方用于正侧化以较少过拟合风险
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])

        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        #求cost对params的梯度，即对权值和偏置的梯度
        grads = T.grad(cost, self.params)
        #在updates将在下面的训练过程中用到，它是由Theano的机制确定的，可以不深究
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        #定义一个训练mini-batch网络的函数，并计算它在交叉验证和测试集的mini-batches准确率

        i = T.lscalar() # 用于遍历每个小批量

      
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

       

        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # 实现训练过程
        best_validation_accuracy = 0.0 #初始化交叉验证准确率
        for epoch in range(epochs):
            #在每一个epoch都要对数据的所有批次作遍历
            for minibatch_index in range(num_training_batches):
                #iteration表示当前已训练的批次数
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))

                cost_ij = train_mb(minibatch_index)

                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("***********************************************************************")

        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### 定义各种网络层

class ConvPoolLayer(object):
    #创建一个卷积和max_池化于一体的网络层
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """
         `filter_shape`是个长度为4的元组，它的元素分别表示：
        （ filters的数目,  输入的feature map 数目，  filter的高，  filter的宽  ）

        `image_shape` 是个长度为4的元组，它的元素分别表示：
        （mini-batch的size，  输入的feature maps数目，  图像的高，   图像的宽   ）

        `poolsize`  是个长度为2的元组，它表示pooling操作的size

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # 初始化权值和偏置
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    #设置网络输入和得到输出
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # 在卷积层中没有dropout操作

class FullyConnectedLayer(object):
    #创建两层全连接层，输入层数：n_in  输出层数：n_out 
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # 初始化权值和偏置
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        #用一个参数将权值和偏置封装在一起，在 Network.SGD 会用到
        self.params = [self.w, self.b]
    #⽤来设置该层的输⼊，并计算相应的输出
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):

        #运用它来计算准确度
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        #T.dot 完成矩阵的乘法
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1) #用于准确率的计算
        
        #运用dropout时用到
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    #准确度计算
    def accuracy(self, y):
        "返回小批量的准确率"
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):
    #和上面的“FullyConnectedLayer”十分类似
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # 初始化权值和偏置
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        #将权值和偏置合并到params中
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "返回对数似然输出"
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "返回小批量的准确率"
        return T.mean(T.eq(y, self.y_out))


def size(data):
    '''返回data 的大小'''
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
