#-*-coding:utf-8-*-
import cPickle
import gzip
import os
import sys
import time
#from ConvPoolLayer import ConvPoolLayer
#from HiddenLayer import HiddenLayer
#from SoftMax import SoftMax
import numpy
import theano
import theano.tensor as T

"""
加载MNIST数据集load_data()
"""
def load_data(dataset):
    # dataset是数据集的路径，程序首先检测该路径下有没有MNIST数据集，没有的话就下载MNIST数据集
    #这一部分就不解释了，与softmax回归算法无关。
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'
#以上是检测并下载数据集mnist.pkl.gz，不是本文重点。下面才是load_data的开始

#从"mnist.pkl.gz"里加载train_set, valid_set, test_set，它们都是包括label的
#主要用到python里的gzip.open()函数,以及 cPickle.load()。
#‘rb’表示以二进制可读的方式打开文件
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    print train_set

#将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中
#GPU里数据类型只能是float。而data_y是类别，所以最后又转换为int返回
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

"""
实现LeNet5
LeNet5有两个卷积层，第一个卷积层有20个卷积核，第二个卷积层有50个卷积核
"""
def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='/Users/yigenliang/PycharmProjects/theano/assets/mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """
 learning_rate:学习速率，随机梯度前的系数。
 n_epochs训练步数，每一步都会遍历所有batch，即所有样本
 batch_size,这里设置为500，即每遍历完500个样本，才计算梯度并更新参数
 nkerns=[20, 50],每一个LeNetConvPoolLayer卷积核的个数，第一个LeNetConvPoolLayer有
 20个卷积核，第二个有50个
    """

    rng = numpy.random.RandomState(23455)

    #加载数据
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 计算batch的个数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    #定义几个变量，index表示batch下标，x表示输入的训练数据，y对应其标签
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'


#我们加载进来的batch大小的数据是(batch_size, 28 * 28)，但是LeNetConvPoolLayer的输入是四维的，所以要reshape
    layer0_input = x.reshape((batch_size, 1, 28, 28))

# layer0即第一个LeNetConvPoolLayer层
#输入的单张图片(28,28)，经过conv得到(28-5+1 , 28-5+1) = (24, 24)，
#经过maxpooling得到(24/2, 24/2) = (12, 12)
#因为每个batch有batch_size张图，第一个LeNetConvPoolLayer层有nkerns[0]个卷积核，
#故layer0输出为(batch_size, nkerns[0], 12, 12)
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )


#layer1即第二个LeNetConvPoolLayer层
#输入是layer0的输出，每张特征图为(12,12),经过conv得到(12-5+1, 12-5+1) = (8, 8),
#经过maxpooling得到(8/2, 8/2) = (4, 4)
#因为每个batch有batch_size张图（特征图），第二个LeNetConvPoolLayer层有nkerns[1]个卷积核
#，故layer1输出为(batch_size, nkerns[1], 4, 4)
    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),#输入nkerns[0]张特征图，即layer0输出nkerns[0]张特征图
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )


#前面定义好了两个LeNetConvPoolLayer（layer0和layer1），layer1后面接layer2，这是一个全连接层，相当于MLP里面的隐含层
#故可以用MLP中定义的HiddenLayer来初始化layer2，layer2的输入是二维的(batch_size, num_pixels) ，
#故要将上层中同一张图经不同卷积核卷积出来的特征图合并为一维向量，
#也就是将layer1的输出(batch_size, nkerns[1], 4, 4)flatten为(batch_size, nkerns[1]*4*4)=(500，800),作为layer2的输入。
#(500，800)表示有500个样本，每一行代表一个样本。layer2的输出大小是(batch_size,n_out)=(500,500)
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

#最后一层layer3是分类层，用的是逻辑回归中定义的LogisticRegression，
#layer3的输入是layer2的输出(500,500)，layer3的输出就是(batch_size,n_out)=(500,10)
    layer3 = SoftMax(input=layer2.output, n_in=500, n_out=10)

#代价函数NLL
    cost = layer3.negative_log_likelihood(y)

# test_model计算测试误差，x、y根据给定的index具体化，然后调用layer3，
#layer3又会逐层地调用layer2、layer1、layer0，故test_model其实就是整个CNN结构，
#test_model的输入是x、y，输出是layer3.errors(y)的输出，即误差。
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x:test_set_x[index * batch_size: (index + 1) * batch_size],
            y:test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
#validate_model，验证模型，分析同上。
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

#下面是train_model，涉及到优化算法即SGD，需要计算梯度、更新参数
    #参数集
    params = layer3.params + layer2.params + layer1.params + layer0.params

    #对各个参数的梯度
    grads = T.grad(cost, params)

#因为参数太多，在updates规则里面一个一个具体地写出来是很麻烦的，所以下面用了一个for..in..,自动生成规则对(param_i, param_i - learning_rate * grad_i)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

#train_model，代码分析同test_model。train_model里比test_model、validation_model多出updates规则
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    ###############
    #   开始训练  #
    ###############
    print '... training'
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience / 2)
 #这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。

    best_validation_loss = numpy.inf   #最好的验证集上的loss，最好即最小
    best_iter = 0                      #最好的迭代次数，以batch为单位。比如best_iter=10000，说明在训练完第10000个batch时，达到best_validation_loss
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

#下面就是训练过程了，while循环控制的时步数epoch，一个epoch会遍历所有的batch，即所有的图片。
#for循环是遍历一个个batch，一次一个batch地训练。for循环体里会用train_model(minibatch_index)去训练模型，
#train_model里面的updatas会更新各个参数。
#for循环里面会累加训练过的batch数iter，当iter是validation_frequency倍数时则会在验证集上测试，
#如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，
#则更新best_validation_loss和best_iter，同时在testset上测试。
#如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。
#当达到最大步数n_epoch时，或者patience<iter时，结束训练
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
#cost_ij 没什么用，后面都没有用到,只是为了调用train_model，而train_model有返回值
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))


                if this_validation_loss < best_validation_loss:


                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)


                    best_validation_loss = this_validation_loss
                    best_iter = iter


                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
if __name__ == '__main__':
    #evaluate_lenet5()
    load_data('/Users/yigenliang/PycharmProjects/theano/assets/mnist.pkl.gz')