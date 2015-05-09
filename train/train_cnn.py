#-*-coding:utf-8-*-
import cPickle
import os
import sys
import time
from ConvPoolLayer import ConvPoolLayer
from HiddenLayer import HiddenLayer
from SoftMax import SoftMax
from PIL import Image
import numpy
import theano
import theano.tensor as T

"""
加载数据集load_data()
"""
def load_data(dataset):
    if(os.path.isdir(dataset)):
        label_paths=os.listdir(dataset)
        label=0
        data_set=[]
        label_set=[]
        for label_path in label_paths:
            images=os.listdir(dataset+"/"+label_path)
            num=0
            for image in images:
                if(num<1):
                    path=dataset+"/"+label_path+"/"+image
                    img=Image.open(open(path)).convert('L')
                    new_img=img.resize((256,256),Image.ANTIALIAS)
                    new_img=numpy.asarray(new_img,dtype=theano.config.floatX)
                    data_set.append(new_img)
                    label_set.append(label)
                num=num+1#控制一下每个类别20张图片
            label=label+1
    else:
        print 'error data path'
        exit(0)
    print '... loading data'
    data_set = (numpy.asarray(data_set,dtype=theano.config.floatX),numpy.asarray(label_set))

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


    set_x, set_y = shared_dataset(data_set)

    return (set_x, set_y)
"""
实现CNN整体架构
有5个卷积层＋pooling层,3个全连接层
卷积层学习速率0.001,前两个全连接层0.002,最后一层0.01,并且每迭代20次学习速率变为原来的1/10.
理由：前几层convolution层从原始数据中提取特征，需要让模型从预训练数据集到目标数据集上保持连续性，所以应该小些。而全连接层要快速适应目标数据集，应快些
各层patch尺寸处理成256*256,224*224,192*192,160*160,128*128，最后的patch不能太小，否则可能更容易导致图片失真
"""
def train_cnn(learning_rate=[0.001,0.002,0.01], n_epochs=200,
                    nkerns=[20, 50, 50, 50, 50], batch_size=500):
    """
 learning_rate:学习速率，随机梯度前的系数。
 n_epochs训练步数，每一步都会遍历所有batch，即所有样本
 batch_size,这里设置为500，即每遍历完500个样本，才计算梯度并更新参数
 nkerns=[20, 50, 50, 50, 50],每一个LeNetConvPoolLayer卷积核的个数，第一个LeNetConvPoolLayer有
 20个卷积核，第二个有50个
    """
    rng = numpy.random.RandomState(23455)

    patch_size=[256,224,192,160,128]
    filter_size=33

    #加载数据
    train_dataset=os.path.dirname(__file__)+"/../assets/ILSVRC2012_img_train_t3"
    train_datasets = load_data(train_dataset)
    test_dataset=os.path.dirname(__file__)+"/../assets/test-indoor-data"
    test_datasets = load_data(test_dataset)
    valid_dataset=os.path.dirname(__file__)+"/../assets/test-indoor-data"
    valid_datasets = load_data(valid_dataset)
    train_set_x, train_set_y = train_datasets
    valid_set_x, valid_set_y = valid_datasets
    test_set_x, test_set_y = test_datasets

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


#我们加载进来的batch大小的数据是(batch_size, 256 * 256)，但是LeNetConvPoolLayer的输入是四维的，所以要reshape
    layer0_input = x.reshape((batch_size, 1, patch_size[0], patch_size[0]))

# layer0即第一个LeNetConvPoolLayer层
#输入的单张图片(256,256)，经过conv得到(256-33+1 , 256-33+1) = (224, 224)，
#经过maxpooling得到(224/1, 224/1) = (224, 224)
#因为每个batch有batch_size张图，第一个LeNetConvPoolLayer层有nkerns[0]个卷积核，
#故layer0输出为(batch_size, nkerns[0], 224, 224)
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, patch_size[0], patch_size[0]),
        filter_shape=(nkerns[0], 1, filter_size, filter_size),
        poolsize=(1, 1)#不迟化
    )

#layer1即第二个LeNetConvPoolLayer层
#输入是layer0的输出，每张特征图为(224,224),经过conv得到(224-33+1, 224-33+1) = (192, 192),
#经过maxpooling得到(192/1, 192/1) = (192, 192)
#因为每个batch有batch_size张图（特征图），第二个LeNetConvPoolLayer层有nkerns[1]个卷积核
#，故layer1输出为(batch_size, nkerns[1], 192, 192)
    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], patch_size[1], patch_size[1]),#输入nkerns[0]张特征图，即layer0输出nkerns[0]张特征图
        filter_shape=(nkerns[1], nkerns[0], filter_size, filter_size),
        poolsize=(1, 1)
    )

#layer2即第三个LeNetConvPoolLayer层
#输入是layer1的输出，每张特征图为(192,192),经过conv得到(192-33+1, 192-33+1) = (160, 160),
#经过maxpooling得到(160/1, 160/1) = (160, 160)
#因为每个batch有batch_size张图（特征图），第三个LeNetConvPoolLayer层有nkerns[2]个卷积核
#，故layer2输出为(batch_size, nkerns[2], 160, 160)
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], patch_size[2], patch_size[2]),#输入nkerns[1]张特征图，即layer2输出nkerns[1]张特征图
        filter_shape=(nkerns[2], nkerns[1], filter_size, filter_size),
        poolsize=(1, 1)
    )

#layer3即第四个LeNetConvPoolLayer层
#输入是layer2的输出，每张特征图为(160,160),经过conv得到(160-33+1, 160-33+1) = (128, 128),
#经过maxpooling得到(128/1, 128/1) = (128, 128)
#因为每个batch有batch_size张图（特征图），第四个LeNetConvPoolLayer层有nkerns[3]个卷积核
#，故layer3输出为(batch_size, nkerns[3], 128, 128)
    layer3 = ConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], patch_size[3], patch_size[3]),#输入nkerns[2]张特征图，即layer3输出nkerns[2]张特征图
        filter_shape=(nkerns[3], nkerns[2], filter_size, filter_size),
        poolsize=(1, 1)
    )


#前面定义好了几个LeNetConvPoolLayer，layer3后面接layer4，这是一个全连接层，相当于MLP里面的隐含层
#故可以用MLP中定义的HiddenLayer来初始化layer4，layer4的输入是二维的(batch_size, num_pixels) ，
#故要将上层中同一张图经不同卷积核卷积出来的特征图合并为一维向量，
#也就是将layer3的输出(batch_size, nkerns[3], 128, 128)flatten为(batch_size, nkerns[1]*128*128),作为layer4的输入。
#(500，800)表示有500个样本，每一行代表一个样本。layer2的输出大小是(batch_size,n_out)=(500,500)
    layer4_input = layer3.output.flatten(2)
    layer4 = HiddenLayer(
        rng,
        input=layer4_input,
        n_in=nkerns[3] * filter_size[4] * filter_size[4],
        n_out=500,
        activation=T.tanh
    )

#第二个全连接层
    layer5 = HiddenLayer(
        rng,
        input=layer4.output,
        n_in=500,#与上一层输出保持一致
        n_out=500,
        activation=T.tanh
    )

#最后一层layer6是分类层，用的是逻辑回归中定义的LogisticRegression，
#layer6的输入是layer5的输出(500,500)，layer6的输出就是(batch_size,n_out)=(500,10)
    layer6 = SoftMax(input=layer5.output, n_in=500, n_out=10)

#代价函数NLL
    cost = layer6.negative_log_likelihood(y)

# test_model计算测试误差，x、y根据给定的index具体化，然后调用layer3，
#layer3又会逐层地调用layer2、layer1、layer0，故test_model其实就是整个CNN结构，
#test_model的输入是x、y，输出是layer3.errors(y)的输出，即误差。
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x:test_set_x[index * batch_size: (index + 1) * batch_size],
            y:test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
#validate_model，验证模型，分析同上。
    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

#下面是train_model，涉及到优化算法即SGD，需要计算梯度、更新参数
    #参数集
    conv_params = layer3.params + layer2.params + layer1.params + layer0.params
    #对各个参数的梯度
    conv_grads = T.grad(cost, conv_params)
    #参数集
    full_params = layer5.params + layer4.params
    #对各个参数的梯度
    full_grads = T.grad(cost, full_params)
    #参数集
    output_params = layer6.params
    #对各个参数的梯度
    output_grads = T.grad(cost, output_params)

#因为参数太多，在updates规则里面一个一个具体地写出来是很麻烦的，所以下面用了一个for..in..,自动生成规则对(param_i, param_i - learning_rate * grad_i)
    updates = []
    for param_i, grad_i in zip(conv_params, conv_grads):
        (param_i, param_i - learning_rate[0] * grad_i)
    for param_i, grad_i in zip(full_params, full_grads):
        updates.append((param_i, param_i - learning_rate[1] * grad_i))
    for param_i, grad_i in zip(output_params, output_grads):
        updates.append((param_i, param_i - learning_rate[2] * grad_i))

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
    patience = 5000 #look as this many examples regardless
    patience_increase = 2 #wait this much longer when a new best is found
    improvement_threshold = 0.995 #a relative improvement of this much isconsidered significant

    validation_frequency = min(n_train_batches, patience / 2) #这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。

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
            if iter % 20 == 0:
                for i in xrange(learning_rate.length):#每20epoch之后学习速率变为原来的0.1倍
                    learning_rate[i]=learning_rate[i]*0.1
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


                    if this_validation_loss < best_validation_loss * improvement_threshold:
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
    train_cnn()
    #load_data(os.path.dirname(__file__)+"/../assets/ILSVRC2012_img_train_t3")