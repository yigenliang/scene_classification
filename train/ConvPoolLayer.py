#-*-coding:utf-8-*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class ConvPoolLayer(object):
    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
        assert filter_shape[1] == image_shape[1]
        self.input=input
        fan_in=numpy.prod(filter_shape[1:])
        fan_out=(filter_shape[0]*numpy.prod(filter_shape[2:]))/numpy.prod(poolsize)
        W_bound=numpy.sqrt(6./(fan_in+fan_out))
        self.W=theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values=numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=b_values,borrow=True)
        conv_out=conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        pooled_out=downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output=T.tanh(pooled_out+self.b.dimshuffle('x',0,'x','x'))
        self.params=[self.W,self.b]
