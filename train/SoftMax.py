#-*-coding:utf-8-*-

import numpy
import theano
import theano.tensor as T

class SoftMax(object):
    def __init__(self,input,n_in,n_out):
        self.W=theano.shared(
            value=numpy.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b=theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x=T.nnet.softmax(T.dot(input,self.W)+self.b)#我们需要的概率密度
        self.y_pred=T.argmax(self.p_y_given_x,axis=1)
        self.params=[self.W,self.b]
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()