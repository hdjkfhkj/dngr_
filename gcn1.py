from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class GraphConvolution(Layer):

    def __init__(self, input_dim,output_dim,support,act, **kwargs):
        self.output_dim = output_dim
        self.input_dim=input_dim
        self.support=support
        self.act=act
        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(GraphConvolution, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        pre= K.dot(x, self.kernel)
        out=K.dot(x,pre)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)