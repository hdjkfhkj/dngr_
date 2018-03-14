import numpy
import pdb

import numpy as Math


class DataGenerator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def fit(self, inputs, targets):
        self.start = 0
        self.inputs = inputs
        self.targets = targets

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < len(self.inputs):
            input_ = self.inputs
            target_ = self.targets
            output1 = target_[self.start:(self.start + self.batch_size)]
            output2 = input_[self.start:(self.start + self.batch_size)]
            self.start += self.batch_size
            return (output1, output2)
        else:
            self.reset()
            return self.next()
