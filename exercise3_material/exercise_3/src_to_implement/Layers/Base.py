class BaseLayer:
    '''
    Parent class for all the connected layers
    '''
    def __init__(self):
        self.trainable = False
        self.default_weights = 0.5
        self.testing_phase = False