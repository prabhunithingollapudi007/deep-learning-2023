import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from copy import deepcopy
from Optimization import Optimizers

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # since it is a trainable layer
        self.trainable = True 
        # Initializing it as False
        self.memorize_flag = False
        # Input dimension
        self.size_input = input_size
        # Hidden state dimension & its initialization
        self.size_hidden = hidden_size
        self.hidden_state = np.zeros(self.size_hidden)
        # Output tensor dimension
        self.size_output = output_size
        #initializing few stuffs
        self.optimizer = None
        self._wghts = None

        # Elman RNN Cell
        self.elman_layers = [
            FullyConnected(input_size = self.size_input + self.size_hidden, 
                           output_size = self.size_hidden),
            TanH(),
            FullyConnected(input_size= self.size_hidden, output_size=output_size),
            Sigmoid()# ==> It will give output at timestamp t
        ]
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        self.input_tensor = input_tensor
        output_tensor = np.zeros([batch_size, self.size_output])
        self.all_states = []
        # self.hidden_state = np.zeros(self.size_hidden)

        if not self.memorize_flag:
            self.hidden_state = np.zeros(self.size_hidden)

        # Batch dimensions as time dim; t = time
        for t in range(batch_size):
            states_t = []
            inp_vector_t = input_tensor[t]

            hidden_tensor = np.concatenate([inp_vector_t, self.hidden_state])
            hidden_tensor = np.expand_dims(hidden_tensor, axis = 0)
            
            # Implementing ELMAN RNN Cell
            out_FC_1 = self.elman_layers[0].forward(hidden_tensor) # FC
            self.hidden_state = self.elman_layers[1].forward(out_FC_1) #tanh
            out_FC_2 = self.elman_layers[2].forward(self.hidden_state) # FC 2
            output_vector_t = self.elman_layers[3].forward(out_FC_2) # sigmoid

            # Saving all hidden states
            states_t.extend([self.elman_layers[0].input_tensor,
                             self.elman_layers[1].f_x,
                             self.elman_layers[2].input_tensor,
                             self.elman_layers[3].f_x
                             ])
            self.all_states.append(states_t)

            self.hidden_state = self.hidden_state.flatten()
            output_tensor[t] = output_vector_t
        
        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        self.error_tensor = error_tensor
        # Initializing tensors
        output_tensor = np.zeros([batch_size, self.size_input])
        hidden_err = 0
        #gradient wghts of FC layers
        FC2_wghts = np.zeros_like(self.elman_layers[2].weights)
        FC1_wghts = np.zeros_like(self.elman_layers[0].weights)

        for t in reversed(range(batch_size)):
            # Elman RNN cells outputs from forward pass
            self.elman_layers[3].f_x = self.all_states[t][3]
            self.elman_layers[2].input_tensor = self.all_states[t][2]
            self.elman_layers[1].f_x = self.all_states[t][1]
            self.elman_layers[0].input_tensor = self.all_states[t][0]

            # Updating error while backpropogating
            err = error_tensor[t]
            err = self.elman_layers[3].backward(err)
            err = self.elman_layers[2].backward(err)
            err = err + hidden_err # adding the internal hidden state error
            err = self.elman_layers[1].backward(err)
            err = self.elman_layers[0].backward(err)
            hidden_err = err[:,self.size_input:]

            FC1_wghts += self.elman_layers[0].gradient_weights
            FC2_wghts += self.elman_layers[2].gradient_weights
            output_tensor[t] = err[0, :self.size_input]
        
        self.gradient_weights = FC1_wghts

        if self.optimizer is not None:
            self.elman_layers[0].weights = self.optimizer.calculate_update(self.elman_layers[0].weights, FC1_wghts)
            self.elman_layers[2].weights = self.optimizer.calculate_update(self.elman_layers[2].weights, FC2_wghts)
        
        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.elman_layers[0].initialize(weights_initializer, bias_initializer)
        self.elman_layers[2].initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        return self.memorize_flag
    
    @memorize.setter
    def memorize(self,memorize):
        self.memorize_flag = memorize

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self.elman_layers[0].gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, wghts):
        self.elman_layers[0]._gradient_weights = wghts

    @property
    def weights(self):
        return self.elman_layers[0].weights
    
    @weights.setter
    def weights(self, wghts):
        self.elman_layers[0].weights = wghts
        

