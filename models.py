"""
Module to build a simple neural network.
"""

class Neural_net:
    
    """
    To build a simple neural network by taking number of layers, number of each nodes and activation
    function as input
    """
    
    def __init__(self, layer_no, node_on_each, activation_list):
        
        self.is_single_layered = True
        self.n_layers = layer_no
        
        if self.n_layers != 1:
            self.is_single_layered = False
            
        self.n_node = node_on_each
        self.activation = activation_list
        

    def initialize_weights(self, method = None):
        input_shape = self.X.shape[1]
        n_nodes = len(self.n_node)
        
        self.weights = np.array([1]).reshape(input_shape, n_nodes)

    
    def fit(self, X, y):
        self.X = X
        self.y = y
        