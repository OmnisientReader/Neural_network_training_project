import random
import numpy as np
from Images_data import ImagesData

# this class has whole network structure: weights, biases, network's layers structure.
# it's also responsible for computing neuron's weighted sum and output
# it has methods to write computed parameters into a file
# and read parameters from a file with the same format


class Network:

    def __init__(self, network_params, data_path, mask_path, learning_rate, minibatch_size, is_face, number_of_training_photos, filepaths):

        self.__weights = None
        self.__biases = None 
        self.__network_params = network_params
        self.__learning_rate = learning_rate
        temporary_data = ImagesData(data_path,mask_path, 0, minibatch_size, is_face, number_of_training_photos)
        self.__network_sample = self.create_network_sample(self.__network_params,temporary_data)
        if filepaths:
            self.set_params(filepaths)
        else:
            self.__weights, self.__biases = self.initialize(self.__network_sample)
        
        
    
    def set_params(self, paths):

        params = [self.set_weights, self.set_biases]
        
        k = 0

        for path in paths:
            params[k](self.read_3d_array_from_file(path))
            k+=1

    def initialize(self,network_sample):
        
        weights = []
        biases = []
        prev_layer = network_sample[0]

        for layer in network_sample[1:]:

            temp_w1 = []
            temp_w2 = []
            temp_b1 = []
            temp_b2 = []

            for w in layer:

                for iteration in prev_layer:

                    temp_w1.append(((random.random()) *0.00001)*random.choice([1,-1]))
                    temp_b1.append(((random.random())*0.00001)*random.choice([1,-1]))

                temp_w2.append(temp_w1)
                temp_b2.append(temp_b1)

                temp_w1 = []
                temp_b1 = []

            weights.append(temp_w2)
            biases.append(temp_b2)
            prev_layer = layer

        return weights,biases
    
    def create_network_sample(self, network_params, temporary_data):

        temporary_data.set_img_data()
        temporary_data.set_features(0)

        network_sample = ([[0.0]*i for i in network_params])
        network_sample.insert(0,temporary_data.get_features())

        return network_sample
    
    def get_network_output(self, weights, biases, initial_values, active_layer = 0, output_values = []):
        layer_output_values = []

        if active_layer == len(weights):
            return output_values

        length = len(weights[active_layer])
        layer_output_values = []

        for i in range(length):
            z = self.neuron_weighted_sum(np.array(weights[active_layer][i]),np.array(biases[active_layer][i]),np.array(initial_values))
            layer_output_values.append(self.neuron_activation_function(z))
        
        output_values.append(layer_output_values)

        return self.get_network_output(weights,biases,output_values[active_layer+1],active_layer+1,output_values)
    
    def neuron_weighted_sum(self, input_weights_vector, input_biases_vector, input_values_vector):

        dt_mlt = (input_weights_vector*input_values_vector)
        vector_sum = np.sum([input_biases_vector,dt_mlt],axis = 0)
        return np.sum(vector_sum)
    
    def neuron_activation_function(self, weighted_sum):
        
        
        sigmoid_output = 1.0/(1.0+np.exp(weighted_sum))
        
        if sigmoid_output > 0.99:
            sigmoid_output = 0.99
        
        elif sigmoid_output < 0.01:
            sigmoid_output = 0.01
        
        return(sigmoid_output)
    
    def write_3d_array_to_file(self, array, filename):
        with open(filename, 'w') as file:
            m = len(array)
            for i in range(m):
                file.write(f'-{i+1}-\n')
                n = len(array[i])
                for j in range(n):
                    values = ' '.join(map(str, array[i][j]))
                    file.write(f'{j+1}) {values}\n')


    def read_3d_array_from_file(self, filename):
        array = []
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            current_layer = -1
            for line in lines:
                line = line.strip()
                if line.startswith('-') and line.endswith('-'):
                
                    current_layer += 1
                    array.append([])
                elif ')' in line:
                
                    parts = line.split(')')
                    index = int(parts[0])
                    values = list(map(float, parts[1].strip().split()))
                    array[current_layer].append(values)
        return array


    def get_weights(self):
        return self.__weights
    
    def get_biases(self):
        return self.__biases
    
    def get_network_sample(self):
        return self.__network_sample
    
    def get_learning_rate(self):
        return self.__learning_rate
    
    def set_weights(self, weights):
        self.__weights = weights

    def set_biases(self, biases):
        self.__biases = biases
