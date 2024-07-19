import numpy as np
import copy

# this class is responsible for every training operation of network
# it gets network outputs, calculates errros and averages it for minibatch
# in the end it changes weights and biasses according to errors

class Training:


    def __init__(self, network, images_data):
        
        self.__network = network
        self.__images_data = images_data


    def start_network_training(self):

        self.__images_data.set_position(0)

        for batch in range(int(self.__images_data.get_number_of_training_photos()/self.__images_data.get_minibatch_size())*2):
            
            print(f'{batch} batch in processing...')

            if (self.__images_data.get_is_face()):
                self.__images_data.set_filetype('data')
                
            else:
                self.__images_data.set_filetype('mask')

            self.__images_data.set_img_data()

            parameters = self.update_parameters()

            self.__network.set_weights(parameters[0])

            self.__network.set_biases(parameters[1])

            self.__images_data.set_is_face(not(self.__images_data.get_is_face()))

            print(f'{batch} batch done')

    def update_parameters(self):
        

        weights = self.mul_function(copy.deepcopy(self.__network.get_weights()), 0)
        biases = self.mul_function(copy.deepcopy(self.__network.get_biases()), 0)

        for image in range(self.__images_data.get_minibatch_size()):
            
            
            self.__images_data.set_features(image)
            
            output = self.__network.get_network_output(self.__network.get_weights(), self.__network.get_biases(), self.__images_data.get_features(), 0, [self.__images_data.get_features()])      
            print(output[-1][-1])
            errors = self.backprop(weights,output, self.output_layer_error(output[-1][-1], self.__images_data.get_is_face()), len(self.__network.get_network_sample())-2)
            errors = self.error_change(errors,len(self.__network.get_network_sample()))

            minibatch_param = self.grad_param(self.__network.get_weights(), errors, output, weights, biases)

            biases = minibatch_param[1]
            weights = minibatch_param[0]

            self.__images_data.set_position_increment() 

        parameters = self.grad_descent(self.__network.get_weights(), self.__network.get_biases(), weights, biases, self.__network.get_learning_rate())

        return parameters[0], parameters[1]


    def output_layer_error(self, a, is_face):
        return [a*(1-a)*self.cost_derivative(is_face,a)]
    
    def cost_derivative(self, is_face, a):
    #print(IfFace)
        if is_face:
            return -1/a
        else:
            return 1/(1-a)
        
    def backprop(self, weights, a_func, next_error, number_of_layers):

        error_array = []
        error_layer = []
        error_array.append(next_error)
        length = number_of_layers
        for layer in range(length-1,-1,-1):

            for error in range(len(weights[layer])):

                a = a_func[layer+1][int(error)]
                error_sum = 0

                for k in range(len(weights[layer+1])):

                    error_sum = error_sum + weights[layer+1][k][error]*next_error[k]*(a*(1-a))

                error_layer.append(error_sum)
            
            next_error = error_layer
            error_array.insert(0,error_layer)
            error_layer = []

        return error_array
    
    def error_change(self,error_init,number_of_layers):

        length = number_of_layers
        mean_error = [0.0]*number_of_layers
        for layer in range(length-2,-1,-1):
            for error in range(len(error_init[layer])):
                mean_error[layer] = sum(error_init[layer])/len(error_init[layer])
        
        
        for layer in range(length-3,-1,-1):
            for error in range(len(error_init[layer])):
                current_er = error_init[layer][error]
                error_init[layer][error] = current_er*(10**self.dvs(current_er,mean_error[layer+1]))
        return error_init
    
    def dvs(self,a, b):

        # Преобразуем числа в научный формат
        a_str = f'{a:e}'
        b_str = f'{b:e}'

        # Извлекаем показатели степени
        a_exp = int(a_str.split('e')[1])
        b_exp = int(b_str.split('e')[1])

        # Вычисляем разницу в показателях степени
        dv = abs(a_exp - b_exp)

        # Если числа отличаются только по мантиссе, то разряд_разница = 0
        if a_exp == b_exp:
            # Сравниваем мантиссы, считая количество нулей в начале
            a_mant = a_str.split('e')[0]
            b_mant = b_str.split('e')[0]
            dv = abs(len(a_mant.split('.')[1]) - len(b_mant.split('.')[1]))

        return dv
    

    def mul_function(self, init_array, multiplicator):

        zeros_ex = []
        current = init_array

        for i in range(len(current)):
            if type(current[i]) == float or type(current[i]) == np.float64:
                current[i] *= multiplicator
                zeros_ex.append(current[i])
                continue
            else:
                zeros_ex.append(self.mul_function(current[i],multiplicator))
        return zeros_ex
    
    def grad_param(self, weights, errors, a, minibatch_mean_weights_parameters, minibatch_mean_biases_parameters):

        number_of_layers = len(weights)
        for layer in range(number_of_layers):
            for neuron in range(len(weights[layer])):
                for param in range(len(weights[layer][neuron])):
                    minibatch_mean_weights_parameters[layer][neuron][param]+=errors[layer][neuron]*a[layer][param]
                    minibatch_mean_biases_parameters[layer][neuron][param]+=errors[layer][neuron]
                    #if layer==1 and neuron ==0 and param==0:
                    #print("g_p ",errors[layer][neuron])
        return [minibatch_mean_weights_parameters, minibatch_mean_biases_parameters]
    

    def grad_descent(self, weights, biases, minibatch_mean_weights_parameters, minibatch_mean_biases_parameters, learning_rate):
        layers = len(weights)
        l_r = 0
        for layer in range(layers):
            
            if layer == 1:
                l_r = 0.05
            else:
                l_r = learning_rate
            for neuron in range(len(weights[layer])):
                for weight in range(len(weights[layer][neuron])):
                    
                    weights[layer][neuron][weight] -= (minibatch_mean_weights_parameters[layer][neuron][weight])*l_r  #*shift
                    biases[layer][neuron][weight] -= (minibatch_mean_biases_parameters[layer][neuron][weight])*l_r    #*shift
                    
        return [weights,biases]
    