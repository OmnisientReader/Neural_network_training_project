 
class Validation:

    def __init__(self, network, image, quantity_to_valid, splitter):
        
        self.__quantity_to_valid = quantity_to_valid
        self.__network = network
        self.__images_data = image
        self.__images_data.set_position(self.__images_data.get_number_of_training_photos(), )
        self.__images_data.set_filetype('data')
        self.print_result(self.start_validation_process(self.__images_data, self.__network, splitter))
    
    def start_validation_process(self, image, network, splitter):

        correctly_validated_quantity = 0

        while self.__images_data.get_position('max') != self.__images_data.get_number_of_training_photos() + self.__quantity_to_valid:


            self.__images_data.set_img_data()
            self.__images_data.set_features(0)

            output = network.get_network_output(network.get_weights(), network.get_biases(), image.get_features(), 0, [image.get_features()])[-1][-1]
            
            print(output)

            if output <= splitter and image.get_filetype() == 'data':
                correctly_validated_quantity += 1

            elif output >= splitter and image.get_filetype() == 'mask':
                correctly_validated_quantity += 1
                
            
            self.__images_data.set_position_increment()

            if (self.__images_data.get_is_face()):

                self.__images_data.set_is_face(0)
                self.__images_data.set_filetype('mask')
                
            else:

                self.__images_data.set_is_face(1)
                self.__images_data.set_filetype('data')


        return correctly_validated_quantity/self.__quantity_to_valid*50
        
    def print_result(self, percentage):

        print(f"network corresponds {percentage} of images correctly")
