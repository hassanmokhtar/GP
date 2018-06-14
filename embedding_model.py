from keras.layers import Embedding  , Dense , Flatten , Dropout
from keras.layers.convolutional import Conv1D , MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



class Embedding_Model:

    def __init__( self ):
        
        """
            this class for build the model word2vector with convoluation neural network.

            first we need to install some instruction like vocabulary size and the dimension
            of the vector and the length of the input

            it's take the each of word has a vector that represented each of word to pass it
            into the model to train it into the cnn model using embedding.

            after training you can predict any word to get the similarity of this word in the meaning.
        
        """

        self.__model = Sequential()


    def build_model( self , vocab_siz , vector_dim , input_length ,
                    filters=32 , kernal_siz=8 , pool_size=4 , dense_feature_hidden=128 ,
                    dense_feature_output=3 , conv_activation='relu' , 
                    dense_activation_hidden='relu' , dense_activation_output='sigmoid' ):
        
        """
            it's prepare the model of word2vector 
            we need to add the input layer that have embedding layer as a input
            and add one layer conv has a 1 dimension only and 
            add maxpooling to resize the matrix and after that need to add flatten to use
            fully connected layer and finally need two dense to be added the first in hidden layer
            and the last for the output layer.

            first of thing need to get the filteration and the size of kernal and the activation
            that need used in:
            - conv1D
            - dense of hidden layer
            - dense of the ouput layer

            @param: integer size of vocabulary
            @param: integer dimension of vector
            @param: integer length of input
            @param: integer size of filteration
            @param: integer kernal size
            @param: integer pool size
            @param: integer feature of dense hidden layer
            @param: integer feature of dense ouput layer
            @param: string  activation function for conv1D
            @param: string  activation function for dense of hidden layer
            @param: string  activation function for dense of ouput layer

        """  
        
        # first layer is the input layer that has a embedding 
        self.__model.add( Embedding( vocab_siz , vector_dim , input_length=input_length ) )

        #second layer is the conv of 1 dimension
        self.__model.add( Conv1D( filters , kernal_siz , activation=conv_activation ) )

        # third add layer of maxpool with one dimension
        self.__model.add( MaxPooling1D( pool_size ) )

        #second layer is the conv of 1 dimension
        self.__model.add( Conv1D( filters * 2 , kernal_siz , activation=conv_activation ) )

        # third add layer of maxpool with one dimension
        self.__model.add( MaxPooling1D( pool_size * 2) )

        # add fully connected layer
        self.__model.add( Flatten() )

        # add Dense for hidden layer
        self.__model.add( Dense( dense_feature_hidden , activation=dense_activation_hidden ) )

        self.__model.add( Dropout(0.1) )

        # add Dense for output layer
        self.__model.add( Dense( dense_feature_output , activation=dense_activation_output ) )



    def compile_model( self , metrices , loss='binary_crossentropy' , optimizer='adam'  ):
        
        """
            after build the model need to compile it to prepare for training the data
            
            then we need the loss function and the optimizer and what we need of the ouput

            @param: list of ouput that needed from model
            @param: string  loss function
            @param: string  optimizer function

        """

        self.__model.compile( optimizer=optimizer , loss=loss , metrics= metrices  )


    def fit( self , X_train , y_train , epochs , batch_size=128 ):
        
        """
            after build yhe model and compile need to train the data then
            need the train data and labels and the number of epochs

            @param: list for x train data
            @param: list for labels of train data
            @param: integer epochs number

        """

        self.__model.fit( X_train , y_train , epochs=epochs , batch_size=batch_size ,  verbose=1 , validation_split=0.2)
        


    def evaluate( self , X_test , y_test ):
        
        """
            it's just take the test data to evaluate the model after training
            to check of the accuracy

            @param: list for test data
            @param: list for label test data

            @return: double of loss , double of accuracy

        """
        
        #return the loss and accuracy after evalute with given test data
        return self.__model.evaluate( X_test , y_test ) 

        

    def predict( self , test_data ):
        """
            it's take a test data and predict it with the model and return the probability
            or the accuracy for each of word in the test data

            @param: list integer test data

            @return: list of double , list of integer of classes.
        """
        return self.__model.predict(test_data) , self.__model.predict_classes(test_data)

    def save_model( self , name_model ):

        """
            it's just save the model after trained to use the model in the future.

        """

        self.__model.save( name_model + '.h5')


    def load_model( self , name_model ):

        """
            it's just load the model after saved to continue the proccess such that the predict
            data and so on.

        """ 

        self.__model = load_model( name_model + '.h5')   

    def check_exist_model( self , name_model ):

        """
            it's just check if the file h5 already found or not and return boolean

            @return: Boolean
        """

        from pathlib import Path

        model_exist = Path('./'+ name_model +'.h5')

        if model_exist.is_file():

            return True

        return False    

    
    
