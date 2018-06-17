from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences , skipgrams
from nltk import word_tokenize
import numpy as np

class cosineSimilarity:

    def __init__(self , target , context ):
        
        self.__tokenize = Tokenizer()

        target = self.__get_words( target )

        context = self.__get_words( context )
       
        self.__v1 = self.__convert_word_vector( target ) 
        
        self.__v2 = self.__convert_word_vector( context )
        
        if len( self.__v1 ) > len( self.__v2 ): 
            
            self.__v2 = self.__same_size( len( self.__v1 ) , self.__v2 )
        
        elif len( self.__v1 ) < len( self.__v2 ): 
            
            self.__v1 = self.__same_size( len( self.__v2 ) , self.__v1 )
    
        self._dot_product()

        self._abs_vectors()


    def _dot_product(self):
       
        self._dot_pro = np.vdot(self.__v1 , self.__v2)
        

    def _abs_vectors(self):
        
        self._abs_vec = np.linalg.norm(self.__v1) * np.linalg.norm(self.__v2)


    def get_result(self):

       return ( ( self._dot_pro / self._abs_vec ) * 100) 

    def __convert_word_vector( self , words , max_length=30):

        """
            this function only for get the words as a list and run the skipgrams and convert words
            to vector with specifiy the length of the vector.

            @param: list of words.
            @param: integer for length.

            @return: numpy of ndarray.
            
        """
        
        if len(words) > 1:

            data , _ = skipgrams( words , max_length )

            self.__tokenize.fit_on_texts(data)

            encod_words =  self.__tokenize.texts_to_sequences(data)

            encod_words = pad_sequences( encod_words , maxlen=max_length , padding="post")
            
            return encod_words

        elif len(words) == 1:
            
            self.__tokenize.fit_on_texts(words)

            encod_words =  self.__tokenize.texts_to_sequences(words)

            encod_words = pad_sequences( encod_words , maxlen=max_length , padding="post")
            
            return encod_words

    def __same_size( self , size , vector ):
        
        """
            this function are used for add rows to get the same matrix to other matrix.
            to continue the proccessing between two vectors.

            @param: size of the larger size vector.
            @param: numpy of ndarray of smaller vector

            @return: numpy ndarray.

        """

        d = np.ndarray( (size - len( vector ) , 30) ,dtype=np.int16)

        for i in range( size - len( vector ) ):

            for k in range( 30 ):
            
                d[i] = 0
        

        return np.concatenate( (vector , d)  , axis=0) 

    def __get_words( self , text ):

        """
            it's deal with the data to extract all words from the text into a list

            @param: string of data

            @return: list of words.

        """

        return word_tokenize( text )    

