from keras.preprocessing.sequence import make_sampling_table , skipgrams , pad_sequences
from keras.preprocessing.text import Tokenizer , one_hot , text_to_word_sequence
from keras.layers import Flatten , Conv1D , MaxPool1D , Dense,Embedding
from keras.models import Sequential
import perpare_dataset
import pandas as pd
import numpy as np

class process_wordVector:

    def __init__( self ):
        
        # create object from tokenize
        self.__tokenize = Tokenizer()
    
    def text_sequence( self , text ):

        """
            it's take the text as a list and use the built function in keras that convert
            text to sequence of number.

            after the call it ,it's return the list of list number 


            @param: list of content

            @return: list of list sequence number.

        """  

        return self.__tokenize.texts_to_sequences( text )

    def text_matrix( self , text ):

        """
            it's take the text as a list and use the built function in keras that convert
            text to matrix of number.

            after the call it ,it's return the list of list number 


            @param: list of content

            @return: list of list matrix sequence number.

        """  

        return self.__tokenize.texts_to_matrix( text )

    def fit_text( self , text ):

        """
            it's take the text as a list and use the built function in keras that fit text.

            @param: list of content.

        """    
        
        self.__tokenize.fit_on_texts( text )

    def padding( self , encoded_data , max_length , padding='post' ):

        """
            it's create matrix with max length that taken as a param
            and add padding 0 to each of index that not equal to length that given.

            this padding only used for resize all index at a same length to work with it.

            the type of padding should be match as a following:
              - post: it's add 0 at end of index.
              - pre : it's add 0 at begin of index.

            @param: string of type padding.

            @return: list of encoded data after add the padding.

        """  

        return pad_sequences( encoded_data , maxlen=max_length , padding=padding)

    def vocab_size( self ):
        
        # it's return the size of vocabulary after process with tokenize
        return len( self.__tokenize.word_index ) + 1

    def skipgram_model( self , data , vocab_size , window_siz=5  ):

        """
            it's just create the skip-grame model to create the similairty and predict the similar 
            word for meaning.

            we need the dataset and the size of vocabulary and size of word the need to taken in each process

            after load skip-gram model it's return the label of each word and also return the target and context

            notice: the target is the word the need to similiar of words and the context it's
            a words that before and after the target word.

            @param: list of dataset
            @param: integer of size of vacabulary
            @param: integer of size window
            @param: integer of negative sample.

            @return: list of data , list of labels 

        """
        
        #sample_table = make_sampling_table(vocab_size)

        data , labels = skipgrams(data , vocab_size , window_size=window_siz )#, sampling_table=sample_table)

        # return the data after reformat by np and labels

        return [ np.array( x ) for x in data ] , np.array( labels , dtype=np.int32 )


