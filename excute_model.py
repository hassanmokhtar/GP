import embedding_model as model
import perpare_dataset as dataset
import proccess_wordVector as wv
from sklearn.preprocessing import LabelBinarizer
import numpy as np

class excute_model:

    def __init__( self , path , filenames ):
       
       """
            it's inialize the algorithm that prepare the dataset and go through convert data to vector 
            matrix and build the model to fit and predict the test data.

            @param: string of the path.
            @param: list of string filenames.

       """

       # create object from the class prepare dataset.
       self.__dataset = dataset.process_data( path , filenames )

       # create object from the class word vector.
       self.__wv = wv.process_wordVector()
       
       # create object from the class embedding model.
       self.__model = model.Embedding_Model() 


    def process_train_data( self , sentence=True , is_train=True ):
        
        """
            it's call run function that built into class prepare dataset to process the data 
            and return from it the occurrence word from each word as a list and the documents data
            as a list

            @return: list of occurence word , list of document data

        """

        return  self.__dataset.run_process( sentence , is_train )


    def process_word_vector( self , docs , vocab_size=0 , labels={} , is_train=True ):

        """
            it's deal with the class word vector to buil the proccess of the word and convert it
            to matrix.

            this function call it by two operation:

            - if you need to get the train data then neither vocab_size nor is_train add thing.

            - if you need to get the test data then change is_train to false and add the vocab_size
            that sould be greater than 0 to test the data. 

            @param: list of words in docs
            @param: integer vocabulary size.
            @param: dictionary of labels.
            @param: boolean of label. 

        """
        
        # prepare to train the text
        self.__wv.fit_text(docs)
        
        # check if the vocab size are 0 then need get the vocab size otherwise used that given

        if vocab_size == 0: 

            vocab_size = self.__wv.vocab_size()
        
        # call function text matrix to convert the words to matrix
        encoded_doc = self.__wv.text_matrix(docs)
        
        # call function padding to get the all index of the matrix as a same size.
        encoded_doc = self.__wv.padding( encoded_doc , vocab_size ) 

        if is_train: # check if you are call function to train or test
            
            # add labels of each class.
            labels = np.array( [i for i , k in enumerate( labels ) for j in range( labels[ k ][0] ) ] )

            #return the data and the labels
            return np.array( encoded_doc , dtype=np.int32 ), labels , vocab_size #data , labels , vocab_size

        #return the data only.
        return np.array( encoded_doc , dtype=np.int32 )  


    def run_model( self , X_train , y_train , X_test , y_test,  vocab_siz , vector_dim , input_length , texts , labels , save_model='embedding_model'):
        
        """
            this function deal with the model such that build and fit the model and also deal with 
            the prediction to show the data that taken are classified in any of the classes.

            first of thing need to check if the model already exist or not.
            if the model are exist then predict the data directly.

            otherwise the model need to build and fit before the predict the data

            @param: array of the train data.
            @param: array of label data that need to train.
            @param: array of test the data the need to predict it.
            @param: integer of the size of vocabulary.
            @param: integer of dimension of the vector.
            @param: integer of then input length.
            @param: list of words.
            @param: dictionary of labels.

        """ 

        #check if the model are found or not to load the model and predict the data.
        if self.__model.check_exist_model(save_model):
            
            # if found then load the model
            
            self.__model.load_model(save_model) 

        else:
            
            # then the model need to build.
            self.__model.build_model(vocab_siz , vector_dim , input_length )

            # compile the mdoel after build the model.
            self.__model.compile_model(['accuracy'])
            
            encode =LabelBinarizer()

            y_train = encode.fit_transform(y_train)


            # and finally fit the data into the model with sepcific epoch and batch size.
            self.__model.fit( X_train , y_train , 10 , batch_size=350 )

            # save model
            self.__model.save_model(save_model)

        # predict the data and get the accurracy and the class.
        acc , label = self.__model.predict(X_test) 
        
        acc = acc.tolist()

        for i in range( len( acc ) ) :
    
            m = max(acc[i])

            if m == acc[i][0]:

                print ( "The "+ str(texts[i]) + " have %0.2f %%" %((acc[i][0]) * 100) + " that belong to class " + str(labels[ 0 ][ 1 ]) )
            
            elif m == acc[i][1]:

                print ( "The "+ str(texts[i]) + " have %0.2f %%" %((acc[i][1]) * 100) + " that belong to class " + str(labels[ 1 ][ 1 ]) )

            else:

                print ( "The "+ str(texts[i]) + " have %0.2f %%" %((acc[i][2]) * 100) + " that belong to class " + str(labels[ 2 ][ 1 ]) )


if __name__ == "__main__":

    exc = excute_model('./dataset/new_dataset/' , [ 'skills.txt' , 'experiences.txt' ,   'educations.txt'] )

    occ_data , docs , detail_data = exc.process_train_data(False)
    
    X_train , y_train , vocab_siz =  exc.process_word_vector( docs , labels=detail_data )

    exc_test = excute_model('./dataset/' , ['test.txt'])

    doc = exc_test.process_train_data(False , is_train=False)

    X_test  = exc_test.process_word_vector(doc , vocab_size=vocab_siz , is_train=False )

    exc.run_model(X_train , y_train , X_test ,y_train, vocab_siz , 128 , vocab_siz , doc , detail_data , save_model='embedding_model' )
