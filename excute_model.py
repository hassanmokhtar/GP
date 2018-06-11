# import embedding_model as model
import embedding_model as model
import perpare_dataset as dataset
import proccess_wordVector as wv
import numpy as np

class excute_model:

    def __init__( self , path , filenames ):
       
       self.__dataset = dataset.process_data( path , filenames )

       self.__wv = wv.process_wordVector()

    def process_dataset( self , sentence=True):
        
        """
            it's call run function that built into class prepare dataset to process the data 
            and return from it the occurrence word from each word as a list and the documents data
            as a list

            @return: list of occurence word , list of document data

        """

        return  self.__dataset.run_process(sentence)    

    def process_word_vector( self , docs , vocab_size=0 , labels={} , label=True ):

        self.__wv.fit_text(docs)
        
        if vocab_size == 0:
            vocab_size = self.__wv.vocab_size()

        encoded_doc = self.__wv.text_matrix(docs)
        
        encoded_doc = self.__wv.padding( encoded_doc , vocab_size ) #max([len(s.split()) for s in data ]) )
        
        #data , labels = self.__wv.skipgram_model( encoded_doc , vocab_size )

        if label:

            labels = np.array( [i for i , k in enumerate( labels ) for j in range( labels[ k ] ) ] )

            return np.array( encoded_doc , dtype=np.int32 ), labels , vocab_size #data , labels , vocab_size

        return np.array( encoded_doc , dtype=np.int32 )  


    def run_model( self , X_train , y_train , X_test , y_test,  vocab_siz , vector_dim , input_length ):

        self.__model = model.Embedding_Model(vocab_siz , vector_dim , input_length )    

        self.__model.build_model()

        self.__model.compile_model(['accuracy'])

        self.__model.fit( X_train , y_train , 10 )


if __name__ == "__main__":

    exc = excute_model('./dataset/' , ['skills.txt' , 'experience.txt' , 'education.txt'])

    occ_data , docs , len_data = exc.process_dataset(False)
    
    X_train , y_train , vocab_siz =  exc.process_word_vector( docs , labels=len_data )

    exc_test = excute_model('./dataset/' , ['test.txt'])

    occ_d , doc , _ = exc_test.process_dataset(False)

    X_test  = exc_test.process_word_vector(doc , vocab_size=vocab_siz , label=False )

   # print(X_train)
    exc.run_model(X_train , y_train , X_test ,y_train, vocab_siz , 128 , vocab_siz )
