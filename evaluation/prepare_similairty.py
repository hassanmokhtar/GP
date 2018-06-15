from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import word_tokenize

class prepare_similarity:

    def __init__( self ):
         
        """
            it's load and prepare the data and inialize the algorithm to start similar with data

        """
        self.__load_dataset( "./glove-300.bin" ) # load glove dataset


       


    def __load_dataset( self , filename ):
        
        """
            it's load dataset into gensim model to start proccess simialrity with it

            @param: string

        """
        
        self.__wv = KeyedVectors.load_word2vec_format( filename , binary=True)



    def preproccess_data( self , text ):

        """

            it's just for clean text from stopwords and split data with space and return it after clean

            @param: string

            @return: string

        """

        text = text.strip().lower().split()

        stop_words = set(stopwords.words("english"))

        text = filter(lambda word : word not in stop_words , text )

        return " ".join( text )

    def get_words( self , text ):

        """
            it's deal with the data to extract all words from the text into a list

            @param: string of data

            @return: list of words.

        """

        return word_tokenize( text )

    def similarity( self , target , data ):

        """
            it's used the glove dataset for similar the target with the data

            and return the accuracy of the target into the data.

            @param: string of target data
            @param: string of data

            @return: float of accuracy 0.2f * 100 % , e.g ( 50,01 % )

        """
        
        acc = self.__wv.similarity( target , data )

        return "%0.2f " %( acc * 100 ) + " %"


if __name__ == "__main__":

    pre = prepare_similarity()
    
    target = pre.get_words( "1 year of experience" )
    
    context = pre.get_words( "2 year of experience" )
    
    acc= []
    
    for i in range( len( target ) ):

        acc.append( pre.similarity( target[i] , context[i] ) )
      

    print( max( [ len(i.split) for i in acc ] ) )   