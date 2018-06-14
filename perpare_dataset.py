from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
import random

class process_data:

    def __init__( self , path=None , filenames=None ):
        
        if path != None and filenames !=None:
            self.__path = path

            self.__filenames = filenames


    def __read_data( self , path , filename ):
        
        """
            it's open the file that give the name and the path as a paramater
            and read the content of the text and then return the content after readed.

            @param: string of path the file.
            @param: string of the name file.

            @return: string of the read content data.

        """

        with open(path + filename , "r" ) as f:

            data = f.read()
        
        return data

    def __clean_sentence_punctuation( self , data ):

        """
            it's split data and remove the punctuation and return it
        
            @param: string of data

            @return: list of sentence data

        """

        data = data.lower().split('\n')

        table = str.maketrans( "","" , punctuation )

        data = [ w.translate( table ) for w in data ]

        return data

    def __clean_word_punctuation( self , data ):
        
        """
            it's split data and remove the punctuation and return it
        
            @param: string of data

            @return: list of word data

        """

        data = data.lower().split()
        
        stop_words = set(stopwords.words('english'))

        table = str.maketrans( "","" , punctuation )

        data = [ w.translate( table ) for w in data ]

        data = [ word for word in data if word.isalnum() ]

        data = [ word for word in data if word not in stop_words ]

        return data

    def collect_data( self, path , filenames , sentence=True ):

        """
            it's just for call the hidden function that open the file and get the data from it's file.

            @param: string of path the file.
            @param: string of the name file.

            @return: string of the read content data.

        """
        
        data = {} #store the content of each file text and key is name file and content is a value.

        for filename in filenames:
            
            r_data = self.__read_data( path , filename ) 
            
            result = []

            if sentence:
                result = self.__clean_sentence_punctuation( r_data )

            else:
                result = self.__clean_word_punctuation( r_data )

            data[ filename.lower().split(".")[0] ] =  result

        return data


    def __check_mini_occurrence( self , doc1 , doc2 ):

        """
            it's check the occurrence for each word if the word in first doc are greater than from 
            other doc then it will remove the small occurrence to still unique the word into only one
            document.

            then we take only two document and check the minimum occurrence.
            the two document should be dictionary which key it's a word and the value it's a occurrence.
            
            after clean the document from duplicate into other document it will return after cleaned.

            @param: dict of first document.
            @param: dict of second document.

            @return: dict of first document clean , dict of the last document clean.

        """

        # it's store the remove words that duplicated into document
        r_doc1 = [] 
        
        r_doc2 = []
        
        # check for maxmimum length of two document to start iterate from the max
        if len(doc1) > len(doc2):
            doc_it = doc1

        else:
            doc_it = doc2


        for k in doc_it: # loop for max doc 

            if k in doc1 and k in doc2: # check if the key are in two document

                if doc1[k] > doc2[k]: 
                    r_doc2.append(k)

                elif doc2[k] > doc1[k] :
                    r_doc1.append(k)

                else:
                    # if the value of the key are equal into two document then create random to remove
                    # the key from any of two document depend on the random. 
                    n = random.randint(0 , 1)
                    if n == 0:
                        r_doc1.append(k)
                    elif n == 1:
                           r_doc2.append(k) 


        #reset the list with unique the index without duplicate and iterate in each
        #to clean the document from duplicate.

        for r in list(set(r_doc1)): 
            
            del doc1[r] 

        for r in list(set(r_doc2)):
            del doc2[r]             

        return doc1 , doc2



    def clean_repeat_word( self , docs ):

        """
            it's take many documents as a list and loop for twice documents for clean the repeat
            words and get the unique word in each document.


            @param: list of documents

            @return: list of documents.
        
        """

        docs_dic = {} #store and update each document after process with it
        
        i = 0
        
        for doc_parent in docs:
            
            p = 'doc' + str(i) #rename the document that iterate for other document after it
            
            for doc_child in docs[ i+1 : ]:
                
                c = 'doc'+ str(i+1) #rename the document that iterate with the main document
                
                doc1 , doc2 = self.__check_mini_occurrence( doc_parent , doc_child )

                #override the happen in index document

                docs_dic[p] = doc1 

                docs_dic[c] = doc2

            i +=1
        
        # get each document as a key and return it into a list
        return [ docs_dic[k] for k in docs_dic ]

            
    def calculate_occurence( self , doc ):
        
        """
            it's just take the document as a a list and calculate the occurrence in each word
            using counter it's a built in from python.

            @param: list of content document

            @return: dict of word as a key and occurrence as a value.

        """

        return Counter(doc)

    def combine_data( self , docs ):
        
        """
            it's just combine all docs as a one list to collect the data.

            first thing need the docs and get the length of each docs and store it into
            dictionary with key the name docs and value is the length.

            after that return two thing first is the list that has all docs 
            and second the dictionary.

            @param: list of data docs

            @return: list of combine data  ,  dictionary of value list have length each docs and name.

        """

        detail_doc = {}
        
        docs_list = []

        for i in range( len( docs ) ):

            docs_list += docs[i]
            
            detail_doc[ i ] = [ len( docs[i] ) , self.__filenames[i].split(".")[0] ]   

        return docs_list , detail_doc


    def run_process( self  , sentence=True , is_train=True):

        """
            it's run the class to load proccess in the data after return it after process.

            @return: list of document occurrence , list of document data.
            
        """

        docs = self.collect_data(self.__path , self.__filenames , sentence=sentence )

        docs = [ docs[k] for k in docs ] 

        #check if is train or not
        if is_train:

            cal_doc_occ = []

            for doc in docs:

                cal_doc_occ.append( self.calculate_occurence(doc) )

            cal_doc_occ = self.clean_repeat_word( cal_doc_occ )
            
            docs , detail_doc = self.combine_data( docs )

            return cal_doc_occ , docs , detail_doc


        return docs[0]    

