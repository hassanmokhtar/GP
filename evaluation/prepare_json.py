import json 
from pprint import pprint

class prepare_json:

    def __init__(self):
        
        """
            it's initialize some property like id_cv , cv key , criteria key 

        """
        pass

    def __read_data(self , filename):
        
        """
            it's open and read the content of the file and return the content

            @param: string of path name
            @param: string of filename

            @return: string of content

        """

        with open( filename , "r") as f:

            data = f.read()

        return data

    def __load_json( self , data ):

        """
            it's get the data after read from the file and load it into json to start proccess with it.

            @param: string of content

            @return: object json

        """

        return json.loads(data)


    def __set_cv_data( self ):

        """
            it's just set the key of the cv that have all content/section of the cv from json object

            @param: object json 

            
        """
        
        self.__cv_data = self.__data_json['cv']

    def __set_criteria_data( self ):

        """
            it's just set the key of the criteria that have  all requirement that need to match in the cv

            @param: object json 

            @return: object json
        """

        self.__criteria_data = self.__data_json['criteria']

    def __set_cv_id( self ):

        """
            it's just set the key of the id that have the id of cv name.

            @param: object json 

            @return: string

        """

        self.__id_cv = self.__data_json['id']

    def get_id_cv( self ):

        """
            it's return the id that given from the json file

            @return: string

        """

        return self.__id_cv

    def get_cv_data( self ):

        """
            it's return the cv data that given from the json file

            @return: object json

        """

        return self.__cv_data

    def get_criteria_data( self ):

        """
            it's return the id that given from the json file

            @return: string

        """

        return self.__criteria_data

    def get_data( self , key , key_cri=None ,  parent="cv" ):
        
        """
            it's return the content of section name that given as a parameter

            @param: string
            @param: string

            @return: object json

        """
        if parent =="cv":
            
            return self.__data_json[parent][key][0]["section"]
        
        elif parent== "criteria":

            return self.__data_json[parent][key][key_cri]
            

    def save_json( self , data , filename="cv_json.json" ):
        
        """
            it's take the data as a dictionary and save it into a json file.

            @param: dictionary of data.

            @param: return Boolean.

        """

        with open( filename , "w" ) as f:
            f.write( json.dumps(data ) )


    def final_accuracy( self , accuraces ):

        sum = 0.0

        for i in range( len( accuraces ) ):

            sum += accuraces[i]

        return "%0.2f " % ( sum / len( accuraces ) ) + "%"

    def run_json( self , filename ):

        """
            it's a main function that load and proccess the class json that deal with the json data

            it's take the all requirement that needed and return you need from the json object

            @param: string 
            
        """

        data = self.__read_data( filename ) # call function read to open file and read the data

        self.__data_json = self.__load_json(data) # pass data to json to load into the class

        self.__set_cv_data()

        self.__set_criteria_data()
        
        self.__set_cv_id()

