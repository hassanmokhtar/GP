import cosine_simialrity as cos
import prepare_json as js

class excute_simialrity:

    def __init__( self , filename ):
        
        """
            it's a class that run the simialrity from json file.

            @param: string of filename with a path.
        """
        self.__js = js.prepare_json() # create object from class prepare json to get the data

        self.__js.run_json( filename ) # call function run the read and prepare the data.


    def get_data( self , sectionName ):

        """
            it's a function that take the section name that need data from it and return two
            values the first is target data and second the context that need to similarity with it
            it's store into criteria.

            @param: string of section name.

            @return: string target , string context
        """
        
        cv = self.__js.get_data(sectionName)  

        cri = self.__js.get_data(sectionName , "criteria")    

        return cv , cri


    def similarity( self , target , context ):
        
        """
            it's function that call class simialrity to pass the target and context to get the accuracy
            of the simialrity.

            @param: string of target data.
            @param: string of context data.

            @return: float of accuracy.

        """

        return cos.cosineSimilarity(target ,context ).get_result()

    def calculate_matched( self , acc ):
        
        """
            it's function that take all accuracies as a list from all sections and calculate the average.

            @param: list of float accuracy.

            @return: float of final accuracy.

        """
        
        return self.__js.final_accuracy( acc )


if __name__ == "__main__":

    import sys # it's used for read the string of filename when execute the class.

    filename = sys.argv[1]

    if "." in filename:
        
        check = filename.split(".")[1]

        if check == "json":
            exc = excute_simialrity( filename )

            target , context = exc.get_data( "educations" ) #operation section education
            
            acc1  = exc.similarity(target , context)

            target , context = exc.get_data( "experiences" ) #operation section experiences
            
            acc2  = exc.similarity(target , context)

            target , context = exc.get_data( "personal_data" ) #operation section personal_data
            
            acc3  = exc.similarity(target , context)

            target , context = exc.get_data( "skills" ) #operation section skills
            
            acc4  = exc.similarity(target , context)

            print( acc1 , acc2 , acc3 , acc4 , exc.calculate_matched( [ acc1 , acc2 , acc3 , acc4 ] ) )

        else:
            print("the input should be json file only")
    else:

        print("the input should be file end with extension 'json'")

        sys.exit(0)
    
