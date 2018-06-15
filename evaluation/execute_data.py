import cosine_simialrity as cos
import prepare_json as js

class excute_simialrity:

    def __init__( self ):

        self.__js = js.prepare_json()


    def get_data( self , sectionName ):

        self.__js.run_json("test.json")

        cv = self.__js.get_data(sectionName)  

        cri = self.__js.get_data(sectionName , "criteria")    

        return cv[0] , cri[0]

    def similarity( self , target , context ):

        return cos.cosineSimilarity(target ,context ).get_result()

    def calculate_matched( self , acc ):
        
        return self.__js.final_accuracy( acc )


if __name__ == "__main__":

    exc = excute_simialrity()

    target , context = exc.get_data("skills")
    
    acc1  = exc.similarity(target , context)

    target , context = exc.get_data("experience")
    
    acc2  = exc.similarity(target , context)
    
    print(acc1 , acc2 , exc.calculate_matched([ acc1 , acc2  ]))
