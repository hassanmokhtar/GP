from keras.layers import Embedding  , Dense , Flatten
from keras.layers.convolutional import Conv1D , MaxPooling1D
from keras.models import Sequential
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Embedding_Model:

    def __init__(self):
        pass

    def build_model(self ,X_train , y_train, X_test , y_test ,  vocab_siz , vector_dim , input_length):

        model = Sequential()

        model.add(Embedding(vocab_siz , vector_dim , input_length=input_length))
        model.add( Conv1D( filters=32 , kernel_size=8 , activation='relu' ) )
        model.add( Conv1D( filters=64 , kernel_size=8 , activation='relu' ) )
        model.add(MaxPooling1D(6))
        model.add( Conv1D( filters=128 , kernel_size=8 , activation='relu' ) )
        model.add(MaxPooling1D(4))
        model.add(Flatten())
        model.add( Dense( 128 , activation='relu' ) )
        model.add( Dense( 1 , activation='sigmoid' ) )

        model.compile( loss='binary_crossentropy' , optimizer='rmsprop' , metrics=['accuracy'] )
        model.fit(X_train , y_train , validation_split=0.2 , batch_size=16 , epochs=15 , verbose=1)
        #loss , acc = model.evaluate(X_test , y_test)
        print( model.predict(X_test) )
        