import numpy as np
import keras
from abc import abstractmethod, ABC
from Utilities.FinUtils import Candles
from Utilities import FinUtils
from PredictorsManagement.Predictors import *

class PredictorFactory(ABC):
    '''abstract factory for constructing different types of predictors'''
    @abstractmethod
    def construct_predictor(self) -> Predictor:
        pass

class AIPredictorFactory(PredictorFactory):
    def __init__(self, data : list[Candles]):
        ''':data: is the Candles object that will be universally used to train the models inside an instance of the factory'''
        self.datalist = data

    def construct_predictor(self, params: PredictorParams) -> AIPredictor:
        model = keras.Sequential()
        
        for each in params.structure:
            model.add(each)
        assert params.input_len in params.structure[0].input_shape
        assert params.structure[-1].units == params.output_len

        model.compile(optimizer=params.optimizer, 
                      loss=params.loss,
                      metrics=params.metrics)
        
        training_sets = []
        for data in self.datalist:
            full = FinUtils.get_training_data(data, 
                                            len_of_sample=params.input_len, 
                                            len_of_label=params.output_len, 
                                            scope=params.scope, 
                                            predictable=params.predictable)
            samples, labels = full[0], full[1]
            
            try:
                samples = np.reshape(samples, (len(samples), model.input_shape[1], model.input_shape[2]))
            except:
                print(f'can not reshape an array of shape {samples.shape} to {model.input_shape}')

            model.fit(samples, labels, batch_size=32, 
                    epochs=params.epochs, 
                    verbose=params.training_verbose,
                    callbacks=params.callbacks,
                    validation_split=params.validation_split,
                    shuffle=1,
                    use_multiprocessing=1)
            training_sets.append((samples, labels))
        
        if params.output_len > 1:
            predictor = AISequencePredictor(model=model, predictable=params.predictable)
        elif params.output_len == 1:
            predictor = AITrendviewer(model=model, predictable=params.predictable)

        print("examining predictor's bias:")
        predictor.examine_bias(training_sets)

        return predictor