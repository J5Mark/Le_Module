import numpy as np
import keras
from abc import abstractmethod, ABC
from Utilities.FinUtils import Candles
from Utilities import FinUtils
from PredictorsManagement.Predictors import *

class NoSuchPredictableError(Exception):
    '''raised when user passes an object of a type that no predictor can predict in params.predictable'''
    def __init__(self, passedpred):
        self.message = f"no predictor can be constructed to predict {passedpred} (type: {type(passedpred)}). \nhas to be either str or market_condition"
        super().__init__(self.message)

class PredictorFactory(ABC):
    '''abstract factory for constructing different types of predictors'''
    @abstractmethod
    def construct_predictor(self) -> Predictor:
        pass

class AIPredictorFactory(PredictorFactory):
    def __init__(self, training_data : list[Candles], biascheck_data : list[Candles] | None=None):
        ''':data: is the Candles object that will be universally used to train the models inside an instance of the factory'''
        self.tr_datalist = training_data
        if biascheck_data == None:
            self.biascheck = self.tr_datalist
            self._biascheck_training_difference = False
        else:
            self.biascheck = biascheck_data
            self._biascheck_training_difference = True

    def construct_predictor(self, params: PredictorParams) -> Predictor:
        model = keras.Sequential()

        if any([isinstance(x, layers.RNN) for x in params.structure]):
            inp_shape = (params.input_width, params.input_len)
        else:
            inp_shape = (params.input_len, params.input_width)
        params.structure[0].build(inp_shape)
        params.structure[-1].units = params.output_len
        
        for each in params.structure[1:]:
            model.add(each)

        assert params.structure[-1].units == params.output_len

        model.compile(optimizer=params.optimizer, 
                      loss=params.loss,
                      metrics=params.metrics)
        
        biascheck_sets = []
        for data in self.tr_datalist:
            full = FinUtils.get_training_data(data, 
                                            len_of_sample=params.input_len, 
                                            len_of_label=params.output_len, 
                                            scope=params.scope, 
                                            predictable=params.predictable)
            samples, labels = full[0], full[1]
            
            try:
                samples = np.reshape(samples, (len(samples), inp_shape[0], inp_shape[1]))
            except:
                print(f'can not reshape an array with units of shape {samples.shape[1:]} to {inp_shape}')

            model.fit(samples, labels, batch_size=32, 
                    epochs=params.epochs, 
                    verbose=params.training_verbose,
                    callbacks=params.callbacks,
                    validation_split=params.validation_split,
                    shuffle=True,
                    use_multiprocessing=True)
            
            if not self._biascheck_training_difference: biascheck_sets.append((samples, labels))
        
        if isinstance(params.predictable, str):
            if params.output_len > 1:
                predictor = AISequencePredictor(v_model=model, predictable_value=params.predictable)
            elif params.output_len == 1:
                predictor = AITrendviewer(v_model=model, predictable_value=params.predictable)
        elif isinstance(params.predictable, market_condition):
            predictor = AIConditionPredictor(c_model=model, predictable=params.predictable)
        else:    
            raise NoSuchPredictableError(params.predictable)
        
        if isinstance(predictor, AITrendviewer) or isinstance(predictor, AISequencePredictor):
            if self._biascheck_training_difference:
                for data in self.biascheck:
                    full = FinUtils.get_training_data(data, 
                                                    len_of_sample=params.input_len, 
                                                    len_of_label=params.output_len, 
                                                    scope=params.scope, 
                                                    predictable=params.predictable)
                    samples, labels = full[0], full[1]
                    try:
                        samples = np.reshape(samples, (len(samples), model.input_shape[1], model.input_shape[2]))
                    except:
                        print(f'can not reshape an array of shape {samples.shape} to {model.input_shape} during prepping data for examining bias')
                        
                    biascheck_sets.append((samples, labels))

            print("examining predictor's bias:")
            predictor.examine_bias(biascheck_sets)
            print(f"predictor's bias based on average square deviation from true values: {predictor.avg_bias}\npredictor's bias based on median deviation from true values: {predictor.median_bias}")

        return predictor