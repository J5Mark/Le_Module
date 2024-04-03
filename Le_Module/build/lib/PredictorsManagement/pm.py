from abc import ABC, abstractmethod
from Utilities.DataClasses import Candles
import numpy as np
import keras
from dataclasses import dataclass

class WrongDimensionsError(Exception):
    '''Exception raised when a dimension of predictor's input is not in line with predictor's model's input shape'''
    def __init__(self, param_dimensions: tuple, model_dimensions: tuple):
        self.message = f"\nsome of the dimensions specified in parameters of the predictor ({param_dimensions}) are not even in it's model's input_shape ({model_dimensions})"
        super().__init__(self.message)

class NoModelInsertedError(Exception):
    '''raised when predictor has no ml model'''
    def __init__(self, model):
        self.message = f"predictor has no models in it.\n{model} was provided as a model"
        super().__init__(self.message)

class WrongPredictableTypeError(Exception):
    '''raised when :predictable: of an Predictor is of wrong type(must be either market_condition or str)'''
    def __init__(self, predictable, correct_type: type):
        self.message = f"predictable of a wrong type was provided. Should be str or market_condition. \n{predictable}(type: {type(predictable)}) was provided. should be a: {correct_type}"
        super().__init__(self.message)
        
class market_condition(ABC):
    '''a callable object designed to represent a condition that is checked or predicted. \nTo design your own subclass just design the __call__(method)'''
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, c: Candles) -> bool:
        pass

class Predictor(ABC):
    '''abstract base class for all predictors'''
    @abstractmethod
    def predict(self, databit):
        pass
    
    @abstractmethod
    def _examine_performance(self, dataset: Candles) -> tuple:
        pass           
    
    @abstractmethod
    def _visualize_performance(self):
        pass
    
@dataclass
class PValue:
    biased: float | np.ndarray 
    avg_bias_adjusted: tuple
    med_bias_adjusted: tuple       

@dataclass
class PredictorResponse():
    pred: PValue | bool
           
@dataclass
class PredictorParams:
    '''dataclass for storing all upcoming AI predictor parameters. 
     :structure: is a list of all layers of a future sequential predictor'''
    loss: keras.losses.Loss
    optimizer: keras.optimizers.Optimizer | str
    structure: list[keras.layers.Layer]
    scope: int=1
    input_len: int=5   
    input_width: int=14
    output_len: int=1 

    epochs: int=300
    training_verbose: bool = False
    callbacks: list | None = None
    predictable: str | market_condition = 'close'
    metrics: list[str] | None = None,
    validation_split: float = 0.1