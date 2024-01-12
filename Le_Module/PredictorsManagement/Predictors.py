from abc import ABC, abstractmethod
from Utilities.FinUtils import *
import matplotlib.pyplot as plt
import numpy as np
import keras
import alive_progress
import time

class Predictor(ABC):
    '''abstract base class for all predictors'''
    @abstractmethod
    def predict(self, databit):
        pass
           
@dataclass
class PredictorParams:
    '''dataclass for storing all upcoming AI predictor parameters. 
     :structure: is a list of all layers of a future sequential predictor'''
    loss: keras.losses.Loss
    optimizer: keras.optimizers.Optimizer | str
    structure: list[keras.layers.Layer]
    scope: int=1
    input_len: int=5   
    output_len: int=1 

    epochs: int=300
    training_verbose: bool=False
    callbacks: list | None = None
    predictable: str = 'close'
    metrics: list[str] | None = None

class AIPredictor(Predictor):
    def __init__(self, model: keras.Sequential, predictable):
        self.model = model
        self.predictable = predictable
        self.avg_bias = (0, 0)
        self.median_bias = (0, 0)
    
    def train(self, dataset: tuple, epochs: int=100, verbose: bool=False, callbacks=list | None) -> None:
        self.model.fit(x=dataset[0], y=dataset[1], epochs=epochs, shuffle=True, verbose=verbose, callbacks=callbacks, validation_split=0.1, batch_size=32)
   
    def biased_predict(self, databit: np.array) -> np.array:
        n_features = databit.shape[1]
        len_of_sample = databit.shape[0]
        if any([lambda x: isinstance(x, keras.layers.RNN) for x in self.model.layers]):
            shape = (1, n_features, len_of_sample)
        else:
            shape = (len_of_sample, n_features)
        
        p = self.model.predict(np.reshape(databit, shape), verbose=False)
        return np.reshape(p, p.shape[-1])

    def predict(self, databit: np.ndarray) -> tuple:
        biased = self.biased_predict(databit)
        return ((biased+biased*self.avg_bias[0], biased, biased+biased*self.avg_bias[1]), (biased+biased*self.median_bias[0], biased, biased+biased*self.median_bias[1]))
 
    @abstractmethod
    def examine_bias(self, dataset: tuple):
        pass

    @abstractmethod
    def _examine_performance(self, dataset: Candles) -> tuple:
        pass

    def _visualize_performance(self, real, predicted, avg_predicted: tuple, median_predicted: tuple):
        plt.plot(avg_predicted[0], 'green', label='average biased')
        plt.plot(avg_predicted[1], 'green')

        plt.plot(median_predicted[0], 'cyan', label='median biased')
        plt.plot(median_predicted[1], 'cyan')

        plt.plot(predicted, 'darkred', label='predicted values')
        plt.plot(real, 'black', label='real values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def see_performance(self, dataset: Candles):
        '''Plots out a a performance graph of a predictor'''
        real, p, avg_p, median_p = self._examine_performance(dataset)
        self._visualize_performance(real=real, predicted=p, avg_predicted=avg_p, median_predicted=median_p)
        

class AITrendviewer(AIPredictor):
    '''A type of predictor that only predicts a single value, a single candle in the future. It can be used for risk management or allowing trade for a certain period of time(greater than the operating dataframe)'''    
    def examine_bias(self, dataset: tuple):
        labels = dataset[1]
        samples = dataset[0]

        predicts = [] 
        biases = []
        with alive_progress.alive_bar(len(labels), force_tty=True, spinner='radioactive') as bar: 
            for i in range(len(labels)):
                time.sleep(0.005)
                prediction = self.biased_predict(samples[i])
                predicts += prediction.tolist()
                biases.append((labels[i] - prediction)/(prediction+0.000001))
                bar()
        
        predicts = np.array(predicts)
        predicts = np.append(np.array([None]*(len(samples[0]) + 1)), predicts)
        positive = [i for i in biases if i < 0]
        negative = [i for i in biases if i > 0]

        avg_positive = np.mean(positive)
        avg_negative = np.mean(negative)
        self.avg_bias = (avg_positive, avg_negative)
        self.median_bias = (np.median(positive), np.median(negative))

    def _examine_performance(self, dataset: Candles):
        dt = dataset.as_dataframe()

        try:
            real = dt[self.predictable]
        except KeyError:
            print(f'{self.predictable} is not a name of an indicator, or it is just written incorrectly\n should be in: \n{[i for i in dt.columns]}')
        
        avg_predicted1 = []
        avg_predicted3 = []

        predicted = []

        median_predicted1 = []
        median_predicted3 = []

        output_len = self.model.output_shape[-1]
        input_len = self.model.input_shape[-1]
        with alive_progress.alive_bar(len(dt)-output_len, force_tty=True, spinner='radioactive') as bar:
            for i in range(input_len, len(real)-output_len):
                pr = self.predict(dt[i-input_len:i].values)
                predicted.append(pr[0][1])

                avg_predicted1.append(pr[0][0])
                avg_predicted3.append(pr[0][2])

                median_predicted1.append(pr[1][0])
                median_predicted3.append(pr[1][2])
                bar()

        return real, predicted, (avg_predicted1, avg_predicted3), (median_predicted1, median_predicted3)
        

class AISequencePredictor(AIPredictor):
    def examine_bias(self, dataset: tuple):
        labels = dataset[1]
        samples = dataset[0]

        predicts = [] 
        biases = []
        with alive_progress.alive_bar(len(labels), force_tty=True, spinner='radioactive') as bar: 
            for i in range(len(labels)):
                time.sleep(0.005)
                prediction = self.biased_predict(samples[i])
                predicts += prediction.tolist()
                biases += ((labels[i] - prediction)/(prediction+0.000001)).tolist()
                bar()
        
        predicts = np.array(predicts)
        predicts = np.append(np.array([None]*(len(samples[0]) + 1)), predicts)
        positive = [i for i in biases if i < 0]
        negative = [i for i in biases if i > 0]

        avg_positive = np.mean(positive)
        avg_negative = np.mean(negative)
        self.avg_bias = (avg_positive, avg_negative)
        self.median_bias = (np.median(positive), np.median(negative))
    
    def _examine_performance(self, dataset: Candles):
        dt = dataset.as_dataframe()

        try:
            real = dt[self.predictable]
        except KeyError:
            print(f'{self.predictable} is not a name of an indicator, or it is just written incorrectly\n should be in: \n{[i for i in dt.columns]}')
        
        avg_predicted1 = []
        avg_predicted3 = []

        predicted = []

        median_predicted1 = []
        median_predicted3 = []
        output_len = self.model.output_shape[-1]
        input_len = self.model.input_shape[-1]
        with alive_progress.alive_bar((len(dt)-output_len)//output_len, force_tty=True, spinner='radioactive') as bar:
            for i in range(input_len, len(real)-output_len, output_len):
                pr = self.predict(dt[i-input_len:i].values)
                predicted += pr[0][1].tolist()

                avg_predicted1 += pr[0][0].tolist()
                avg_predicted3 += pr[0][2].tolist()

                median_predicted1 += pr[1][0].tolist()
                median_predicted3 += pr[1][2].tolist()
                bar()

        return real, predicted, (avg_predicted1, avg_predicted3), (median_predicted1, median_predicted3)