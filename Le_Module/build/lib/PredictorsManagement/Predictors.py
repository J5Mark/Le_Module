from abc import abstractmethod
from Utilities.FinUtils import *
import matplotlib.pyplot as plt
import numpy as np
import keras
import alive_progress
import time
from keras import layers
from PredictorsManagement.pm import *

class AIConditionPredictor(Predictor):
    def __init__(self, c_model: keras.Sequential, predictable_condition: market_condition, input_len: int=5, input_width: int=11):
        self.c_model = c_model
        if not(self.c_model): raise NoModelInsertedError(model=self.c_model)
        if not isinstance(predictable_condition, market_condition):
            raise WrongPredictableTypeError(predictable_condition, market_condition)
        self.predictable_condition = predictable_condition
        self.input_length = input_len
        self.input_width = input_width
        if self.input_length not in self.c_model.input_shape or self.input_width not in self.c_model.input_shape:
            raise WrongDimensionsError((self.input_length, self.input_width), self.c_model.input_shape)
        
    def train(self, dataset: tuple, epochs: int=100, verbose: bool=False, callbacks:list | None=None, validation_split: float=0.1) -> None:
        if self.c_model:
            self.c_model.fit(x=dataset[0], y=dataset[1], epochs=epochs, shuffle=True, verbose=verbose, callbacks=callbacks, validation_split=validation_split, batch_size=32)
        else:
            raise NoModelInsertedError(self.c_model)
        
    def probability_predict(self, databit: np.ndarray) -> float:
        if self.c_model:
            try:
                databit = np.reshape(databit, [1]+list(self.c_model.input_shape[1:]))
            except:
                print(f'seems like the sample (with shape: {databit.shape}) cannot be reshaped to the c_model`s input_shape: {self.c_model.input_shape[1:]}')
            prob = self.c_model.predict(databit, verbose=False)
            return np.reshape(prob, (1))[0]
        else:
            raise NoModelInsertedError(self.c_model)
    
    def predict_condition(self, databit: np.ndarray) -> bool:
        return bool(round(self.probability_predict(databit)))   
        
    def predict(self, databit) -> PredictorResponse:
        return PredictorResponse(pred=self.predict_condition(databit))
    
    def _examine_performance(self, dataset: Candles) -> tuple:
        dt = dataset.as_dataframe()
        try:
            real = [dataset.Close.values[i] if self.predictable_condition(dataset[i]) else None for i in range(input_len, len(dataset.Close))]
        except:
            print(f'something in {self.predictable_condition} is not a name of an indicator or a number, or it is just written incorrectly\nindicators in your dataset: \n{[i for i in dt.columns]}')
        
        predictions = []
        output_len = 1                

        if any([isinstance(x, layers.RNN) for x in self.c_model.layers]):
            input_len = self.c_model.input_shape[-1]
        else:
            input_len = self.c_model.input_shape[-2] 
        
        with alive_progress.alive_bar(len(dt)-output_len, force_tty=True, spinner='radioactive') as bar:
            for i in range(input_len, len(dataset.Close)-1):
                pr = self.predict(dt[i-input_len:i].values)
                predictions += [dataset.Close.values[i] if pr.pred else None]
                bar()   
        
        return real, predictions
            
    def _visualize_performance(self, dataset: Candles, real: list, predictions: list):
        plt.plot(dataset.Close.values[-len(real):], 'k', label='close')
        for each in dataset.MAs:
            plt.plot(each.values[-len(real):], 'darkgreen', label='mas')
        for each in dataset.BollingerBands:
            plt.plot(each.values[-len(real):], 'cyan', label='bollinger bands')

        plt.plot(predictions, color='darkred', marker='.', label='market condition was predicted to be true')
        plt.plot(real, color='darkblue', marker='.', label='market condition is true')
        plt.legend()
        plt.grid(True)
        plt.show()

    def see_preformance(self, dataset: Candles):
        real, p = self._examine_performance(dataset)
        self._visualize_performance(dataset, real, p)


class AIValuesPredictor(Predictor):
    def __init__(self, v_model: keras.Sequential, predictable_value: str='close', input_len: int=5, input_width: int=11):
        self.v_model = v_model
        if not(self.v_model): raise NoModelInsertedError(model=self.v_model)
        if not isinstance(predictable_value, str):
            raise WrongPredictableTypeError(predictable_value, str)
        self.predictable_value = predictable_value
        self.input_length = input_len
        self.input_width = input_width
        self.avg_bias = (0, 0)
        self.median_bias = (0, 0)
        if self.input_length not in self.v_model.input_shape or self.input_width not in self.v_model.input_shape:
            raise WrongDimensionsError((self.input_length, self.input_width), self.v_model.input_shape)
        
    def train(self, dataset: tuple, epochs: int=100, verbose: bool=False, callbacks:list | None=None, v_split: float=0.1) -> None:
        if self.v_model:
            self.v_model.fit(x=dataset[0], y=dataset[1], epochs=epochs, shuffle=True, verbose=verbose, callbacks=callbacks, validation_split=v_split, batch_size=32)
        else:
            raise NoModelInsertedError()
            
    def biased_predict(self, databit: np.ndarray) -> np.ndarray:
        if self.v_model:
            try:
                databit = np.reshape(databit, [1]+list(self.v_model.input_shape[1:]))
            except:
                print(f'seems like the sample (with shape: {databit.shape}) cannot be reshaped to the v_model`s input_shape: {self.v_model.input_shape[1:]}')
            
            pred = self.v_model.predict(databit, verbose=False)
            return np.reshape(pred, pred.shape[-1])
        else:
            raise NoModelInsertedError(self.v_model)
        
    def predict_value(self, databit: np.ndarray) -> PValue:
        biased = self.biased_predict(databit)
        return PValue(biased=biased, 
            avg_bias_adjusted=(biased+biased*self.avg_bias[0], biased+biased*self.avg_bias[1]),
            med_bias_adjusted=(biased+biased*self.median_bias[0], biased+biased*self.median_bias[1]))
            
    def predict(self, databit: np.ndarray) -> PredictorResponse:
        return PredictorResponse(pred=self.predict_value(databit))

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
        

class AITrendviewer(AIValuesPredictor):
    '''A type of predictor that only predicts a single value, a single candle in the future. It can be used for risk management or allowing trade for a certain period of time(greater than the operating dataframe)'''    
    def __init__(self, v_model: keras.Sequential, predictable_value: str='close', input_len: int=5, input_width: int=11):
        super().__init__(v_model, predictable_value, input_len, input_width)
        
    def examine_bias(self, training_sets: list[tuple]):
        avg_positives, avg_negatives, med_positives, med_negatives = [], [], [], []
        
        for dataset in training_sets:
            labels = dataset[1]
            samples = dataset[0]

            predicts = [] 
            biases = []
            with alive_progress.alive_bar(len(labels), force_tty=True, spinner='radioactive') as bar: 
                for i in range(len(labels)):
                    prediction = self.biased_predict(samples[i])
                    predicts += prediction.tolist()
                    biases.append((labels[i] - prediction)/(prediction+0.000001))
                    bar()
            
            predicts = np.array(predicts)
            predicts = np.append(np.array([None]*(len(samples[0]) + 1)), predicts)
            positive = [i for i in biases if i < 0]
            negative = [i for i in biases if i > 0]

            avg_positives.append(np.mean(positive))
            avg_negatives.append(np.mean(negative))
            med_positives.append(np.median(positive))
            med_negatives.append(np.median(negative))
        self.avg_bias = (np.mean(avg_positives), np.mean(avg_negatives))
        self.median_bias = (np.mean(med_positives), np.mean(avg_negatives))

    def _examine_performance(self, dataset: Candles):
        dt = dataset.as_dataframe()

        try:
            real = dt[self.predictable_value]
        except KeyError:
            print(f'{self.predictable_value} is not a name of an indicator, or it is just written incorrectly\nshould be in: \n{[i for i in dt.columns]}')
        
        avg_predicted1 = []
        avg_predicted3 = []
        predicted = []
        median_predicted1 = []
        median_predicted3 = []

        output_len = self.v_model.output_shape[-1]
        if any([isinstance(x, layers.RNN) for x in self.v_model.layers]):
            input_len = self.v_model.input_shape[-1]
        else:
            input_len = self.v_model.input_shape[-2]

        with alive_progress.alive_bar(len(dt)-output_len, force_tty=True, spinner='radioactive') as bar:
            for i in range(input_len, len(real)-output_len):
                pr = self.predict(dt[i-input_len:i].values)
                predicted.append(pr.pred.biased[0])

                avg_predicted1.append(pr.pred.avg_bias_adjusted[0])
                avg_predicted3.append(pr.pred.avg_bias_adjusted[1])

                median_predicted1.append(pr.pred.med_bias_adjusted[0])
                median_predicted3.append(pr.pred.med_bias_adjusted[1])
                bar()

        return real, predicted, (avg_predicted1, avg_predicted3), (median_predicted1, median_predicted3)
        

class AISequencePredictor(AIValuesPredictor):
    def examine_bias(self, training_sets: list[tuple]):
        avg_positives, avg_negatives, med_positives, med_negatives = [], [], [], []
        
        for dataset in training_sets:
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

            avg_positives.append(np.mean(positive))
            avg_negatives.append(np.mean(negative))
            med_positives.append(np.median(positive))
            med_negatives.append(np.median(negative))
        self.avg_bias = (np.mean(avg_positives), np.mean(avg_negatives))
        self.median_bias = (np.mean(med_positives), np.mean(med_negatives))
    
    def _examine_performance(self, dataset: Candles):
        dt = dataset.as_dataframe()

        try:
            real = dt[self.predictable_value]
        except KeyError:
            print(f'{self.predictable_value} is not a name of an indicator, or it is just written incorrectly\n should be in: \n{[i for i in dt.columns]}')
        
        avg_predicted1 = []
        avg_predicted3 = []

        predicted = []

        median_predicted1 = []
        median_predicted3 = []
        output_len = self.v_model.output_shape[-1]
        input_len = self.v_model.input_shape[-1] if any([isinstance(x, layers.RNN) for x in self.v_model.layers]) else self.v_model.input_shape[-2]
        with alive_progress.alive_bar((len(dt)-output_len)//output_len, force_tty=True, spinner='radioactive') as bar:
            for i in range(input_len, len(real)-output_len, output_len):
                pr = self.predict(dt[i-input_len:i].values)
                predicted += pr.pred.biased.tolist()

                avg_predicted1 += pr.pred.avg_bias_adjusted[0].tolist()
                avg_predicted3 += pr.pred.avg_bias_adjusted[1].tolist()

                median_predicted1 += pr.pred.med_bias_adjusted[0].tolist()
                median_predicted3 += pr.pred.med_bias_adjusted[0].tolist()
                bar()

        return real, predicted, (avg_predicted1, avg_predicted3), (median_predicted1, median_predicted3)