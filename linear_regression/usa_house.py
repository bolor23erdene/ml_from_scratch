import csv
import random
import math
import operator
import pandas as pd
import numpy as np

class Dataloader:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.length = len(self.data)
        #self.drop_address()
        #self.feature, self.label = self.get_feature_label()
        #self.normalize()
        #self.bias()




class LinearRegression:
    def __init__(self, path='./USA_Housing.csv'):
        self.dataloader = Dataloader(path)
        self.train, self.val = self.dataloader.train_val_split



if __name__ == '__main__':
    lr = LinearRegression()
    print(lr.train)
    #lr.fit()
    #print('MSE:', lr.get_mse())