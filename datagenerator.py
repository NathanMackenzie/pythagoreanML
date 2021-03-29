import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self, file_path, names=None, features=None, labels=None):
        raw_data = pd.read_csv(file_path, names=names)
        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()
        #Parse data into features and labels
        for i, feat in enumerate(features):
            self.features.insert(i, feat, raw_data.pop(feat))
        for i, label in enumerate(labels):
            self.labels.insert(i, label, raw_data.pop(label))
        #Cast data to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        print(self.labels)

        



        
        

    