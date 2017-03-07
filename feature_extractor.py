# insert Frapy's feat extractor
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


class FeatureExtractor(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        # Make sure each polluant is there
        X_df['NO2'] = (X_df['pollutant']=='NO2').astype(int)
        X_df['PM10'] = (X_df['pollutant']=='PM10').astype(int)
        X_df['PM2_5'] = (X_df['pollutant']=='PM2_5').astype(int)
        X_df = X_df.drop(['pollutant'], axis = 1)
        return X_df.values
