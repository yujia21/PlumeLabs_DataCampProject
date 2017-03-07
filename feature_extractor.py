# insert Frapy's feat extractor
import pandas as pd
from sklearn.base import TransformerMixin


class FeatureExtractor(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_ = pd.get_dummies(X_df, drop_first=False, columns=['pollutant'])
        return X_df_.values
