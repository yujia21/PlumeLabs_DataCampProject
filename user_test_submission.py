from __future__ import print_function
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit
import feature_extractor
import regressor
import time
if __name__ == '__main__':
    tic=time.time()
    print('Reading file ...')
    X_df = pd.read_csv('X_train.csv')
    y_array = pd.read_csv('Y_train.csv')

    # Merge by ID
    full_df = X_df.merge(y_array, left_on='ID', right_on='ID')

    # Reseparate
    X_df = full_df.drop(['TARGET'], axis=1)
    y_array = full_df[['ID','TARGET']]

    station_list = list(set(X_df['station_id'].values))
    zone_station_list = X_df.groupby(['zone_id'])['station_id'].aggregate(lambda x : set(x)).values

    print('Training file ...')
    for i in range(4) :
        # Choose 12 stations to predict
        stations_test = []
        for zone in zone_station_list : 
            if (len(zone) == 3):
                stations_test.append(random.sample(zone, 1)[0])
        stations_selected = list(set(station_list) - set(stations_test))
        print('Stations selected : '+str(stations_selected))

        # Separate data in train/test
        X_train_df = X_df[X_df['station_id'].isin(stations_selected)]
        X_train_df = X_train_df.drop(['station_id'], axis=1)
        y_train_array = y_array[y_array['ID'].isin(X_train_df['ID'])]['TARGET'].values

        X_test_df = X_df[X_df['station_id'].isin(stations_test)]
        X_test_df = X_test_df.drop(['station_id'], axis=1)
        y_test_array = y_array[y_array['ID'].isin(X_test_df['ID'])]['TARGET'].values

        # Feature extraction
        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_train_df, y_train_array)
        X_train_array = fe.transform(X_train_df)

        reg = regressor.Regressor()
        reg.fit(X_train_array, y_train_array)
        # Feature extraction
        X_test_array = fe.transform(X_test_df)

        # Regression
        y_pred_array = reg.predict(X_test_array)
        print('rmse = ', np.sqrt(
            np.mean(np.square(y_test_array - y_pred_array))))
    tac=time.time()
    print('time = ', tac-tic)
