import pandas as pd

def model_data_prepare(self, use_add: False):
    self.use_add = use_add
    X = pd.DataFrame()
    X = self.gestures_clean[self.CLEAN_SENSORS_FINAL].copy()
    X_init = pd.DataFrame()
    X_init = self.gestures[self.CLEAN_SENSORS_FINAL].copy()

    if use_add:
        df_add_1 = self.additional_features_1[self.CLEAN_SENSORS_ADD_FEATURE_1].copy()
        X.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1] = df_add_1.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1].values
        X_init.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1] = df_add_1.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1].values
    y = self.gestures_clean['gesture'].values 
    
    return X, X_init, y