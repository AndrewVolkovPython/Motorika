import pandas as pd

def model_data_prepare(self, use_add:bool =  False):
    """ Подготовка X, y для обучения

    Args:
        use_add (False): Использовать признаки из add_features_1

    Returns:
        _type_: Функция возвращает датасет для обучения, согласно выбранным признакам - X, датасет для проверки, 
        который не менялся, кроме сдвига данных - X_init, и y - целевой признак
    """
    # Использовать ли дополнительно полученные признаки
    self.use_add = use_add
    
    X = pd.DataFrame()
    X = self.gestures_clean[self.CLEAN_SENSORS_FINAL].copy()
    
    X_init = pd.DataFrame()
    X_init = self.gestures[self.CLEAN_SENSORS_FINAL].copy()

    if use_add:
        df_add_1 = self.additional_features_1[self.CLEAN_SENSORS_ADD_FEATURE_1].copy()
        if df_add_1.shape[0] != 0:
            X.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1] = df_add_1.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1].values
            X_init.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1] = df_add_1.loc[:,self.CLEAN_SENSORS_ADD_FEATURE_1].values
        else:
            print('Датасет с дополнительными признаками нулевой!!!')    
    y = self.gestures_clean['gesture'].values 
    
    return X, X_init, y