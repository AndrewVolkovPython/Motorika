from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

def change_params_and_apply(self, cosine, sensor_power, cosine_add_feature_1, shift):
    """
    Обновляет параметры обработки жестов на основе заданных значений 
    косинусной схожести, мощности сенсоров и дополнительных признаков. 

    Метод выполняет последовательные шаги по очистке сенсоров, 
    вычислению медианных значений, получению векторов жестов, 
    а также создаёт исправленный DataFrame жестов на основе 
    указанных параметров.

    Параметры
    ----------
    cosine : float
        Порог для фильтрации сенсоров по косинусной схожести.
    
    sensor_power : float
        Порог для фильтрации сенсоров по мощности.
    
    cosine_add_feature_1 : float
        Порог для фильтрации дополнительных признаков на основе 
        косинусной схожести.
    
    shift : tuple
        Смещения для начала и конца жеста, используемые в методе 
        `create_fix_gestures_df`, по умолчанию (0, 0).

    Примечание
    ---------
    Данный метод последовательно вызывает другие методы для 
    очистки данных и получения необходимых векторов и DataFrame. 
    Вся информация о чистых сенсорах сохраняется в 
    `CLEAN_SENSORS_FINAL` и `CLEAN_SENSORS_ADD_FEATURE_1`.

    Примеры
    --------
    >>> change_params(0.5, 10, 0.6, (5, -3))
    >>> print(CLEAN_SENSORS_FINAL)
    >>> print(CLEAN_SENSORS_ADD_FEATURE_1)
    """        
    self.verbose = False
    self.clean_sensor_getter_by_cosine(cosine)
    self.median_sensor_power_getter()
    self.clean_sensor_getter_by_power(sensor_power)
    print(self.CLEAN_SENSORS_FINAL)
    self.clean_sensor_getter_by_cosine_add_feat_1(cosine_add_feature_1)
    print(self.CLEAN_SENSORS_ADD_FEATURE_1)
    self.get_vectors_final()
    self.get_df_final()
    self.cleaned_df_diff_adding()
    self.create_fix_gestures_df(shift)
    
def fit_and_predict(self, pilote_id : int, model_name : str, model_params : dict, selection_params : dict, use_add: bool,  stat_file : str = 'stats.json', test_size: float = 0.2, random_state : int = 42):
    
    self.pilote_id = pilote_id
    self.test_size_split = test_size
    self.random_state = random_state
    self.model_name = model_name
    
   
    # Проверка на правильный тип модели
    if model_name not in ['rf', 'lgb']:
        raise ValueError(f"Неподдерживаемый тип модели: {model}. Поддерживаемые типы 'rf' and 'lgb'.")
    
    # Проверка структуры selection_params
    required_keys = ['cosine', 'sensor_power', 'cosine_add_feature_1', 'shift']
    for key in required_keys:
        if key not in selection_params:
            raise ValueError(f"Отстутствуют обязательные параметры: '{key}'")

    change_params_and_apply(self
                            ,selection_params['cosine']
                            ,selection_params['sensor_power']
                            ,selection_params['cosine_add_feature_1']
                            ,selection_params['shift'])
    
    X,X_init,y =self.model_data_prepare(use_add)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    
    # Подбор модели
    if model_name == 'rf':
        model = RandomForestClassifier(**model_params)
    elif model_name == 'lgb':
        model = lgb.LGBMClassifier(**model_params, verbose=-1)
      
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Обучение модели
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_train_init = model.predict(X_init)
    
    self.get_statistic(y_train, y_pred_train,1, False)
    self.get_statistic(y_test, y_pred_test,2, False)
    self.get_statistic(y, y_pred_train_init,3, False)
    
    model_parameters = self.get_params()
    model_parameters['model_params'] = model_params
    self.save_stat(stat_file, model_parameters)