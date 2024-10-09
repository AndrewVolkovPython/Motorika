import os
import json
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

def save_stat(self, file_path : str, new_object):
    """
    Сохраняет новый объект в файл в формате JSON. Если файл не существует,
    создается новый файл с пустым массивом. Если файл поврежден, выводится сообщение об ошибке.

    Параметры:
    ----------
    file_path : str
        Путь к файлу, в который необходимо сохранить данные.
    
    new_object :
        Объект, который будет добавлен в массив данных. Может быть любым объектом,
        который может быть сериализован в JSON.

    Исключения:
    -----------
    json.JSONDecodeError:
        Возникает, если файл содержит некорректный JSON и не может быть прочитан.
    """    
    # Проверка, существует ли файл
    if not os.path.exists(file_path):
        # Если файл не существует, создаем новый с пустым массивом
        with open(file_path, 'w') as file:
            json.dump([], file)
    
    # Читаем существующие данные
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)  # Загружаем массив из файла
        except json.JSONDecodeError:
            print("Ошибка при чтении файла. Возможно, файл поврежден.")
            return

    # Добавляем новый объект в массив
    data.append(new_object)

    # Записываем обновленный массив обратно в файл
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # Форматируем вывод для читаемости
    print("Новый объект добавлен успешно.")
    
def read_stat(self,file_path : str) -> str:
    """
    Читает и загружает содержимое файла в формате JSON.

    Параметры:
    ----------
    file_path : str
        Путь к файлу, который необходимо прочитать.

    Возвращает:
    -----------
    str:
        Содержимое файла в формате JSON, если загрузка прошла успешно.

    Исключения:
    -----------
    json.JSONDecodeError:
        Возникает, если файл содержит некорректный JSON.
    Exception:
        Общая ошибка чтения файла, если что-то пошло не так.
    """    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # Пытаемся загрузить весь файл как JSON
            data = json.loads(content)
            return data
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON: {e}")
    except Exception as e:
        print(f"Ошибка: {e}")


def get_stat(self, file_path : str) -> pd.DataFrame:
    """
    Читает JSON файл, обрабатывает статистику для stat_train, stat_test и stat_init, 
    добавляет количество признаков для каждой записи, и возвращает объединённый DataFrame.

    Параметры:
    ----------
    file_path : str
        Путь к файлу с данными в формате JSON.

    Возвращает:
    -----------
    pd.DataFrame
        Обработанный DataFrame с расширенной информацией по статистике и количеству признаков.
    """    
    df = pd.read_json(file_path)
    
    # Преобразуем каждую строку с JSON в словарь
    df['stat_train'] = df['stat_train'].apply(json.loads)
    df['stat_test'] = df['stat_test'].apply(json.loads)
    df['stat_init'] = df['stat_init'].apply(json.loads)

    # Преобразуем вложенные словари в отдельные DataFrame
    stat_train_df = pd.json_normalize(df['stat_train'])
    stat_test_df = pd.json_normalize(df['stat_test'])
    stat_init_df = pd.json_normalize(df['stat_init'])

    # Объединяем данные train, test и init по горизонтали
    result_df = pd.concat([df.drop(columns=['stat_train','stat_test','stat_init']), stat_train_df], axis=1)
    result_df = pd.concat([df.drop(columns=['stat_train','stat_test','stat_init']), stat_test_df], axis=1)
    result_df = pd.concat([df.drop(columns=['stat_train','stat_test','stat_init']), stat_init_df], axis=1)

    # Добавляем количество признаков
    result_df['features_num'] = result_df.clean_sensor_final.apply(lambda x: len(x))
    result_df['features_add_1_num'] = result_df.clean_sensors_add_feature_1.apply(lambda x: len(x)) 
    
    return result_df    

def get_params(self) -> dict:
    """
    Возвращает параметры текущего объекта в виде словаря.

    Параметры:
    ----------
    Нет.

    Возвращает:
    -----------
    dict:
        Словарь с ключевыми параметрами:
        - 'pilote_id' : int — идентификатор пилота.
        - 'cosine' : float — значение косинусного сходства.
        - 'sensor_power' : float — коридор отбора для соабых сигналов, указывается в процентах от максимального сигнала.
        - 'cosine_add_feature_1' : float — косинус для дополнительного признака 1.
        - 'clean_sensor_final' : list — очищенные данные сенсоров.
        - 'clean_sensors_add_feature_1' : list — очищенные данные сенсоров для дополнительного признака 1.
        - 'use_add' : bool — флаг использования дополнительных признаков.
        - 'stat_train' : str — статистика для тренировочных данных.
        - 'stat_test' : str — статистика для тестовых данных.
        - 'stat_init' : str — статистика для всего набора данных.
        - 'shift' : float — значение сдвига.
    """    
    return {
        'pilote_id' : int(self.pilote_id),
        'cosine' : float(self.cosine),
        'sensor_power' : float(self.sensor_power),
        'cosine_add_feature_1' : float(self.cosine_add_feature_1),
        'clean_sensor_final' : self.CLEAN_SENSORS_FINAL,
        'clean_sensors_add_feature_1' : self.CLEAN_SENSORS_ADD_FEATURE_1,
        'use_add' : bool(self.use_add),
        'stat_train' : self.stat_train,            
        'stat_test' : self.stat_test,            
        'stat_init' : self.stat_init, 
        'shift' : self.shift,         
        'random_state' : int(self.random_state),
        'test_size': float(self.test_size_split), 
        'model' : self.model_name
    }

def get_statistic(self, train, pred, type : int, show : bool = True):
    """
    Вычисляет и сохраняет статистику по классификации, выводя отчёт для заданных предсказаний.

    Параметры:
    ----------
    train : array-like
        Массив истинных меток (обучающие данные).
    
    pred : array-like
        Массив предсказанных меток.
    
    type : int
        Тип статистики для сохранения:
        - 1: сохраняет в self.stat_train
        - 2: сохраняет в self.stat_test
        - 3: сохраняет в self.stat_init
    
    show : bool, по умолчанию True
        Если True, выводит на экран подробный отчёт классификации.

    Возвращает:
    -----------
    None
    
    Описание:
    ---------
    - Генерирует отчёт классификации с помощью sklearn's `classification_report`.
    - Преобразует отчёт в DataFrame и сохраняет в JSON-формате.
    - В зависимости от параметра `type`, сохраняет отчёт в переменные: 
      `self.stat_train`, `self.stat_test`, или `self.stat_init`.
    - Если `show=True`, выводит DataFrame отчёта на экран.
    
    Пример:
    -------
    >>> self.get_statistic(train_data, predictions, type=1, show=True)
    """
        
    pd.options.display.float_format = '{:.3f}'.format
    report_train = classification_report(train, pred, target_names=self.GESTURES, output_dict=True)
    report_train = pd.DataFrame(report_train).transpose()
    report_json = report_train.loc[self.GESTURES,:].iloc[:,:-1].to_json()
    if show:
        print(report_train)
    if type == 1:
        self.stat_train = report_json
    if type == 2:
        self.stat_test = report_json
    if type == 3:
        self.stat_init = report_json
        
def __stat_mean_calculate(row):
    stat_train = pd.DataFrame(json.loads(row['stat_train'])).mean().mean()
    stat_test = pd.DataFrame(json.loads(row['stat_test'])).mean().mean()
    stat_init = pd.DataFrame(json.loads(row['stat_init'])).mean().mean()    
    
    return np.array([stat_train, stat_test, stat_init]).mean()    
        
def get_min_max_stat_for_pilote(self, df : pd.DataFrame, pilote_id : int):
    """
    Возвращает минимальные и максимальные статистические значения (precision, recall, f1-score) для конкретного пилота.

    Параметры:
    ----------
    df : pd.DataFrame
        Исходный DataFrame, содержащий статистические данные по пилотам.
    
    pilote_id : int
        Идентификатор пилота, для которого необходимо вычислить статистику.

    Возвращает:
    -----------
    dict
        Словарь с ключами:
        - 'min': минимальное значение среднего по precision, recall, f1-score.
        - 'min_id': индекс строки с минимальным значением.
        - 'max': максимальное значение среднего по precision, recall, f1-score.
        - 'max_id': индекс строки с максимальным значением.
    
    Описание:
    ---------
    - Фильтрует DataFrame по указанному `pilote_id`.
    - Для каждого пилота вычисляет среднее значение для precision, recall и f1-score.
    - Возвращает минимальные и максимальные средние значения, а также индексы строк, где эти значения были найдены.
    
    Пример:
    -------
    >>> self.get_min_max_stat_for_pilote(df_stats, pilote_id=2)
    {'min': 0.87, 'min_id': 3, 'max': 0.95, 'max_id': 1}
    """    
    
    pd.options.display.float_format = '{:.3f}'.format
    
    stat_mean = df[df.pilote_id == pilote_id]
    stat_mean = stat_mean.copy()
    stat_mean.loc[:,'mean'] = stat_mean.apply(__stat_mean_calculate, axis = 1)
    
    rf = stat_mean[stat_mean.model == 'rf']
    lgb = stat_mean[stat_mean.model == 'lgb']
    
    final = {}

    if lgb.shape[0] > 0:
        final['lgb'] =  {
        'min': lgb['mean'].min(),
        'min_id': lgb['mean'].idxmin(),
        'max': lgb['mean'].max(),
        'max_id': lgb['mean'].idxmax()
        }
        
        
    if rf.shape[0] > 0:
        final['rf'] = {
        'min': rf['mean'].min(),
        'min_id': rf['mean'].idxmin(),
        'max': rf['mean'].max(),
        'max_id': rf['mean'].idxmax()
        }    

    return final

def get_model_params_by_id(self, df : pd.DataFrame, record_id : int) -> dict:
    """
    Возвращает параметры модели по указанному идентификатору записи.

    Параметры:
    ----------
    df : pd.DataFrame
        DataFrame, содержащий параметры моделей, включая колонку с параметрами 'rf_params'.
    
    record_id : int
        Идентификатор записи, для которой нужно получить параметры модели.

    Возвращает:
    -----------
    dict
        Параметры модели и тип модели, в виде ключа, соответствующие указанному `record_id`.

    Описание:
    ---------
    - Функция ищет запись по идентификатору `record_id` в DataFrame `df`.
    - Возвращает параметры модели, хранящиеся в столбце 'rf_params' для этой записи.
    
    Пример:
    -------
    >>> self.get_model_params_by_id(df_models, record_id=5)
    {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    """    
    return {df.loc[record_id, 'model'] : df.loc[record_id, 'model_params']}   

def get_params_by_id(self, df:pd.DataFrame, id : int) -> dict:
    """
    Возвращает параметры по заданному идентификатору.

    Параметры:
    ----------
    df : pd.DataFrame
        DataFrame, содержащий параметры, такие как 'cosine', 'sensor_power', 'cosine_add_feature_1', 
        'use_add', 'shift', 'features_num', 'features_add_1_num'.
    
    id : int
        Идентификатор записи, для которой нужно получить параметры.

    Возвращает:
    -----------
    dict
        Словарь с параметрами, соответствующими указанной записи, содержащий следующие ключи:
        - 'cosine'
        - 'sensor_power'
        - 'cosine_add_feature_1'
        - 'use_add'
        - 'shift'
        - 'features_num'
        - 'features_add_1_num'
    
    Описание:
    ---------
    - Функция ищет запись по указанному идентификатору `id` в DataFrame `df`.
    - Извлекает значения для указанных параметров: 'cosine', 'sensor_power', 'cosine_add_feature_1', 'use_add', 'shift', 
      'features_num', 'features_add_1_num'.
    - Возвращает эти значения в виде словаря.

    Пример:
    -------
    >>> self.get_params_by_id(df, id=5)
    {
        'cosine': 0.99,
        'sensor_power': 1.5,
        'cosine_add_feature_1': 0.98,
        'use_add': True,
        'shift': 2,
        'features_num': 50,
        'features_add_1_num': 10
    }
    """    
    params = df.loc[id,['cosine', 'sensor_power','cosine_add_feature_1','use_add', 'shift','features_num','features_add_1_num']]
    return {
         'cosine' : params.cosine
        ,'sensor_power': params.sensor_power
        ,'cosine_add_feature_1': params.cosine_add_feature_1
        ,'use_add': params.use_add
        ,'shift': params.shift
        ,'features_num' : params.features_num
        ,'features_add_1_num': params.features_add_1_num
        
    }
    
def get_statistic_by_id(self, df, record_id) -> pd.DataFrame:
    """
    Извлекает и объединяет статистические данные (precision, recall, f1-score) для заданной записи.

    Параметры:
    ----------
    df : pd.DataFrame
        DataFrame, содержащий записи с колонками 'stat_train', 'stat_test', 'stat_init', в которых хранятся данные в формате JSON.
    
    record_id : int
        Идентификатор записи, для которой необходимо извлечь статистику.

    Возвращает:
    -----------
    pd.DataFrame
        DataFrame с объединённой статистикой precision, recall и f1-score для категорий ['Neutral', 'Open', 'Pistol', 'Thumb', 'OK', 'Grab'].
        Колонки имеют многоуровневый индекс с категориями ['precision', 'recall', 'f1-score'] на первом уровне и 
        ['stat_train', 'stat_test', 'stat_init'] на втором уровне.

    Описание:
    ---------
    - Функция извлекает строки по указанному `record_id` из DataFrame `df`.
    - Раскодирует JSON-строки, содержащиеся в полях 'stat_train', 'stat_test', 'stat_init', и преобразует их в DataFrame.
    - Объединяет три DataFrame для каждой метрики (precision, recall, f1-score) и каждого источника данных (train, test, init).
    - Возвращает итоговый DataFrame с многоуровневыми индексами для колонок, представляющими метрики и источники данных.

    Пример:
    -------
    >>> self.get_statistic_by_id(df, record_id=5)
                 precision                   recall                     f1-score               
                 stat_train stat_test stat_init stat_train stat_test stat_init stat_train stat_test stat_init
    Neutral          0.969     0.906     0.955      0.997     0.977     0.993     0.983     0.940     0.974
    Open             0.993     0.962     0.987      0.966     0.894     0.952     0.980     0.927     0.969
    Pistol           0.997     0.944     0.987      0.924     0.788     0.897     0.959     0.859     0.940
    Thumb            0.997     0.946     0.987      0.949     0.791     0.917     0.972     0.861     0.951
    OK               0.995     0.947     0.986      0.941     0.825     0.918     0.967     0.882     0.950
    Grab             0.994     0.961     0.988      0.957     0.854     0.937     0.975     0.904     0.962
    """    
    records = df.loc[record_id,:]

    stat_train = pd.DataFrame(json.loads(records['stat_train']))
    stat_test = pd.DataFrame(json.loads(records['stat_test']))
    stat_init = pd.DataFrame(json.loads(records['stat_init']))

    dfs = [stat_train,stat_test,stat_init]
    # подготовка признаков для обьединения, выбираем поочередно по признаку от каждой метрики
    df_concat = pd.concat([df_.iloc[:, i] for i in range(3) for df_ in dfs], axis=1)

    index = ['Neutral', 'Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    columns = pd.MultiIndex.from_product([['precision', 'recall', 'f1-score'],['stat_train', 'stat_test', 'stat_init']])

    return pd.DataFrame(df_concat.values, index = index, columns=columns)
  