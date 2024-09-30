import os
import json
import pandas as pd

from sklearn.metrics import classification_report

def save_stat(self, file_path, new_object):
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
    
def read_stat(self,file_path):
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


def get_stat(self, file_path):
    df = pd.read_json(file_path)

    df['stat_train'] = df['stat_train'].apply(json.loads)
    df['stat_test'] = df['stat_test'].apply(json.loads)
    df['stat_init'] = df['stat_init'].apply(json.loads)

    stat_train_df = pd.json_normalize(df['stat_train'])
    stat_test_df = pd.json_normalize(df['stat_test'])
    stat_init_df = pd.json_normalize(df['stat_init'])

    result_df = pd.concat([df.drop(columns=['stat_train','stat_test','stat_init']), stat_train_df], axis=1)
    result_df = pd.concat([df.drop(columns=['stat_train','stat_test','stat_init']), stat_test_df], axis=1)
    result_df = pd.concat([df.drop(columns=['stat_train','stat_test','stat_init']), stat_init_df], axis=1)

    result_df['features_num'] = result_df.clean_sensor_final.apply(lambda x: len(x))
    result_df['features_add_1_num'] = result_df.clean_sensors_add_feature_1.apply(lambda x: len(x)) 
    
    return result_df    

def get_params(self):
    return {
        'pilote_id' : self.pilote_id,
        'cosine' : self.cosine,
        'sensor_power' : self.sensor_power,
        'cosine_add_feature_1' : self.cosine_add_feature_1,
        'clean_sensor_final' : self.CLEAN_SENSORS_FINAL,
        'clean_sensors_add_feature_1' : self.CLEAN_SENSORS_ADD_FEATURE_1,
        'use_add' : self.use_add,
        'stat_train' : self.stat_train,            
        'stat_test' : self.stat_test,            
        'stat_init' : self.stat_init, 
        'shift' : self.shift,           
    }

def get_statistic(self, train, pred, type, show = True):
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
        
def get_min_max_stat_for_pilote(self, df, pilote_id):
    stat_mean = pd.DataFrame({
    'pecision_mean' :df[df.pilote_id == pilote_id].filter(regex='^(precision)').mean(axis = 1)
    ,'recall_mean' : df[df.pilote_id == pilote_id].filter(regex='^(recall)').mean(axis = 1)
    ,'f1-score_mean' :df[df.pilote_id == pilote_id].filter(regex='^(f1)').mean(axis = 1)
    })    
    
    return {
        'min': stat_mean.mean(axis = 1).min(),
        'min_id': stat_mean.mean(axis = 1).idxmin(),
        'max': stat_mean.mean(axis = 1).max(),
        'max_id': stat_mean.mean(axis = 1).idxmax()
        }

def get_model_params_by_id(self, df, record_id):
    return df.loc[record_id, 'rf_params']    

def get_params_by_id(self, df, id):
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
def get_statistic_by_id(self, df, record_id):
    records = df.loc[record_id,:]

    stat_train = pd.DataFrame(json.loads(records['stat_train']))
    stat_test = pd.DataFrame(json.loads(records['stat_test']))
    stat_init = pd.DataFrame(json.loads(records['stat_init']))

    dfs = [stat_train,stat_test,stat_init]

    df_concat = pd.concat([df_.iloc[:, i] for i in range(3) for df_ in dfs], axis=1)

    index = ['Neutral', 'Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    columns = pd.MultiIndex.from_product([['precision', 'recall', 'f1-score'],['stat_train', 'stat_test', 'stat_init']])

    return pd.DataFrame(df_concat.values, index = index, columns=columns)
  