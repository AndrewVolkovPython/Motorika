import proshestic_processor.proshestic_graphs as pg
import proshestic_processor.proshestic_model as pm
import proshestic_processor.proshestic_statistic as ps


import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import norm



from sklearn.preprocessing import LabelEncoder

class ProstheticHandDataProcessor:
    
    # graps
    plot_sensors = pg.plot_sensors
    plotly_sensor = pg.plotly_sensor
    graph_sensor_gestures = pg.graph_sensor_gestures
    plot_results = pg.plot_results
    
    # modeling
    model_data_prepare = pm.model_data_prepare
    
    # statistics
    save_stat = ps.save_stat
    read_stat = ps.read_stat
    get_stat = ps.get_stat
    get_params = ps.get_params
    get_statistic = ps.get_statistic
    get_min_max_stat_for_pilote = ps.get_min_max_stat_for_pilote
    get_model_params_by_id = ps.get_model_params_by_id
    get_params_by_id = ps.get_params_by_id
    get_statistic_by_id = ps.get_statistic_by_id
    
    def __init__(self, 
                 files: str | list, 
                 n_omg_channels: int, 
                 pilote_id:int = 0,
                 n_acc_channels: int = 0, 
                 n_gyr_channels: int = 0, 
                 n_mag_channels: int = 0, 
                 n_enc_channels: int = 0,
                 button_ch: bool = True, 
                 sync_ch: bool = True, 
                 timestamp_ch: bool = True,
                 verbose: bool = True):
        """
        Инициализация процессора данных.
        Принимает один файл или список файлов, проверяет их наличие.
        
        :param files: строка (один файл) или список строк (несколько файлов)
        """
        # Проверка, является ли входным параметром строка (один файл) или список файлов
        if isinstance(files, str):
            self.files = [files]  # Преобразуем в список, если передана строка
        elif isinstance(files, list):
            self.files = files
        else:
            raise TypeError("Ожидается строка (имя файла) или список строк (список файлов).")
        
        # Проверка существования каждого файла
        self.__check_files_exist()
        
        self.n_omg_channels = n_omg_channels
        self.n_acc_channels = n_acc_channels
        self.n_gyr_channels = n_gyr_channels
        self.n_mag_channels = n_mag_channels
        self.n_enc_channels = n_enc_channels
        self.button_ch = button_ch
        self.sync_ch = sync_ch
        self.timestamp_ch = timestamp_ch
        
        self.pilote_id = pilote_id
        
        self.cosine = 0
        self.cosine_add_feature_1 = 0
        self.use_add = False
        self.sensor_power = 25
        
        self.shift = (0,0)
        
        self.stat_train = {}
        self.stat_test = {}
        self.stat_init = {}
        
        
        self.vectors = {}
        self.vectors_sum = {}
        self.sensor_mean = {}
        self.CLEAN_SENSORS = []
        self.CLEAN_SENSORS_ADD_FEATURE_1 = []
        self.CLEAN_SENSORS_FINAL = []
        
        self.vectors_final = {}
        self.vectors_neutral = {}
        self.vector_median_length = 0
        
        self.min_max_sensor = {}
        
        self.median_vectors_all = {}
        self.median_vectors_diff = {}
        
                
        self.median_signal_power = pd.DataFrame()
        self.additional_features_1 = pd.DataFrame()
        self.changes_df_info = pd.DataFrame()
        
        self.additional_features_1_vector = {}
        self.additional_features_1_cosine = {}
        self.additional_features_1_cosine_mean = {}
        
        
        self._verbose = verbose
        
        
        self.GESTURES = ['Neutral', 'Open', 'Pistol', 'Thumb', 'OK', 'Grab']
        self.OMG_CH = [str(i) for i in range(n_omg_channels)]
        
        self.__process()
        
    @property
    def verbose(self):
        """Getter для verbose."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """Setter для verbose."""
        if not isinstance(value, bool):
            raise ValueError("verbose должен быть булевым значением.")
        self._verbose = value      
        
    def __log(self, message):
        """Вспомогательный метод для вывода сообщений, если verbose=True."""
        if self.verbose:
            if message == 'sep':
                print('-'*100)
            else:    
                print(message)              
        
    def __check_files_exist(self):
        """Проверяет, существуют ли файлы, указанные в списке."""
        for file in self.files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"Файл {file} не найден.")
        for file in self.files:
            if not os.path.isfile(f'{file}.protocol.csv'):
                raise FileNotFoundError(f"Файл {file}.protocol.csv не найден.")
   
    def __process(self):
        self.gestures = pd.DataFrame()
        self.__log(f"{'Всего файлов:':<40} {len(self.files)}")
        for file in self.files:
            self.__log(f"{'Чтение файла значений:':<40} {file}")
            df_data = self.__read_omg_csv(file)
            
            self.__log(f"{'Чтение файла протокола жестов:':<40} {file}")
            df_gestures_protocol = self.__gestures_getter(file)
        
            self.__log(f"{'Количество записей:':<40} {df_data.shape[0]}")
        
            df_data['step'] =  (df_data['SYNC'] != df_data['SYNC'].shift(1)).cumsum()
            df_data = pd.merge(df_data, df_gestures_protocol, how = 'inner', left_on='SYNC', right_index=True)
               
            df_summary =df_data[df_data.step.diff() != 0][['step','gesture']].reset_index().rename(columns={'index':'start_step'})
            slices = self.__get_first_last_slices(df_summary)
            
            self.__log(f"{'Срезаем с и до:':<40} {slices[0]} <> {slices[1]}")
            
            df_data = df_data.iloc[slices[0]:slices[1]]
            
            self.gestures =  pd.concat([self.gestures, df_data], axis = 0, ignore_index=True)
            
            self.__log('sep')    
        self.gestures['step']  = (self.gestures['SYNC'] != self.gestures['SYNC'].shift(1)).cumsum()
        self.gestures = self.gestures.set_index('step')
        self.__log(f"{'Итоговый размер датасета:':<40} {self.gestures.shape}")
        

    def __get_first_last_slices(self,df: pd.DataFrame) -> (tuple):
        for row in df.itertuples(index=True):
            if row.Index == 0:
                continue  
            if row.Index == len(df) - 1:
                break
            if row.gesture != 0:
                start_slice = row.start_step
                break
            last_slice = df[-1:]['start_step'].values[0]
        del df
        return (start_slice, last_slice)
        
        
            
    def __read_omg_csv(self, path_palm_data: str) -> pd.DataFrame:

        df_raw = pd.read_csv(path_palm_data, sep=' ', 
                            header=None, 
                            skipfooter=1, 
                            skiprows=1, 
                            engine='python')
        
        columns = np.arange(self.n_omg_channels).astype('str').tolist()
        
        for label, label_count in zip(['ACC', 'GYR', 'MAG', 'ENC'], 
                                    [self.n_acc_channels, self.n_gyr_channels, \
                                     self.n_mag_channels, self.n_enc_channels]):
            columns = columns + ['{}{}'.format(label, i) for i in range(label_count)]
            
        if self.button_ch:
            columns = columns + ['BUTTON']
            
        if self.sync_ch:
            columns = columns + ['SYNC']
            
        if self.timestamp_ch:
            columns = columns + ['ts']
            
        try:
            if (real_cols:=len(df_raw.columns)) != (asked_rows:=len(columns)):
                raise ValueError(f"Некорректное количество признаков: Реально в файле данных {real_cols} против {asked_rows},\n указанных через параметры при инициализации обьекта")
            df_raw.columns = columns

        except Exception as e:
            print(f"Произошла ошибка: {e}")        
        
        return df_raw    
    
    def __gestures_getter(self, palm_file) ->pd.DataFrame:
        
        gestures_protocol = pd.read_csv(f'{palm_file}.protocol.csv', index_col=0)
        
        le = LabelEncoder()

        # FIT
        le.fit(
            gestures_protocol[[
                "Thumb","Index","Middle","Ring","Pinky",
                'Thumb_stretch','Index_stretch','Middle_stretch','Ring_stretch','Pinky_stretch'
            ]]
            .apply(lambda row: str(tuple(row)), axis=1)
        )

        # TRANSFORM
        gestures_protocol['gesture'] = le.transform(
            gestures_protocol[[
                "Thumb","Index","Middle","Ring","Pinky",
                'Thumb_stretch','Index_stretch','Middle_stretch','Ring_stretch','Pinky_stretch'
            ]]
            .apply(lambda row: str(tuple(row)), axis=1)
        )
        
        return gestures_protocol['gesture']       
    
       
    def __steps_getter(self, gesture):
        '''
        Получаем список шагов для указанного и жеста
        Т.е например. У нас в датасете есть жест 3, и мы хотим получить список шагов, где он присутствует. например Шаги [8, 12,34]
        
        '''
        # return self.gestures[self.gestures.gesture == gesture][[sensor,'step']].step.unique()
        return self.gestures[self.gestures.gesture == gesture].index.unique().tolist()

    def __vector_getter(self, sensor, steps):
        '''
        Возращаем векторы в словаре для указанных шагов
        '''
        v = {}
        for step in steps:
            v[step] = self.gestures.loc[step,sensor].to_numpy().T
        return v

    def calc_mean_cos_for_vector(self):
        sensor_mean = {}
        for sensor in range(0,self.n_omg_channels):
            vector_sum = np.zeros(5)
            for idx, gesture in enumerate(range(1,6)):
                arr_for_inter = self.vectors[sensor][gesture].values()
                min_length, max_length = min(len(el) for el in arr_for_inter), max(len(el) for el in arr_for_inter)
                if min_length != max_length:
                    arr_for_inter = [self.interpolate_vector(el, max_length) if len(el) != max_length else el for el in arr_for_inter]
                arr = np.array(list(arr_for_inter))
                dot_product = np.dot(arr, arr.T)
                norm = np.linalg.norm(arr, axis=1)
                norm_matrix = np.outer(norm, norm)
                cosine_similarity = dot_product / norm_matrix
                vector_sum[idx] = cosine_similarity.mean()
            sensor_mean[sensor] = vector_sum.mean()
        return sensor_mean


    def vectors_cousine(self):
        '''
            Расчитываем косинусную схожесть путем пересчета ее для всех векторов одинакового шага и жеста, и получаем сумму их
            Т.е например 1 шаг это жест 2, там 30 векторов. Расчет будет произведен каждого вектора со всеми другими в этом списке. Т.е. 30 x 30 = 900 значений.
            
            vectors_all - все вектора, не включая нейтральную позицию
            vectors_sum_all - суммы косинусов между текущим векторам и всеми остальными в данном канале по каждому жесту канал, жест, шаги
        '''    
        vectors_sum_all = {}
        vectors_all = {}
        
        self.__log(f"{'Создание словаря с векторами в формате [сенсор][жест][шаг][значения], доступен через <object>.vectors':<40} ")

        for column in range(0,self.n_omg_channels):
            vectors_sum_all[column] = {}
            vectors_all[column] = {}

            for gesture in range(1,6):
                steps_ = self.__steps_getter(gesture)
                vectors_= self.__vector_getter(str(column), steps_)
                vectors_all[column][gesture] = vectors_

                
        self.vectors =  vectors_all
        self.__log(f"{'Расчет косинусного сходства между векторами, сгруппированными по сенсорам, доступны через <object>.vectors_sum'} ")
        self.vectors_sum = self.calc_mean_cos_for_vector()    

    def interpolate_vector(self, vector, new_length):
        """
        Функция для линейной интерполяции вектора до новой длины.

        Параметры:
        vector (numpy.ndarray): Исходный вектор.
        new_length (int): Новая длина вектора.

        Возвращает:
        numpy.ndarray: Интерполированный вектор с новой длиной.
        """    
        x = np.linspace(0, 1, len(vector))
        f = interp1d(x, vector, kind='linear')
        new_x = np.linspace(0, 1, new_length)
        return f(new_x)    


    def cosine_similarity_calc(self, vector, initial_vector, vector_length ):
        ''' 
        Вычисляем косинусное сходство между двумя векторами
        Если длины векторов не соотвествуют заданной длине, то интерполируем вектора до требуемого размера
        '''

        if len(initial_vector) != vector_length:
            initial_vector = self.interpolate_vector(initial_vector, vector_length)

        if len(vector) != vector_length:
            vector = self.interpolate_vector(vector, vector_length)
            
        return np.dot(vector,initial_vector)/ (norm(vector)*norm(initial_vector))             


    def clean_sensor_getter_by_cosine(self, treshold:float):
        self.cosine = treshold
        # рассчитаем среднее значение по все векторам жестов по сенсору
        # Идея в этом, что если у нас есть повторяющийся паттерн для жеста, то и косинусное сходство высокое. Если оно низкое, то у нас наблюдается высокий шум, и такие данные надо выброить из измерения, так как они не дают нам четкой картины
        self.CLEAN_SENSORS = list({str(k): v for k, v in self.vectors_sum.items() if v > treshold}.keys())            

    def average_dict(self, dict_):    
        '''
        Расчитываем  среднее для значений в словаре
        '''
        return sum(dict_.values())/len(dict_)
    
        
    def median_sensor_power_getter(self):
        # Проверим медианные занчения по сенсорам из ранее выбранных признаков
        df_median  = pd.DataFrame(self.gestures[self.CLEAN_SENSORS].median(), columns=['median_value']).rename_axis('sensor_id')
        # Чтобы грубо автоматизировать процесс выбор сильных сигналов
        max_sensor = df_median.max().iloc[0]
        print(f'Максимальная сила сигнала: {max_sensor}')
        # Рассчитаем процент силы сигнала для каждого сенсора, относительно максимального
        df_median['percent'] = (df_median['median_value']/max_sensor * 100).round(2)
        self.median_signal_power = df_median 
        
    def  clean_sensor_getter_by_power(self, treshold: int):
         self.sensor_power = treshold
         self.__log(f"{'Выбор сенсоров с сильным сигналом, доступны через <object>.CLEAN_SENSORS_FINAL'} ")
         self.CLEAN_SENSORS_FINAL = list(self.median_signal_power[self.median_signal_power.percent > treshold].percent.index)  
         return self.CLEAN_SENSORS_FINAL

    def get_vectors_final(self):
        self.__log(f"{'Создание словаря с векторами только для выбранных сенсоров в формате [сенсор][жест][шаг][значения],\n доступны через <object>.vectors_final'} ")
        self.vectors_final = {key: self.vectors[key] for key in [int(id) for id in self.CLEAN_SENSORS_FINAL] if key in self.vectors}

    def get_df_final(self):
        self.__log(f"{'Создание pd.DataFrame для выбранных сенсоров,\n доступны через <object>.gestures_clean\n'} ")
        self.gestures_clean = self.gestures[self.CLEAN_SENSORS_FINAL + ['gesture']]
    
    def  get_neutral_vectors(self):
        # Соберем все нейтральные вектора
        self.__log(f"{'Создание словаря с векторами для нейтроального положения, только для выбранных сенсоров\nв формате [сенсор][жест][шаг][значения],доступны через <object>.vectors_neutral\n'} ")
        for column in self.CLEAN_SENSORS_FINAL:
            self.vectors_neutral[column] = {}
            for gesture in range(0,1):
                steps_ = self.__steps_getter(gesture)
                vectors_= self.__vector_getter(str(column), steps_)
                self.vectors_neutral[column][gesture] = vectors_       
    
    def get_vector_median_length(self):
        self.__log(f"{'Получаем медианную длину вектора для нейтрального положения, доступны через <object>.vector_median_length\n'} ")
        lengths = [len(v) for v in self.vectors_neutral[self.CLEAN_SENSORS_FINAL[0]][0].values()]
        self.vector_median_length = int(np.median(lengths))

    def get_median_vectors(self):
        self.__log(f"Получаем словари с медианными длинами векторов, дисперсией")
        self.__log(f"формат [sensor][жесть][медианный вектор]")
        self.__log(f"Доступны через <object>.median_vectors_all, <object>.median_vectors_diff\n")
        for column in self.CLEAN_SENSORS_FINAL:
            # Интерполируем все массивы до одной длины
            interpolated_arrays = []
            self.median_vectors_all[column] = {}
            self.median_vectors_diff[column] = {}
            
            for array in self.vectors_neutral[column][0].values():
                x = np.linspace(0, 1, len(array))
                f = interp1d(x, array, kind='linear', fill_value='extrapolate')
                new_x = np.linspace(0, 1, self.vector_median_length)
                interpolated_arrays.append(f(new_x))

            matrix = np.stack(interpolated_arrays)

            # Вычисляем медианный вектор
            median_vector = np.median(matrix, axis=0)
            self.median_vectors_all[column][0] = median_vector
            self.median_vectors_diff[column][0] = np.diff(median_vector)
            
            
            for gesture in range(1,6):
                for array in self.vectors_final[int(column)][gesture].values():
                    x = np.linspace(0, 1, len(array))
                    f = interp1d(x, array, kind='linear', fill_value='extrapolate')
                    new_x = np.linspace(0, 1, self.vector_median_length)
                    interpolated_arrays.append(f(new_x))

                matrix = np.stack(interpolated_arrays)

                # Вычисляем медианный вектор
                median_vector = np.median(matrix, axis=0)
                self.median_vectors_all[column][gesture] = median_vector
                self.median_vectors_diff[column][gesture] = np.diff(median_vector)          

    def correct_vector(self, new_vector, median_vector, threshold=0.2):
        corrected_vector = []
        corrected_vector_idx = []
        median_vector = self.interpolate_vector(median_vector,len(new_vector))
        for i, v in enumerate(new_vector):
            expected_value = median_vector[i]
            min_value = expected_value * (1 - threshold)  # -20%
            max_value = expected_value * (1 + threshold)  # +20%
            
            if v < min_value or v > max_value:
                corrected_vector.append(expected_value)  # Корректируем
                corrected_vector_idx.append(True)
            else:
                corrected_vector.append(v)  # Оставляем как есть
                corrected_vector_idx.append(False)
        return np.array(corrected_vector), corrected_vector_idx                
    
    def correct_vector_accordingly(self,vector, vector_idx, vector_diff):
        vector_ = vector.copy()
        vector_diff = self.interpolate_vector(vector_diff, len(vector_))
        flag_ = False
        to_change = []
        for idx, cor_vec in enumerate(vector_idx):
            if (cor_vec): 
                if idx == len(vector_idx) - 1:
                        to_change.append(idx)
                        for el in to_change:
                            vector_[el] = vector_[el-1]+vector_diff[el-1]
                else:
                    if (flag_== False): 
                            to_change.append(idx)
                            flag_ = True
                            continue
                    if flag_:
                        to_change.append(idx)
                        continue
            if (cor_vec==False) & (flag_== True):
                flag_ = False
                for el in to_change[::-1]:
                    vector_[el] = vector_[el+1]+vector_diff[el+1]
                to_change = [] 
        return vector_ 

    def vector_corretion_assembly(self, vector, median_vector, median_vector_diff):
        
        corrected_vector, corrected_vector_idx = self.correct_vector(vector, median_vector)
        return self.correct_vector_accordingly(corrected_vector, corrected_vector_idx, median_vector_diff)      
    
    def get_min_max_for_senser(self):
        self.__log(f"{'Получаем мин макс показатели сигнала для каждого сенсора на основе медианного вектора, в формате [sensor][жест]\nдоступны через <object>.min_max_sensor\n'} ")        
        for col in self.CLEAN_SENSORS_FINAL:
            self.min_max_sensor[col] = {}
            for gesture in range(0,6):
                self.min_max_sensor[col][gesture] = {'min': min(self.median_vectors_all[col][gesture]), 'max': max(self.median_vectors_all[col][gesture]) }

    
    def vector_correction_all(self, treshold = 1.5):
        for sensor in self.CLEAN_SENSORS_FINAL:
            for idx, item in self.vectors_neutral[sensor][0].items():
                outliers = [x for x in item if x > (self.min_max_sensor[sensor][0]['max'])*treshold or x < (self.min_max_sensor[sensor][0]['min'])/treshold]
                if outliers:
                    self.gestures_clean.loc[idx, sensor] = self.vector_corretion_assembly(self.vectors_neutral[sensor][0][idx], self.median_vectors_all[sensor][0],self.median_vectors_diff[sensor][0]).astype('int64')
            for gesture in range(1,6):
                for idx, item in self.vectors_final[int(sensor)][gesture].items():
                    outliers = [x for x in item if x > (self.min_max_sensor[sensor][gesture]['max'])*treshold or x < (self.min_max_sensor[sensor][gesture]['min'])/treshold]
                    if outliers:
                        self.gestures_clean.loc[idx, sensor] = self.vector_corretion_assembly(self.vectors_final[int(sensor)][gesture][idx], self.median_vectors_all[sensor][gesture],self.median_vectors_diff[sensor][gesture]).astype('int64')      
                        
    def create_additional_features_1(self):
        self.__log(f"{'Создадим дополнительные признаки, путем ссумирования подряд по 5 сенсоров\nДоступны через <object>.additional_features_1\n'} ")        
        for i in range(0, self.n_omg_channels, 5):
            self.additional_features_1[f'new_feature_{i//5 + 1}'] = self.gestures.iloc[:, i:i+5].sum(axis=1)

    def create_additional_features_1_vector(self):
        self.__log(f"{'Создадим векторы на базе additional_features_1 и интерполируем их\nДоступны через <object>.additional_features_1_vector\n'} ")        
        groups = self.additional_features_1.groupby(self.additional_features_1.index)
        group_sizes = groups.size()
        median_size = int(group_sizes.median())

        v = {col: {} for col in groups.obj.columns}
        for step, group in groups:
            for column in group.columns:
                v[column][step] = self.interpolate_vector(group.loc[:,column].to_numpy().T, median_size)
        self.additional_features_1_vector = v

    def calculate_additional_features_1_cosine(self):
        self.__log(f"{'Рассчитаем косинусную схожесть на векторе additional_features_1_vector\nДоступны через <object>.additional_features_1_cosine_mean\n'} ")        
        sensor_mean = {}
        for feature in self.additional_features_1.columns:
            arr_for_inter = self.additional_features_1_vector[feature].values()
            arr = np.array(list(arr_for_inter))
            dot_product = np.dot(arr, arr.T)
            norm = np.linalg.norm(arr, axis=1)
            norm_matrix = np.outer(norm, norm)
            cosine_similarity = dot_product / norm_matrix
            sensor_mean[feature] = cosine_similarity.mean()        
        self.additional_features_1_cosine_mean = sensor_mean

    def clean_sensor_getter_by_cosine_add_feat_1(self, treshold:float):
        self.__log(f"{'Выберем признаки из additional_features_1 согласно cosine\nДоступны через <object>.CLEAN_SENSORS_ADD_FEATURE_1\n'} ")        
        self.cosine_add_feature_1 = treshold
        self.CLEAN_SENSORS_ADD_FEATURE_1 = list({key: val for key, val in self.additional_features_1_cosine_mean.items() if val > treshold}.keys())   

    def cleaned_df_diff_adding(self):
        diff_ = self.gestures_clean[self.CLEAN_SENSORS_FINAL].diff().abs().copy()
        self.gestures_clean.loc[:,'diff'] = diff_.sum(axis=1).values

    def create_fix_gestures_df(self, shift = (0,0)):
        self.shift = shift
        
        if 'level_0' in self.gestures_clean.columns:
            self.gestures_clean.drop(columns=['level_0'], inplace=True)
    
        step_changes = self.gestures_clean.reset_index()
        step_changes = step_changes[step_changes['step'].diff() != 0]

        self.__log(f"{'Создаем таблицу смещения\nДоступны через <object>.changes_df_info\n'} ")
        self.changes_df_info = step_changes[['step', 'gesture']].copy()
        self.changes_df_info['index'] = step_changes.index

        self.changes_df_info = self.changes_df_info[['step', 'index', 'gesture']]

        self.changes_df_info['step_length'] = np.array(pd.concat([self.changes_df_info['index'].diff().iloc[1:],pd.Series([0])], axis=0).reset_index().drop(columns='index')[0])
        self.changes_df_info = self.changes_df_info.reset_index().drop(columns='level_0').rename(columns={'index':'start_position'})
        
        self.changes_df_info['fact_start'] = 0
        self.changes_df_info['fact_finish'] = 0     
        
        shift_non_zero_start = shift[0]
        shift_non_zero_finish = shift[1]
        
        self.gestures_clean = self.gestures_clean.reset_index()

        for row in self.changes_df_info.itertuples(index=True):
            if row.Index == len(self.changes_df_info) - 1:
                break
            if row.gesture != 0:
                self.changes_df_info.loc[int(row.Index),'fact_start'] =(fact_start:=self.gestures_clean.loc[row.start_position:row.start_position + row.step_length//2+(row.step_length//5), 'diff'].idxmax() + shift_non_zero_start)
                fact_finish = self.gestures_clean.loc[fact_start + row.step_length - (row.step_length//5):fact_start + row.step_length + (row.step_length//5), 'diff'].idxmax()
                self.changes_df_info.loc[int(row.Index),'fact_finish'] = fact_finish + shift_non_zero_finish
        self.gestures_clean.loc[:,'gesture'] = 0
        
        self.__log(f"{'Делаем смещение начало и конца жеста, согласно changes_df_info'} ")
        for row in self.changes_df_info.itertuples(index=True):
            if row.gesture != 0:
                self.gestures_clean.loc[row.fact_start:row.fact_finish,'gesture'] = row.gesture                
    

    def change_params(self, cosine, sensor_power, cosine_add_feature_1, shift):
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