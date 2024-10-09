import proshestic_processor.proshestic_graphs as pg
import proshestic_processor.proshestic_model as pm
import proshestic_processor.proshestic_statistic as ps
import proshestic_processor.prosthetic_hand_data_processing as pr
import proshestic_processor.proshestic_inference as pi


import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import norm



from sklearn.preprocessing import LabelEncoder

class ProstheticHandDataProcessor:
    """
    Класс для обработки данных протезов рук, включая построение графиков,
    моделирование, статистику и анализ данных.
    """
    
    # graphs
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
    
    # processing
    
    change_params_and_apply = pr.change_params_and_apply
    fit_and_predict = pr.fit_and_predict
    
    # inference
    
    inference = pi.inference
    
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
            Инициализация процессора данных для анализа сигналов от протезов рук.

            Принимает один файл или список файлов, проверяет их наличие и
            инициализирует параметры, необходимые для обработки данных.

            Параметры:
            ----------
            files : str | list
                Строка (один файл) или список строк (несколько файлов), 
                представляющих собой пути к файлам с данными.

            n_omg_channels : int
                Количество каналов OMG (измерений).

            pilote_id : int, по умолчанию 0
                Идентификатор пилота, для которого собираются данные.

            n_acc_channels : int, по умолчанию 0
                Количество каналов акселерометра.

            n_gyr_channels : int, по умолчанию 0
                Количество каналов гироскопа.

            n_mag_channels : int, по умолчанию 0
                Количество каналов магнитометра.

            n_enc_channels : int, по умолчанию 0
                Количество каналов энкодеров.

            button_ch : bool, по умолчанию True
                Флаг, указывающий, нужно ли учитывать данные кнопок.

            sync_ch : bool, по умолчанию True
                Флаг, указывающий, нужно ли учитывать данные синхронизации.

            timestamp_ch : bool, по умолчанию True
                Флаг, указывающий, нужно ли учитывать временные метки.

            verbose : bool, по умолчанию True
                Флаг, указывающий, выводить ли подробные сообщения о процессе.

            Исключения:
            -----------
            TypeError
                Если параметр files не является строкой или списком строк.
            
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
        
        # id пилота
        self.pilote_id = pilote_id
        
        # нижняя граница косинусного сходства для основных каналов
        self.cosine = 0
        # нижняя граница косинусного сходства для дополнительных признаков 1
        self.cosine_add_feature_1 = 0
        # использовать дополнительные признаки
        self.use_add = False
        # процентное отклонение от максимального уровня сигнала, для отбрасывания слабых сигналов
        self.sensor_power = 25
        
        # тип модели rf или lgb
        self.model_name = ''
        
        # random state
        self.random_state = 0
        
        # процентное соотношение при делении на трнировочную и тестовую выборки
        self.test_size_split = 0
        
        # расширение или сужение для расчитанного диапазона момента жеста
        self.shift = (0,0)
        
        # статистика для тренировочной выборки
        self.stat_train = {}
        # статистика для тестовой выборки
        self.stat_test = {}
        # статистика для всей выборки
        self.stat_init = {}
        
        # вектора всех жестов
        self.vectors = {}
        # сумма косинусного сходства по сенсорам
        self.vectors_sum = {}
        # признаки отобранные на основании косинусного сходства
        self.CLEAN_SENSORS = []
        # признаки отобранные на основании косинусного сходства для дополнительного признака 1
        self.CLEAN_SENSORS_ADD_FEATURE_1 = []
        # признаки отобранные на основании косинусного сходства и силы сигнала
        self.CLEAN_SENSORS_FINAL = []
        
        # вектора всех жестов, только для сенсоров из CLEAN_SENSORS_FINAL
        self.vectors_final = {}
        # вектора для нейтрального положения
        self.vectors_neutral = {}
        # медианная длина вектора, используетя для интерполяции при расчете косинусного сходства
        self.vector_median_length = 0
        
        # min max значения для сенсоров
        self.min_max_sensor = {}
        
        # медианные вектора для сенсоров
        self.median_vectors_all = {}
        # пошаговое изменение значений в медианных векторах для сенсоров
        self.median_vectors_diff = {}
        
                
        self.median_signal_power = pd.DataFrame()
        self.additional_features_1 = pd.DataFrame()
        self.changes_df_info = pd.DataFrame()
        
        self.additional_features_1_vector = {}
        self.additional_features_1_cosine = {}
        self.additional_features_1_cosine_mean = {}
        
        # Использовать вывод дополнительной инфорации при работе модели
        self._verbose = verbose
        
        # список жестов
        self.GESTURES = ['Neutral', 'Open', 'Pistol', 'Thumb', 'OK', 'Grab']
        # список изначальных сенсоров
        self.OMG_CH = [str(i) for i in range(n_omg_channels)]
        
        # чтение данных и первичная обрабботка
        self.__process()
        
    @property
    def verbose(self):
        """
        Возвращает значение флага подробного вывода.

        Этот метод позволяет узнать, активирован ли режим
        подробного вывода в процессе обработки данных.

        Returns
        -------
        bool
            True, если режим подробного вывода включен; 
            False в противном случае.
        """        
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """
        Устанавливает значение флага подробного вывода.

        Этот метод позволяет активировать или деактивировать
        режим подробного вывода в процессе обработки данных.

        Parameters
        ----------
        value : bool
            Значение, указывающее, должен ли режим подробного вывода
            быть включен (True) или выключен (False).

        Raises
        ------
        ValueError
            Если значение не является булевым (True или False).
        """    
        if not isinstance(value, bool):
            raise ValueError("verbose должен быть булевым значением.")
        self._verbose = value      
        
    def __log(self, message):
        """Вспомогательный метод для вывода сообщений, если verbose=True."""
        if self.verbose:
            if message == 'sep':
                # Вывести линию сепаратор
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
        """
        Обрабатывает файлы и собирает данные жестов в один DataFrame.

        Этот метод читает данные из CSV-файлов, соответствующих жестам, и 
        объединяет их в один DataFrame. Он также создает столбец 'step',
        который представляет собой нарастающую сумму изменений в столбце 'SYNC'.

        Процесс включает следующие шаги:
        1. Чтение данных из каждого файла.
        2. Чтение протокола жестов.
        3. Слияние данных по столбцу 'SYNC'.
        4. Определение шагов, соответствующих различным жестам.
        5. Удаление лишних записей с помощью обрезки повторяющихся нейтральных значений, 
            до момента начала движения сначала датасета и обрезки датачета на последнем не нейтральном жесте.
        6. Конкатенация данных в общий DataFrame.

        Логирует ключевые этапы обработки и размер итогового набора данных.

        Raises
        ------
        ValueError
            Если не удается прочитать файл или если возникают проблемы
            с форматированием данных.
        """        
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
        """
        Получает первые и последние индексы для среза данных.

        Параметры
        ----------
        df : pd.DataFrame
            DataFrame, содержащий информацию о жестах, с колонками 
            'gesture' и 'start_step'.

        Возвращает
        -------
        tuple
            Кортеж из двух значений: индекса начала и индекса конца
            для среза данных.

        Raises
        ------
        ValueError
            Если DataFrame не содержит жестов, отличных от 0.
        """        
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
        
        """
        Читает данные из CSV файла, содержащего информацию о сенсорах.

        Этот метод загружает данные из указанного CSV файла и создает
        DataFrame с соответствующими столбцами на основе заданного 
        количества каналов для различных сенсоров (OMG, ACC, GYR, MAG, ENC).

        Параметры
        ----------
        path_palm_data : str
            Путь к CSV файлу с данными сенсоров. Файл должен быть в 
            формате, где данные разделены пробелами.

        Возвращает
        -------
        pd.DataFrame
            DataFrame, содержащий данные из CSV файла с именами столбцов, 
            соответствующими настройкам и параметрам инициализации.

        Raises
        ------
        ValueError
            Если количество столбцов в загруженном файле не совпадает с
            ожидаемым количеством, определяемым параметрами инициализации.
        """        

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
    
    def __gestures_getter(self, palm_file : str) ->pd.DataFrame:
        """
        Извлекает протокол жестов из файла и кодирует жесты в числовые значения.

        Этот метод читает CSV файл протокола жестов, используя информацию о
        различных положениях пальцев и растяжении. Он применяет 
        кодирование меток для преобразования жестов в числовые значения.

        Параметры
        ----------
        palm_file : str
            Путь к файлу протокола жестов (без расширения). Ожидается, что
            файл имеет название '{palm_file}.protocol.csv'.

        Возвращает
        -------
        pd.Series
            Серия, содержащая закодированные значения жестов.

        Raises
        ------
        FileNotFoundError
            Если файл протокола жестов не найден по указанному пути.
        ValueError
            Если в данных нет необходимых столбцов для кодирования жестов.
        """        
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
    
       
    def __steps_getter(self, gesture : pd.DataFrame) -> list:
        """
        Получает уникальные шаги для указанного жеста из набора данных.

        Данный метод фильтрует данные и возвращает список уникальных индексов
        (шагов), где указанный жест присутствует в датасете. 

        Параметры
        ----------
        gesture : pd.DataFrame
            Жест, для которого необходимо получить шаги. Ожидается, что
            это значение должно соответствовать кодированному жесту в
            столбце 'gesture' в классе `ProstheticHandDataProcessor`.

        Возвращает
        -------
        list
            Список уникальных шагов, на которых был зафиксирован указанный жест.

        Примеры
        --------
        >>> steps = self.__steps_getter(3)
        >>> print(steps)
        [8, 12, 34]
    """
        return self.gestures[self.gestures.gesture == gesture].index.unique().tolist()

    def __vector_getter(self, sensor : str, steps : list):
        """
        Возвращает векторы данных для указанных шагов и заданного сенсора.

        Данный метод создает словарь, в котором ключами являются шаги,
        а значениями — векторы данных, извлеченные из столбца сенсора
        в датафрейме жестов для каждого указанного шага.

        Параметры
        ----------
        sensor : str
            Название сенсора, для которого необходимо получить векторы.
        steps : list
            Список шагов, для которых нужно извлечь векторы.

        Возвращает
        -------
        dict
            Словарь, где ключами являются шаги, а значениями — векторы 
            данных, извлеченные для каждого шага.

        Примеры
        --------
        >>> vectors = self.__vector_getter('ACC0', [0, 1, 2])
        >>> print(vectors)
        {0: array([...]), 1: array([...]), 2: array([...]}
        """
        v = {}
        for step in steps:
            v[step] = self.gestures.loc[step,sensor].to_numpy().T
        return v

    def calc_mean_cos_for_vector(self):
        """
        Вычисляет средние косинусные схожести для векторов по всем сенсорам.

        Этот метод перебирает векторы для каждого сенсора и для каждой жеста
        (от 1 до 5), вычисляет косинусную схожесть между векторами и
        возвращает среднюю косинусную схожесть для каждого сенсора.

        Возвращает
        -------
        dict
            Словарь, где ключами являются индексы сенсоров, а значениями — 
            средние косинусные схожести для каждого сенсора.

        Примеры
        --------
        >>> mean_cosine = self.calc_mean_cos_for_vector()
        >>> print(mean_cosine)
        {0: 0.85, 1: 0.78, 2: 0.90, ...}
        """        
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
        """
        Рассчитывает косинусную схожесть для всех векторов, соответствующих одинаковому шагу и жесту.

        Этот метод создает словарь, в котором хранятся векторы, организованные по сенсорам, жестам и шагам.
        Затем вычисляется сумма косинусной схожести между каждым вектором и всеми другими векторами 
        в этом же канале для каждого жеста. Например, если для жеста 2 существует 30 векторов на шаге 1,
        будет рассчитано 30 x 30 = 900 значений косинусной схожести.

        Параметры
        ----------
        None

        Возвращает
        -------
        None
            Результаты сохраняются в атрибутах `self.vectors` и `self.vectors_sum`,
            где `self.vectors` содержит все векторы, а `self.vectors_sum` 
            содержит суммы косинусной схожести для каждого сенсора.

        Примечания
        ----------
        - Метод использует вспомогательные методы `__steps_getter` и `__vector_getter`
        для получения необходимых данных.

        Примеры
        --------
        >>> self.vectors_cousine()
        >>> print(self.vectors)
        >>> print(self.vectors_sum)
        """
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
        
        self.create_additional_features_1()
        self.create_additional_features_1_vector()
        self.calculate_additional_features_1_cosine()          

    def interpolate_vector(self, vector, new_length : int):
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
        """
        Вычисляет косинусное сходство между двумя векторами.

        Если длины векторов не соответствуют заданной длине, вектора интерполируются до требуемого размера.
        
        Параметры
        ----------
        vector : np.ndarray
            Вектор, для которого вычисляется косинусное сходство.
            
        initial_vector : np.ndarray
            Вектор, с которым сравнивается первый вектор.
            
        vector_length : int
            Желаемая длина векторов. Если длины входных векторов не соответствуют,
            они будут интерполированы до этой длины.
            
        Возвращает
        -------
        float
            Значение косинусного сходства между двумя векторами, 
            которое находится в диапазоне от -1 до 1. 
            Значение 1 указывает на полное совпадение, 
            0 — на отсутствие сходства, 
            а -1 — на полное противопоставление.
        
        Примечания
        ----------
        - Для интерполяции используется метод `interpolate_vector`, который должен 
        быть реализован в классе.
        - Векторы должны быть одномерными массивами NumPy.

        Примеры
        --------
        >>> vec1 = np.array([1, 2, 3])
        >>> vec2 = np.array([4, 5, 6])
        >>> length = 3
        >>> similarity = self.cosine_similarity_calc(vec1, vec2, length)
        >>> print(similarity)  # Выводит значение косинусного сходства
        """

        if len(initial_vector) != vector_length:
            initial_vector = self.interpolate_vector(initial_vector, vector_length)

        if len(vector) != vector_length:
            vector = self.interpolate_vector(vector, vector_length)
            
        return np.dot(vector,initial_vector)/ (norm(vector)*norm(initial_vector))             


    def clean_sensor_getter_by_cosine(self, treshold:float) -> list:
        """
        Фильтрует сенсоры на основе заданного порога косинусного сходства.

        Идея заключается в том, что если у нас есть повторяющийся паттерн для жеста, 
        то косинусное сходство будет высоким. Если оно низкое, это может указывать 
        на высокий уровень шума, и такие данные следует исключить, 
        так как они не предоставляют четкой картины.

        Параметры
        ----------
        threshold : float
            Пороговое значение косинусного сходства. 
            Сенсоры с средним косинусным сходством ниже этого значения 
            будут исключены из дальнейшего анализа.

        Примечания
        ----------
        - Метод использует атрибут `vectors_sum`, который должен содержать 
        средние значения косинусного сходства для каждого сенсора.
        - Результат фильтрации сохраняется в атрибуте `CLEAN_SENSORS`.

        Примеры
        --------
        >>> processor.clean_sensor_getter_by_cosine(0.5)
        >>> print(processor.CLEAN_SENSORS)  # Выводит список сенсоров, которые 
        # имеют косинусное сходство выше 0.5
        """
        self.cosine = treshold
        self.CLEAN_SENSORS = list({str(k): v for k, v in self.vectors_sum.items() if v > treshold}.keys())            

    def average_dict(self, dict_: dict) -> float:    
        """
        Рассчитывает среднее значение для значений в словаре.

        Этот метод принимает словарь, извлекает его значения и вычисляет 
        среднее значение.

        Параметры
        ----------
        dict_ : dict
            Словарь, значения которого будут использованы для вычисления среднего.
            Ожидается, что значения словаря являются числовыми.

        Возвращает
        ----------
        float
            Среднее значение значений в словаре. Если словарь пуст, 
            будет выброшено исключение ZeroDivisionError.

        Исключения
        ----------
        ZeroDivisionError
            Возникает, если словарь пуст, что приводит к делению на ноль.

        Примечания
        ----------
        - Убедитесь, что все значения в словаре числовые, иначе может возникнуть 
        ошибка выполнения при попытке вычислить среднее.

        Примеры
        --------
        >>> avg = average_dict({'a': 10, 'b': 20, 'c': 30})
        >>> print(avg)  # Выводит: 20.0

        >>> avg_empty = average_dict({})
        # Возникает ошибка ZeroDivisionError
        """
        return sum(dict_.values())/len(dict_)
    
        
    def median_sensor_power_getter(self) -> pd.DataFrame:
        """
        Рассчитывает медианные значения силы сигналов по выбранным сенсорам.

        Этот метод анализирует данные жестов, чтобы получить медианные значения
        для каждого сенсора из ранее выбранных признаков (каналов). Он также
        вычисляет процентное соотношение медианной силы сигнала для каждого сенсора
        относительно максимальной силы сигнала.

        Возвращает
        ----------
        pd.DataFrame
            DataFrame, содержащий медианные значения и процент силы сигнала для
            каждого сенсора, где индексом является идентификатор сенсора.

        Примечания
        ----------
        - Метод предполагает, что атрибуты `gestures` и `CLEAN_SENSORS` 
        уже инициализированы в классе и содержат необходимые данные.
        
        Примеры
        --------
        >>> processor = ProstheticHandDataProcessor(...)
        >>> processor.median_sensor_power_getter()
        Максимаальная сила сигнала: 85
        >>> print(processor.median_signal_power)
                    median_value  percent
        sensor_id                     
        0               75.50     88.24
        1               80.00     94.12
        2               60.00     70.59
        3               85.00    100.00
        """        
        # Проверим медианные занчения по сенсорам из ранее выбранных признаков
        df_median  = pd.DataFrame(self.gestures[self.CLEAN_SENSORS].median(), columns=['median_value']).rename_axis('sensor_id')
        # Чтобы грубо автоматизировать процесс выбор сильных сигналов
        max_sensor = df_median.max().iloc[0]
        print(f'Максимальная сила сигнала: {max_sensor}')
        # Рассчитаем процент силы сигнала для каждого сенсора, относительно максимального
        df_median['percent'] = (df_median['median_value']/max_sensor * 100).round(2)
        self.median_signal_power = df_median 
        
    def  clean_sensor_getter_by_power(self, treshold: int) -> list:
        """
        Выбирает сенсоры с силой сигнала, превышающей указанный порог.

        Этот метод фильтрует сенсоры на основе их медианных значений силы сигнала,
        которые были рассчитаны ранее. Сенсоры с процентом силы сигнала выше
        указанного порога добавляются в список `CLEAN_SENSORS_FINAL`.

        Параметры
        ----------
        threshold : int
            Пороговое значение (в процентах) для силы сигнала. Сенсоры с 
            процентом силы выше этого значения будут выбраны.

        Возвращает
        ----------
        list
            Список идентификаторов сенсоров, которые имеют силу сигнала выше
            указанного порога.

        Примечания
        ----------
        - Метод предполагает, что атрибут `median_signal_power` уже был
        рассчитан ранее и содержит данные о медианной силе сигнала для 
        сенсоров.
        
        Примеры
        --------
        >>> processor = ProstheticHandDataProcessor(...)
        >>> processor.median_sensor_power_getter()  # Вычисление медианных значений
        >>> strong_sensors = processor.clean_sensor_getter_by_power(75)
        >>> print(strong_sensors)
        [0, 1, 3]  # Идентификаторы сенсоров с силой сигнала выше 75%
        """        
        self.sensor_power = treshold
        self.__log(f"{'Выбор сенсоров с сильным сигналом, доступны через <object>.CLEAN_SENSORS_FINAL'} ")
        self.CLEAN_SENSORS_FINAL = list(self.median_signal_power[self.median_signal_power.percent > treshold].percent.index)  
        return self.CLEAN_SENSORS_FINAL
    
    def get_cosine_steps(self, col_numbers : int = 5):
            
            df_median  = pd.DataFrame(self.gestures[self.OMG_CH].median(), columns=['median_value']).rename_axis('sensor_id')
            # Чтобы грубо автоматизировать процесс выбор сильных сигналов
            max_sensor = df_median.max().iloc[0]

            # # Рассчитаем процент силы сигнала для каждого сенсора, относительно максимального
            df_median['percent'] = (df_median['median_value']/max_sensor * 100).round(2)
            # self.median_signal_power = df_median 
            sensor_power_list = df_median[df_median.percent > self.sensor_power].index.to_list()
            params = sorted({key: val for key, val in self.vectors_sum.items() if str(key) in sensor_power_list}.values())
            return params[-(col_numbers + 1): -1]
    

    def get_vectors_final(self) -> list:
        """
        Создает словарь с векторами только для выбранных сенсоров.

        Этот метод формирует словарь, содержащий векторы только для сенсоров,
        которые были отобраны ранее, на основе силы сигнала. Векторы
        организованы в формате [сенсор][жест][шаг][значения].

        Возвращает
        ----------
        list
            Список идентификаторов сенсоров с их соответствующими векторами,
            доступными через атрибут `vectors_final`.

        Примечания
        ----------
        - Метод предполагает, что атрибут `CLEAN_SENSORS_FINAL` уже был
        установлен ранее и содержит идентификаторы сенсоров, которые были
        выбраны на основе медианной силы сигнала.

        Примеры
        --------
        >>> processor = ProstheticHandDataProcessor(...)
        >>> processor.clean_sensor_getter_by_power(75)  # Выбор сенсоров с сильным сигналом
        >>> processor.get_vectors_final()
        {'sensor_id_0': {...}, 'sensor_id_1': {...}}  # Словарь с векторами для выбранных сенсоров
        """        
        self.__log(f"{'Создание словаря с векторами только для выбранных сенсоров в формате [сенсор][жест][шаг][значения],\n доступны через <object>.vectors_final'} ")
        self.vectors_final = {key: self.vectors[key] for key in [int(id) for id in self.CLEAN_SENSORS_FINAL] if key in self.vectors}

    def get_df_final(self) -> pd.DataFrame:
        """
        Создает DataFrame для выбранных сенсоров.

        Этот метод формирует DataFrame, содержащий данные только для
        сенсоров, которые были отобраны на основе силы сигнала.
        В итоговом DataFrame также включается столбец жестов.

        Возвращает
        ----------
        pd.DataFrame
            DataFrame с данными выбранных сенсоров и соответствующим столбцом жестов,
            доступный через атрибут `gestures_clean`.

        Примечания
        ----------
        - Метод предполагает, что атрибут `CLEAN_SENSORS_FINAL` уже был
        установлен ранее и содержит идентификаторы сенсоров, которые были
        выбраны на основе медианной силы сигнала.

        Примеры
        --------
        >>> processor = ProstheticHandDataProcessor(...)
        >>> processor.clean_sensor_getter_by_power(75)  # Выбор сенсоров с сильным сигналом
        >>> processor.get_df_final()
        DataFrame с данными для выбранных сенсоров и жестов.
        """        
        self.__log(f"{'Создание pd.DataFrame для выбранных сенсоров,\n доступны через <object>.gestures_clean\n'} ")
        self.gestures_clean = self.gestures[self.CLEAN_SENSORS_FINAL + ['gesture']]
    
    def  get_neutral_vectors(self) -> dict:
        """
        Собирает нейтральные векторы для выбранных сенсоров.

        Этот метод формирует словарь, содержащий векторы, представляющие
        нейтральное положение для каждого выбранного сенсора. Нейтральные
        векторы собираются только для жеста, идентифицируемого как 0.

        Возвращает
        ----------
        dict
            Словарь, где ключами являются идентификаторы сенсоров, а значениями
            являются вложенные словари. Вложенные словари имеют жесты в качестве
            ключей и соответствующие векторы в качестве значений. Формат:
            {sensor_id: {gesture: vectors}}.

        Примечания
        ----------
        - Метод предполагает, что атрибут `CLEAN_SENSORS_FINAL` уже был
        установлен ранее и содержит идентификаторы сенсоров, для которых
        будут собираться нейтральные векторы.
        - Векторы нейтрального положения сохраняются в атрибуте
        `vectors_neutral`, доступном через <object>.vectors_neutral.

        Примеры
        --------
        >>> processor = ProstheticHandDataProcessor(...)
        >>> processor.clean_sensor_getter_by_power(75)  # Выбор сенсоров с сильным сигналом
        >>> neutral_vectors = processor.get_neutral_vectors()
        Словарь с нейтральными векторами будет доступен через <object>.vectors_neutral.
        """
        self.__log(f"{'Создание словаря с векторами для нейтроального положения, только для выбранных сенсоров\nв формате [сенсор][жест][шаг][значения],доступны через <object>.vectors_neutral\n'} ")
        for column in self.CLEAN_SENSORS_FINAL:
            self.vectors_neutral[column] = {}
            for gesture in range(0,1):
                steps_ = self.__steps_getter(gesture)
                vectors_= self.__vector_getter(str(column), steps_)
                self.vectors_neutral[column][gesture] = vectors_       
    
    def get_vector_median_length(self) -> int:
        """
        Вычисляет медианную длину векторов для нейтрального положения.

        Этот метод анализирует длины векторов, представляющих нейтральное положение
        для выбранных сенсоров, и рассчитывает их медианное значение. Он выбирает
        векторы только для первого сенсора из списка `CLEAN_SENSORS_FINAL`.

        Возвращает
        ----------
        int
            Медианная длина векторов для нейтрального положения. Это целое
            число, которое представляет собой медиану всех длин векторов для
            нейтрального жеста.

        Примечания
        ----------
        - Метод предполагает, что атрибут `vectors_neutral` и
        `CLEAN_SENSORS_FINAL` уже инициализированы.
        - Данная медианная длина сохраняется в атрибуте `vector_median_length`,
        доступном через <object>.vector_median_length.

        Примеры
        --------
        >>> processor = ProstheticHandDataProcessor(...)
        >>> processor.get_neutral_vectors()  # Сбор нейтральных векторов
        >>> median_length = processor.get_vector_median_length()
        Доступная медианная длина векторов будет через <object>.vector_median_length.
        """        
        self.__log(f"{'Получаем медианную длину вектора для нейтрального положения, доступны через <object>.vector_median_length\n'} ")
        lengths = [len(v) for v in self.vectors_neutral[self.CLEAN_SENSORS_FINAL[0]][0].values()]
        self.vector_median_length = int(np.median(lengths))

    def get_median_vectors(self):
        """
        Вычисляет медианные векторы и их дисперсию для выбранных сенсоров.

        Этот метод создает словари, содержащие медианные векторы и их
        изменения (разности) для каждого жеста. Он использует интерполяцию
        для приведения всех векторов к одинаковой длине, основанной на
        заранее определенной медианной длине.

        Доступные атрибуты:
        --------------------
        median_vectors_all : dict
            Словарь, содержащий медианные векторы для каждого сенсора и жеста.
            Формат: [sensor][gesture][median_vector].
        median_vectors_diff : dict
            Словарь, содержащий разности медианных векторов для каждого
            сенсора и жеста. Формат: [sensor][gesture][difference_vector].

        Примечания
        ----------
        - Метод подразумевает, что атрибуты `CLEAN_SENSORS_FINAL` и
        `vector_median_length` уже инициализированы.
        - Векторы интерполируются до длины, заданной в `vector_median_length`,
        чтобы обеспечить корректное сравнение и вычисление медиан.

        Примеры
        --------
        >>> processor = ProstheticHandDataProcessor(...)
        >>> processor.get_neutral_vectors()  # Сбор нейтральных векторов
        >>> processor.get_vector_median_length()  # Получение медианной длины
        >>> processor.get_median_vectors()  # Получение медианных векторов
        Доступные медианные векторы можно будет найти через <object>.median_vectors_all и <object>.median_vectors_diff.
        """        
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

    def correct_vector(self, new_vector, median_vector, threshold : float=0.2) -> np.array:
        """
        Корректирует вектор значений на основе медианного вектора и заданного порога.

        Метод сравнивает значения нового вектора с соответствующими значениями медианного вектора.
        Если значение нового вектора выходит за пределы ожидаемого диапазона (определенного
        медианным значением и порогом), оно корректируется до медианного значения.

        Параметры
        ----------
        new_vector : np.array
            Вектор значений, который необходимо скорректировать.
        median_vector : np.array
            Медианный вектор, используемый для определения ожидаемых значений.
        threshold : float, optional
            Пороговое значение, определяющее допустимые отклонения от медианного вектора (по умолчанию 0.2).
            Значение 0.2 соответствует отклонению на 20%.

        Возвращает
        ----------
        tuple
            - np.array: Скорректированный вектор значений.
            - list: Список индексов, указывающих, были ли значения скорректированы (True) или нет (False).

        Примечания
        ----------
        - Если значение нового вектора выходит за пределы ожидаемого диапазона,
        оно будет заменено на соответствующее медианное значение.
        
        Примеры
        --------
        >>> new_vector = np.array([0.1, 0.5, 1.2, 0.8])
        >>> median_vector = np.array([0.2, 0.5, 1.0, 0.9])
        >>> corrected_vector, indices = correct_vector(new_vector, median_vector, threshold=0.2)
        >>> print(corrected_vector)
        array([0.2, 0.5, 1.0, 0.8])
        >>> print(indices)
        [True, False, True, False]
        """        
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
    
    def correct_vector_accordingly(self,vector, vector_idx, vector_diff) -> np.array:
        """
        Корректирует вектор значений на основе индексов, указывающих на необходимость коррекции,
        и вектора разностей, который используется для корректировки значений.

        Метод выполняет коррекцию значений в исходном векторе, основываясь на том, 
        какие значения необходимо изменить (по индексу) и разностях между текущими 
        и ожидаемыми значениями. Коррекция происходит последовательно, начиная с 
        индексов, которые нуждаются в изменении.

        Параметры
        ----------
        vector : np.array
            Исходный вектор значений, который необходимо скорректировать.
        vector_idx : list
            Список индексов, указывающих, какие элементы вектора следует скорректировать (True) или оставить без изменений (False).
        vector_diff : np.array
            Вектор разностей, который используется для корректировки значений.

        Возвращает
        ----------
        np.array
            Скорректированный вектор значений.

        Примечания
        ----------
        - Корректировка происходит таким образом, что значения, находящиеся в диапазоне
        изменений, получают значения, скорректированные с использованием разностей
        из вектора разностей.
        - Метод обрабатывает индексы последовательно, учитывая связанные изменения.

        Примеры
        --------
        >>> vector = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        >>> vector_idx = [False, True, True, False, True]
        >>> vector_diff = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> corrected_vector = correct_vector_accordingly(vector, vector_idx, vector_diff)
        >>> print(corrected_vector)
        array([1.0, 1.6, 1.7, 2.5, 3.0])
        """        
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

    def vector_corretion_assembly(self, vector, median_vector, median_vector_diff) -> np.array:
        """
        Собирает и корректирует вектор значений на основе медианного вектора и разностей.

        Этот метод объединяет два этапа коррекции в одном процессе. Сначала он 
        корректирует входной вектор на основе медианного вектора, а затем 
        применяет дополнительные изменения, используя разности между медианными 
        векторами.

        Параметры
        ----------
        vector : np.array
            Исходный вектор значений, который необходимо скорректировать.
        median_vector : np.array
            Медианный вектор, используемый для первой коррекции значений.
        median_vector_diff : np.array
            Вектор разностей между медианными значениями, который используется для 
            дополнительной коррекции.

        Возвращает
        ----------
        np.array
            Итоговый скорректированный вектор значений.

        Примечания
        ----------
        - Метод использует два вспомогательных метода: 
        `correct_vector` для первой коррекции и 
        `correct_vector_accordingly` для применения изменений на основе разностей.

        Примеры
        --------
        >>> vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> median_vector = np.array([1.0, 1.5, 3.0, 4.5, 5.0])
        >>> median_vector_diff = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> corrected_vector = vector_corretion_assembly(vector, median_vector, median_vector_diff)
        >>> print(corrected_vector)
        array([1.0, 1.6, 3.0, 4.5, 5.0])
        """        
        corrected_vector, corrected_vector_idx = self.correct_vector(vector, median_vector)
        return self.correct_vector_accordingly(corrected_vector, corrected_vector_idx, median_vector_diff)      
    
    def get_min_max_for_senser(self):
        """
        Получает минимальные и максимальные значения сигнала для каждого сенсора 
        на основе медианных векторов.

        Этот метод вычисляет минимальные и максимальные показатели сигнала 
        для каждого сенсора, а также для каждого жеста. Результаты сохраняются 
        в словаре, который доступен через атрибут `min_max_sensor`.

        Изменяет содержимое
        ----------
        dict
            Словарь с минимальными и максимальными значениями для каждого 
            сенсора и жеста. Формат: {sensor_id: {gesture_id: {'min': min_value, 
            'max': max_value}}}

        Примечания
        ----------
        - Данный метод предполагает, что медианные векторы для каждого 
        сенсора и жеста уже были рассчитаны и доступны в атрибуте 
        `median_vectors_all`.

        Примеры
        --------
        >>> get_min_max_for_senser()
        >>> print(min_max_sensor)
        {
            1: {0: {'min': 0.1, 'max': 1.0}, 1: {'min': 0.2, 'max': 1.2}, ...},
            2: {0: {'min': 0.05, 'max': 0.95}, 1: {'min': 0.15, 'max': 1.15}, ...},
            ...
        }
        """        
        self.__log(f"{'Получаем мин макс показатели сигнала для каждого сенсора на основе медианного вектора, в формате [sensor][жест]\nдоступны через <object>.min_max_sensor\n'} ")        
        for col in self.CLEAN_SENSORS_FINAL:
            self.min_max_sensor[col] = {}
            for gesture in range(0,6):
                self.min_max_sensor[col][gesture] = {'min': min(self.median_vectors_all[col][gesture]), 'max': max(self.median_vectors_all[col][gesture]) }

    
    def vector_correction_all(self, treshold : float = 1.5):
        """
        Корректирует векторы для всех сенсоров, основываясь на медианных 
        значениях и пороге для выбросов.

        Этот метод проходит по всем сенсорам и жестам, проверяет векторы 
        на наличие выбросов и корректирует их, если значения выходят за 
        заданные пределы (максимум или минимум, умноженные на порог). 
        Коррекция производится с использованием медианных векторов и их 
        различий.

        Параметры
        ----------
        treshold : float, optional
            Пороговое значение для определения выбросов. Значения, 
            превышающие максимальное значение, умноженное на этот порог 
            или ниже минимального значения, деленного на этот порог, 
            считаются выбросами. По умолчанию равно 1.5.

        Примечания
        ----------
        - Данный метод изменяет DataFrame `gestures_clean`, корректируя 
        векторы на основе их медианных значений и предопределенных 
        порогов для каждого сенсора и жеста.

        Примеры
        --------
        >>> vector_correction_all(treshold=1.5)
        >>> print(gestures_clean.head())
        ```
        
        В данном примере после вызова метода в DataFrame `gestures_clean` 
        будут обновлены значения векторов на основе корректировки.
        """        
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
        """
        Создает дополнительные признаки путем суммирования значений 
        сенсоров, расположенных подряд по 5, и сохраняет их в атрибуте 
        `additional_features_1`.

        Этот метод проходит по всем сенсорам, группируя их по 5, 
        и вычисляет сумму значений для каждой группы. 
        Результаты сохраняются в виде нового DataFrame, где каждая 
        новая особенность соответствует сумме 5 сенсоров. 
        Названия новых признаков формируются как `new_feature_1`, 
        `new_feature_2`, и так далее.

        Примечания
        ----------
        - Данный метод предполагает, что DataFrame `gestures` уже 
        содержит значения сенсоров, и что количество сенсоров 
        кратно 5. В противном случае, последние группы будут 
        игнорированы, если в них недостаточно сенсоров.

        Примеры
        --------
        >>> create_additional_features_1()
        >>> print(additional_features_1.head())
        """        
        self.__log(f"{'Создадим дополнительные признаки, путем ссумирования подряд по 5 сенсоров\nДоступны через <object>.additional_features_1\n'} ")        
        for i in range(0, self.n_omg_channels, 5):
            self.additional_features_1[f'new_feature_{i//5 + 1}'] = self.gestures.iloc[:, i:i+5].sum(axis=1)

    def create_additional_features_1_vector(self):
        """
        Создает векторы на основе дополнительных признаков `additional_features_1` 
        и интерполирует их до медианной длины. Результаты сохраняются в атрибуте 
        `additional_features_1_vector`.

        Этот метод группирует дополнительные признаки по индексам, 
        определяет медианный размер группы и интерполирует каждый 
        вектор до этого размера. Каждая колонка из `additional_features_1` 
        преобразуется в отдельный вектор.

        Примечания
        ----------
        - Интерполяция выполняется с использованием линейной интерполяции, 
        которая позволяет сгладить данные и привести их к одинаковой длине.
        - Рекомендуется проверять корректность входных данных перед 
        вызовом этого метода, чтобы избежать ошибок во время интерполяции.

        Примеры
        --------
        >>> create_additional_features_1_vector()
        >>> print(additional_features_1_vector)
        """        
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
        """
        Рассчитывает косинусную схожесть для векторов из `additional_features_1_vector` 
        и сохраняет средние значения схожести в атрибуте `additional_features_1_cosine_mean`.

        Этот метод вычисляет косинусную схожесть для каждого признака в `additional_features_1`, 
        основываясь на векторах, интерполированных ранее. Он использует скалярное произведение 
        и нормализацию векторов для получения матрицы косинусной схожести, из которой 
        вычисляется среднее значение для каждого признака.

        Примечания
        ----------
        - Косинусная схожесть измеряет схожесть между двумя векторами, 
        и значение близкое к 1 указывает на высокую схожесть.
        - В случае нулевых векторов может возникнуть деление на ноль, 
        поэтому рекомендуется проверять данные перед вызовом метода.

        Примеры
        --------
        >>> calculate_additional_features_1_cosine()
        >>> print(additional_features_1_cosine_mean)
        """        
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

    def clean_sensor_getter_by_cosine_add_feat_1(self, treshold:float) -> list:
        """
        Выбирает сенсоры из `additional_features_1` на основе косинусной схожести, 
        превышающей заданный порог.

        Метод создает список сенсоров, которые имеют среднюю косинусную схожесть 
        выше заданного порога. Эти сенсоры считаются важными для анализа 
        и доступны через атрибут `CLEAN_SENSORS_ADD_FEATURE_1`.

        Параметры
        ----------
        treshold : float
            Пороговое значение для фильтрации сенсоров по средней косинусной схожести.

        Возвращает
        -------
        list
            Список сенсоров, удовлетворяющих условию косинусной схожести выше порога.

        Примеры
        --------
        >>> sensors_above_threshold = clean_sensor_getter_by_cosine_add_feat_1(0.5)
        >>> print(sensors_above_threshold)
        """        
        self.__log(f"{'Выберем признаки из additional_features_1 согласно cosine\nДоступны через <object>.CLEAN_SENSORS_ADD_FEATURE_1\n'} ")        
        self.cosine_add_feature_1 = treshold
        self.CLEAN_SENSORS_ADD_FEATURE_1 = list({key: val for key, val in self.additional_features_1_cosine_mean.items() if val > treshold}.keys())   

    def cleaned_df_diff_adding(self):
        """
        Добавляет столбец с суммой абсолютных различий для очищенных данных жестов.

        Метод вычисляет абсолютные различия между последовательными значениями 
        для сенсоров, указанных в `CLEAN_SENSORS_FINAL`, и добавляет новый столбец 
        'diff' в DataFrame `gestures_clean`. Этот новый столбец представляет собой 
        сумму абсолютных различий по всем сенсорам.

        Примечание:
        ---------
        Если в `gestures_clean` нет очищенных сенсоров, метод может вернуть NaN 
        для всех значений в новом столбце.

        Примеры
        --------
        >>> cleaned_df_diff_adding()
        >>> print(gestures_clean.head())
        """        
        diff_ = self.gestures_clean[self.CLEAN_SENSORS_FINAL].diff().abs().copy()
        # self.gestures_clean.loc[:,'diff'] = diff_.sum(axis=1).values
        
        self.gestures_clean = self.gestures_clean.assign(diff=diff_.sum(axis=1).values)


    def create_fix_gestures_df(self, shift = (0,0)):
        """
        Создает таблицу смещения жестов и обновляет DataFrame с учетом фактических 
        начальных и конечных позиций жестов.

        Метод вычисляет фактические начальные и конечные позиции жестов на основе 
        столбца 'diff' в `gestures_clean`, а также применяет смещения, указанные в 
        параметре `shift`. Он создает новый DataFrame, который содержит информацию 
        о сменах жестов и их фактических позициях.

        Параметры
        ----------
        shift : tuple, optional
            Параметр смещения, представляющий смещение начала и конца жеста, 
            по умолчанию (0, 0).

        Примечание
        ---------
        Если столбец 'level_0' присутствует в `gestures_clean`, он будет удален. 
        Метод предполагает, что DataFrame `gestures_clean` имеет столбец 'step' и 
        'gesture'.

        Примеры
        --------
        >>> create_fix_gestures_df(shift=(5, -3))
        >>> print(gestures_clean.head())
        """        
        self.shift = shift
        
        # в случае если уже применялся метод reset_index, то данный признак уже существует и его надо удалить
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
    
