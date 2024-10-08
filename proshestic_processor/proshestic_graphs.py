from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np

def plot_sensors(self, cols: list, range_: range, df_source = 'init'):
    """
    Визуализирует указанные столбцы из выбранного DataFrame на множестве подграфиков.

    Параметры:
    ----------
    cols : list
        Список названий столбцов, которые будут отображены на графиках.
    
    range_ : range
        Диапазон индексов строк для отображения на графиках (напр. range(0, 100)).
    
    df_source : str, по умолчанию 'init'
        Источник данных для построения графиков. Может быть одним из:
        - 'init': исходные данные (self.gestures)
        - 'clean': очищенные данные (self.gestures_clean)
        - 'af1': данные из дополнительного признака 1 (self.additional_features_1)

    Исключения:
    -----------
    ValueError:
        Выбрасывается, если значение `df_source` не принадлежит к ['init', 'clean', 'af1'].
    
    Описание:
    ---------
    Функция создает сетку из 50 подграфиков (10x5) и отображает данные для каждого столбца
    из списка `cols`. Графики будут построены на основе выбранного DataFrame, а данные для графиков
    извлекаются в пределах диапазона строк, указанного в `range_`.
    
    Если количество указанных столбцов меньше 50, лишние подграфики удаляются.

    Пример использования:
    ---------------------
    >>> self.plot_sensors(cols=['sensor1', 'sensor2'], range_=range(0, 100), df_source='clean')
    """
    
    valid_sources = ['init', 'clean', 'af1']
    
    if df_source not in valid_sources:
        raise ValueError(f"Некорректно указан источник: {df_source}. Должен быть один из {valid_sources}")
    
    if df_source == 'init':
        df = self.gestures
    elif df_source =='clean':
        df = self.gestures_clean    
    elif df_source == 'af1':
        df = self.additional_features_1    
    
    fig, axes = plt.subplots(10, 5, figsize=(20, 20), sharex=True, sharey=True)

    # Преобразуем 2D массив осей в 1D массив для удобства
    axes = axes.flatten()

    for i, column in enumerate(cols):
        axes[i].plot(df[column].values[range_])
        axes[i].set_title(column)

    # Удаляем лишние подграфики, если их меньше 50
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()     

def plotly_sensor(self, gesture, sensor, df_source = 'init'):
    """
    Строит интерактивный график с помощью Plotly для указанного сенсора и типа жеста.

    Параметры:
    ----------
    gesture : str
        Тип жеста, для которого необходимо построить график. 
        Должен соответствовать одному из значений в столбце 'gesture' в DataFrame.

    sensor : str
        Название сенсора, данные которого будут визуализированы. 
        Должен соответствовать одному из названий столбцов в DataFrame.

    df_source : str, optional
        Источник данных для построения графика. 
        Принимает значения 'init' для использования начального DataFrame или 
        'clean' для использования очищенного DataFrame. По умолчанию 'init'.

    Возвращает:
    ----------
    None

    Примечания:
    ----------
    Функция фильтрует данные по указанному жесту и строит график зависимости 
    значения указанного сенсора от временных шагов. 
    Используется библиотека Plotly для создания интерактивных графиков.
    """
    if df_source == 'init':
        df = self.gestures
    else:
        df = self.gestures_clean   
    
    filtered_data = df[df.gesture == gesture][[sensor]]
    fig = px.line(filtered_data, x=filtered_data.index, y=sensor, title=f'График зависимости признака {sensor} от steps')

    fig.show()
    
def graph_sensor_gestures(self, range_ : range, type : int = 1):
    """
    Строит графики для сенсорных данных и жестов на основе переданного диапазона и типа графика.

    Параметры:
    ----------
    range_ : range
        Диапазон индексов данных для визуализации. 
        Указывает, какие временные шаги будут включены в график.
    
    type : int, optional
        Тип графика, который нужно построить. 
        1 - стандартный график с тремя подграфиками (сенсоры, разница, жест).
        2 - график с четырьмя подграфиками, добавляющий расположение жестовов после сдвига.
        По умолчанию 1.

    Возвращает:
    ----------
    None

    Исключения:
    ----------
    ValueError
        Если указанный тип графика не входит в допустимый диапазон [1, 2].
    
    Примечания:
    ----------
    Функция проверяет наличие признака 'diff' в данных `gestures_clean`. 
    Если признак отсутствует, будет выведено предупреждение. 
    Графики отображают сенсорные данные, разницу и информацию о жестах, 
    с вертикальными линиями, обозначающими временные интервалы.
    """    
    
    valid_sources = range(1,3)
    
    if type not in valid_sources:
        raise ValueError(f"Некорректно указан тип графика: {type}. Должен быть один из {valid_sources}")
    
    if 'diff' not in self.gestures_clean.columns:
        self.__log(f"{'Добавьте сначала diff признак через запуск cleaned_df_diff_adding()'} ")        
        return
    data = self.gestures_clean[self.CLEAN_SENSORS_FINAL].values[range_]

    if type == 1:
        graph_number = 3
    else:
        graph_number = 4    

    fig, axx = plt.subplots(graph_number, 1, sharex=True, figsize=(20, 10), dpi=300)
    plt.sca(axx[0])
    plt.plot(data, linewidth=.8)

    plt.sca(axx[1])
    plt.plot(self.gestures_clean['diff'].values[range_], linewidth=.8)

    plt.sca(axx[2])
    plt.plot(self.gestures['gesture'].values[range_], color='red', linewidth=5)
    plt.grid()

    if type == 2:
        plt.sca(axx[3])
        plt.plot(self.gestures_clean['gesture'].values[range_], color='darkgrey', linewidth=5)
        plt.grid()

    period = 40  
    x_ticks = np.arange(0, len(range_), period)

    # Рисуем графики
    for ax in axx:
        for tick in x_ticks:
            ax.axvline(x=tick, color='white', linestyle='--', linewidth=0.5)

    plt.yticks(np.arange(len(self.GESTURES)), self.GESTURES)
    plt.xlabel('Timesteps')
    plt.title('Protocol')


    plt.suptitle('OMG and Protocol')
    plt.tight_layout()    

def plot_results(self, y_true, y_pred, figzize : tuple = (20,4) , linewidht : float = 1):
    """
    Визуализирует истинные значения и предсказанные значения на графике.

    Параметры:
    ----------
    y_true : array-like
        Истинные значения (например, метки классов) для сравнения с предсказанными.
    
    y_pred : array-like
        Предсказанные значения, которые будут отображены на графике.
    
    figzize : tuple, optional
        Размер фигуры в формате (ширина, высота). По умолчанию (20, 4).
    
    linewidht : float, optional
        Толщина линий на графике. По умолчанию 1.

    Возвращает:
    ----------
    None

    Примечания:
    ----------
    График отображает истинные и предсказанные значения, с использованием сетки и легенды для
    улучшения визуализации. На оси Y отображаются метки классов, определяемые в self.GESTURES.
    """    
    fig = plt.figure(figsize=figzize)
    plt.plot(y_true,  c='C0', label='y_true', linewidth=linewidht)
    plt.plot(y_pred, c='C1', label='y_pred', linewidth=linewidht)

    plt.yticks(np.arange(len(self.GESTURES)), self.GESTURES)
    plt.grid()
    plt.xlabel('Timesteps')
    plt.legend()
    plt.title('Test')
    plt.tight_layout()        