from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np

def plot_sensors(self, cols, range_: range, df_source = 'init'):
    
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
    '''
    Plotly график для указанного сенсора и типа жеста
    ''' 
    
    if df_source == 'init':
        df = self.gestures
    else:
        df = self.gestures_clean   
    
    filtered_data = df[df.gesture == gesture][[sensor]]
    fig = px.line(filtered_data, x=filtered_data.index, y=sensor, title=f'График зависимости признака {sensor} от steps')

    fig.show()
    
def graph_sensor_gestures(self, range_, type):
    
    # if ('diff' not in self.gestures.columns) | ('diff' not in self.gestures_clean.columns):
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
    fig = plt.figure(figsize=figzize)
    plt.plot(y_true,  c='C0', label='y_true', linewidth=linewidht)
    plt.plot(y_pred, c='C1', label='y_pred', linewidth=linewidht)

    plt.yticks(np.arange(len(self.GESTURES)), self.GESTURES)
    plt.grid()
    plt.xlabel('Timesteps')
    plt.legend()
    plt.title('Test')
    plt.tight_layout()        