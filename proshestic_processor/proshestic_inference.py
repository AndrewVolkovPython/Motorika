import time 
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


def __preprocessing(self, x):
    indices = [int(i) for i in self.CLEAN_SENSORS_FINAL + self.CLEAN_SENSORS_ADD_FEATURE_1]
    # Извлекаем элементы по индексам
    y = x[indices]
    return y

def __inference(self, model_name, x):
    if isinstance(model_name, RandomForestClassifier):
        x = pd.DataFrame(x.reshape(1, len(x)), columns=self.CLEAN_SENSORS_FINAL + self.CLEAN_SENSORS_ADD_FEATURE_1)
        y = model_name.predict(x)
    else:
        y = model_name.predict([x])
    return y

def __postprocessing(x, prev):
    if prev is None:
        y = x
    else:
        y = x*0.1 + prev*0.9 # Holt-Winters filter
    return y

def __commands(x):
    y = np.round(np.clip(x / 100, 0, 1)*100).astype(int)
    return y

def inference(self, model_name , df_sim : pd.DataFrame, timeout : float = 0.033):
    TIMEOUT = timeout
        
    i = 0
    ts_old = time.time()
    ts_diff = 0
    ts_diff_all = np.array([])

    y_previous = None
    y_dct = {
        'omg_sample':[],
        'enc_sample':[],
        'sample_preprocessed':[],
        
        'y_predicted':[],
        'y_postprocessed':[],
        'y_commands':[],
    }
    while True:    
        
        # [Data reading]
        ts_start = time.time()
        
        try:
            # [Sim data]
            if i < len(df_sim):
                sample = df_sim.values[i]
            else:
                break
            # [/Sim data]
            [omg_sample, acc_sample, enc_sample, [button, sync, ts]] = np.array_split(sample, [50, 56, 62])
            
        except Exception as e:
            print(e)
            
            
        # [/Data Reading]
            
        # [Data preprocessing]
        sample_preprocessed = __preprocessing(self, omg_sample)
        # [/Data preprocessing]
        
        # [Inference]
        y_predicted         = __inference(self, model_name, sample_preprocessed)
        # [/Inference]
        
        # [Inference Postprocessing]
        y_postprocessed     = __postprocessing(y_predicted, y_previous)
        # [/Inference Postprocessing]
        
        # [Commands composition]
        y_commands          = __commands(y_postprocessed)
        # [/Commands composition]
        
        # [Commands sending]
        # NO COMMANDS SENDING IN SIMULATION
        # [/Commands sending]
        
        # [Data logging]
        y_dct['omg_sample'].append(omg_sample)
        y_dct['enc_sample'].append(enc_sample)
        y_dct['sample_preprocessed'].append(sample_preprocessed)
        y_dct['y_predicted'].append(y_predicted)
        y_dct['y_postprocessed'].append(y_postprocessed)
        y_dct['y_commands'].append(y_commands)
        # [/Data logging]

        y_previous = y_postprocessed
        
        ts_diff = time.time() - ts_start
        ts_diff_all = np.append(ts_diff_all,ts_diff)

        assert(ts_diff<TIMEOUT), 'Calculation cycle takes more than TIMEOUT, halting...'
        ts_old = ts_start
        i += 1
        
    print(f'Среднее время предсказания: {(av_time := ts_diff_all.mean()):.6f}, что составляет {av_time/TIMEOUT*100:.2f}% от требуемого лимита') 
