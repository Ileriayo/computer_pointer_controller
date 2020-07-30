import time
import os

class ModelMetrics:
        
    def save_to_file(self, text_file, model_precision, inf_time, fps, load_time):
        with open(os.path.join('../metrics/' + model_precision, text_file), 'w') as f:
            f.write(str(inf_time) + '\n' + str(fps) + '\n' + str(load_time))
