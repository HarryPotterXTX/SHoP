
import os
import sys
import time
import torch
import random
import numpy as np
from typing import Dict
from torch.utils.tensorboard import SummaryWriter

timestamp = time.strftime("_%Y_%m%d_%H%M%S")

class MyLogger():
    def __init__(self, project_name:str, stdlog:bool=True, tensorboard:bool=True, outputs_dir:str='outputs'):
        self.project_dir = os.path.join(outputs_dir, project_name)
        self.stdlog = stdlog
        self.tensorboard = tensorboard
        # self.project_dir += timestamp
        # temp_name = self.project_dir
        # for i in range(10):
        #     if not os.path.exists(temp_name):
        #         break
        #     temp_name = self.project_dir + '-' + str(i)
        # self.project_dir = temp_name
        self.logdir = self.project_dir
        self.logger_dict = {}
        if tensorboard:
            self.tensorboard_init()
        else:
            os.makedirs(self.project_dir, exist_ok=True)
        if stdlog:
            self.stdlog_init()
        self.dir_init()

    def stdlog_init(self):
        stderr_handler=open(os.path.join(self.logdir,'stderr.log'), 'w')
        sys.stderr=stderr_handler
        
    def tensorboard_init(self,):
        self.tblogger = SummaryWriter(self.logdir, flush_secs=30)
        self.logger_dict['tblogger']=self.tblogger
    
    def dir_init(self,):
        self.script_dir = os.path.join(self.project_dir, 'script')
        self.model_dir = os.path.join(self.project_dir, 'model')
        os.mkdir(self.script_dir)
        os.mkdir(self.model_dir)

    def log_metrics(self, metrics_dict: Dict[str, float], iters):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'csvlogger':
                self.logger_dict[logger_name].log_metrics(metrics_dict, iters)
                self.logger_dict[logger_name].save()
            elif logger_name == 'clearml_logger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].report_scalar(k, k, metrics_dict[k], iters)
            elif logger_name == 'tblogger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].add_scalar(k, metrics_dict[k], iters)

    def close(self):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                self.logger_dict[logger_name].close()

def reproduc():
    """Make experiments reproducible
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True