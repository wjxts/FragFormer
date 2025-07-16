import logging 
import datetime
from pathlib import Path 
import sys

class Logger:
    def __init__(self, name='fragment_mol'):
        time_tick = datetime.datetime.now()
        self.current_time = time_tick.strftime('%Y-%m-%d-%H:%M:%S')
                
        log_file = str(Path('log') / f'{self.current_time}.log')
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # logging.basicConfig(filename=log_file,
        #             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        #             filemode='w')
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.removeHandler(self.logger.handlers[0]) # remove the default console handler
        self.logger.addHandler(file_handler)
       
        self.print(f"current time: {self.current_time}")
       
    def print(self, msg):
        print(msg)
        self.logger.info(msg)

logger = Logger()
# logger = None