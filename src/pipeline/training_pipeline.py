import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import save_object

from src.components.data_ingesion import DataIngestion
from src.components.data_transfermation import DataTransfermation
from src.components.model_training import ModelTrainer



if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transfermation = DataTransfermation()
    train_arr,test_arr,_ = data_transfermation.initiate_data_transfermation(train_data_path,test_data_path)
    data_training = ModelTrainer()
    data_training.initiate_model_traininig(train_arr,test_arr)