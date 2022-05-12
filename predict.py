import logging
import os

import numpy as np
import pandas as pd
import yaml

import mlflow
import src.GetPoolData as GetPoolData
import src.ML_Strategy as ML_Strategy
import src.ActiveStrategyFramework as ActiveStrategyFramework
from datetime import datetime, timedelta

config_path = os.path.join('/Users/amir1/PycharmProjects/uniswap-automation/config/params_all.yaml')
position_path = os.path.join('/Users/amir1/PycharmProjects/uniswap-automation/config/position.yaml')

config = yaml.load(open(config_path), Loader=yaml.Loader)['predict']
position_data = yaml.load(open(position_path), Loader=yaml.Loader)

os.chdir(config['dir_folder'])

logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def log_strategy_step(current_observation, model):
    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment(config['name_experiment'])
    with mlflow.start_run():
        dict_metrics = model.dict_components(current_observation)
        dict_metrics.pop('time')
        dict_metrics.pop('reset_reason')
        mlflow.log_metrics(dict_metrics)
        mlflow.end_run()


def main():
    """
    Получение тематик из текста и сохранение их в файл
    """

    # Загрузка последних сохраненнных моделей из MLFlow
    mlflow.set_tracking_uri("http://localhost:5050")
    #model_uri_xgb = f"models:/{config['xgboost_model']}/{config['version_xgb']}"

    #model_xgb = mlflow.xgboost.load_model(model_uri_xgb)
    #price_data = GetPoolData.get_price_data_bitquery(**config['price_data'])
    #price_data = price_data[config['price_data']['date_begin']+" 00:00:00+00:00": config['price_data']['date_end']+" 00:00:00+00:00"]
    # Get data for required time
    now = datetime.now() #- timedelta(days=0)
    price_date_end = now.strftime("%Y-%m-%d")    #%H:%M:%S") Add in future
    hour_before = datetime.now() - timedelta(days=1)
    price_date_begin = hour_before.strftime("%Y-%m-%d") #%H:%M:%S") Add in future
    DOWNLOAD_DATA = True
    price_data = GetPoolData.get_price_data_bitquery(config['price_data']['token_0_address'], config['price_data']['token_1_address'], price_date_begin, price_date_end,
                                                     config['price_data']['api_token'], config['price_data']['file_name'], DOWNLOAD_DATA)
    #print(price_data)
    strategy = ML_Strategy.ML_Strategy(price_data, **config['model'])

    #yaml_file = yaml.load(open(config_path), Loader=yaml.Loader)
    #yaml_file['predict']['last_timepoint_updated'] = price_data_end
    #with open(config_path, 'w') as fp:
    #    yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True, default_flow_style=False)

    # Подгрузка данных каждые 10 минут - расчет текущих параметров (ликвидность, цена) - загрузка их в StartegyObservation - вычисление новых параметров -

    # Подкачка данных с grapthQL
    #price_data = GetPoolData.get_current_state()
    #yaml_file = yaml.safe_load(open(config_path))
    #yaml_file['predict']['current_position_info']['current_price'] = ((int(price_data['data']['pool']['sqrtPrice']) /
    #                                                                   2**96)**2) * 10**(config['current_position_info']['decimals_0']-config['current_position_info']['decimals_1'])

    # создание StrategyObservation

    #config = yaml.safe_load(open(config_path))['predict']
    #os.chdir(config['dir_folder'])

    current_observation = ActiveStrategyFramework.StrategyObservation(strategy_in=strategy, current_price=price_data['quotePrice'][-1],
                                                                      **position_data['current_position_info'], timepoint=price_data.index[-1])
    # Установка параметров yaml на новые значения(по позиции)
    res_dict = strategy.dict_components(current_observation)
    #print(current_observation.liquidity_ranges)
    #print(current_observation.strategy_info)
    yaml_file = yaml.load(open(position_path), Loader=yaml.Loader)
    yaml_file['current_position_info']['liquidity_in_0'] = current_observation.liquidity_in_0
    yaml_file['current_position_info']['liquidity_in_1'] = current_observation.liquidity_in_1
    yaml_file['current_position_info']['token_0_left_over'] = current_observation.token_0_left_over
    yaml_file['current_position_info']['token_1_left_over'] = current_observation.token_1_left_over
    yaml_file['current_position_info']['token_0_fees_uncollected'] = current_observation.token_0_fees_uncollected
    yaml_file['current_position_info']['token_1_fees_uncollected'] = current_observation.token_1_fees_uncollected
    yaml_file['current_position_info']['liquidity_ranges'] = current_observation.liquidity_ranges
    yaml_file['current_position_info']['strategy_info'] = current_observation.strategy_info
    yaml_file['last_timepoint_updated'] = str(price_data.index[-1])
    with open(position_path, 'w') as fp:
        yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True, default_flow_style=False)

    log_strategy_step(current_observation, strategy)


if __name__ == "__main__":
    main()
