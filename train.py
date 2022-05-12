import logging
import os

import yaml
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import mlflow
import src.GetPoolData as GetPoolData
import src.ML_Strategy as ML_Strategy
import src.Strategy_Wrapper as Strategy_Wrapper
import src.ActiveStrategyFramework as ActiveStrategyFramework

import pandas as pd
import numpy as np

config_path = os.path.join('/Users/amir1/PycharmProjects/uniswap-automation/config/params_all.yaml')
config = yaml.safe_load(open(config_path))['train']
os.chdir(config['dir_folder'])

logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


############
def get_version_model(config_name, client):
    """
    Получение последней версии модели из MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f"name='{config_name}'")):
        # Все версии модели
        dict_push[count] = value
    return dict(list(dict_push.items())[-1][1])['version']
########


def main():

    swap_data = GetPoolData.get_pool_data_flipside(**config['swap_data'])
    price_data = GetPoolData.get_price_data_bitquery(**config['price_data'])

    # little preprocessing

    DATE_BEGIN = pd.to_datetime('2022-01-15 00:00PM', utc=True)
    DATE_END = pd.to_datetime('2022-01-28 00:00PM', utc=True)
    z_score_cutoff = 5
    window_size = 60

    # Data for strategy simulation cleaning
    STRATEGY_FREQUENCY = 'M'
    simulate_data_filtered = ActiveStrategyFramework.aggregate_price_data(price_data, STRATEGY_FREQUENCY)
    simulate_data_filtered_roll = simulate_data_filtered.quotePrice.rolling(window=window_size)
    simulate_data_filtered['roll_median'] = simulate_data_filtered_roll.median()
    roll_dev = np.abs(simulate_data_filtered.quotePrice - simulate_data_filtered.roll_median)
    simulate_data_filtered['median_abs_dev'] = 1.4826 * roll_dev.rolling(window=window_size).median()
    outlier_indices = np.abs(simulate_data_filtered.quotePrice - simulate_data_filtered.roll_median) >= z_score_cutoff * \
                      simulate_data_filtered['median_abs_dev']
    simulate_data_price = simulate_data_filtered[~outlier_indices]['quotePrice'][DATE_BEGIN:DATE_END]

    # Data for statistical analaysis (AGGREGATED_MINUTES frequency data)
    #STAT_MODEL_FREQUENCY = 'H'  # forecast returns at a daily frequency

    # Initial Position Details
    INITIAL_TOKEN_0 = 100000
    INITIAL_TOKEN_1 = INITIAL_TOKEN_0 * simulate_data_price[0]
    INITIAL_POSITION_VALUE = 2 * INITIAL_TOKEN_0
    FEE_TIER = 0.003

    # Set decimals according to your pool
    DECIMALS_0 = 6
    DECIMALS_1 = 18
    swap_data['virtual_liquidity'] = swap_data['VIRTUAL_LIQUIDITY_ADJUSTED'] * (10 ** ((DECIMALS_1 + DECIMALS_0) / 2))
    swap_data['traded_in'] = swap_data.apply(lambda x: -x['amount0'] if (x['amount0'] < 0) else -x['amount1'],
                                             axis=1).astype(float)
    swap_data['traded_out'] = swap_data.apply(lambda x: x['amount0'] if (x['amount0'] > 0) else x['amount1'],
                                              axis=1).astype(float)

    # little preprocessing

    # MLFlow tracking
    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment(config['name_experiment'])
    with mlflow.start_run():
        strategy = ML_Strategy.ML_Strategy(price_data, **config['model'])
        simulated_strategy = ActiveStrategyFramework.simulate_strategy(simulate_data_price, swap_data, strategy,
                                                                       INITIAL_TOKEN_0, INITIAL_TOKEN_1, FEE_TIER,
                                                                       DECIMALS_0, DECIMALS_1)
        sim_data = ActiveStrategyFramework.generate_simulation_series(simulated_strategy, strategy)
        strat_result = ActiveStrategyFramework.analyze_strategy(sim_data, frequency=STRATEGY_FREQUENCY)

        # logging model and params
        mlflow.log_params(config['model'])
        mlflow.log_metrics(strat_result)

        #wrapped_model = Strategy_Wrapper(strategy)
        #signature = infer_signature()

        #mlflow.pyfunc.log_model("ml_model", python_model=wrapped_model)

        mlflow.xgboost.log_model(strategy.model,
                                 artifact_path="model_xgb",
                                 registered_model_name=f"{config['model_xgb']}")

        mlflow.log_artifact(local_path='./train.py',
                            artifact_path='code')
        mlflow.end_run()

    # Get model last version
    client = MlflowClient()
    last_version_xgb = get_version_model(config['model_xgb'], client)

    yaml_file = yaml.safe_load(open(config_path))
    yaml_file['predict']["version_xgb"] = int(last_version_xgb)
    yaml_file['predict']["model"]["model_link"] = f"models:/{config['model_xgb']}/{last_version_xgb}"

    with open(config_path, 'w') as fp:
        yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True)


if __name__ == "__main__":
    main()

