import pandas as pd
import numpy as np
import math
import arch
from src import UNI_v3_funcs
from src import ActiveStrategyFramework
import copy
import xgboost as xgb
from src.feature_extraction import *
import mlflow


# ML strategy


class ML_Strategy():
    def __init__(self, model_data, alpha_param, tau_param, volatility_reset_ratio, tokens_outside_reset=.05,
                 data_frequency='M', default_width=.5, days_ar_model=180, return_forecast_cutoff=0.15,
                 z_score_cutoff=5, gamma=0.01, learning_rate=0.05, max_depth=8, n_estimators=400, load_model=False, model_link=None):

        # Allow for different input data frequencies, always get 1 day ahead forecast
        # Model data frequency is expressed in minutes

        if data_frequency == 'D':
            self.annualization_factor = 365 ** .5
            self.resample_option = '1D'
            self.window_size = 15
        elif data_frequency == 'H':
            self.annualization_factor = (24 * 365) ** .5
            self.resample_option = '1H'
            self.window_size = 30
        elif data_frequency == 'M':
            self.annualization_factor = (60 * 24 * 365) ** .5
            self.resample_option = '1 min'
            self.window_size = 60

        self.alpha_param = alpha_param
        self.tau_param = tau_param
        self.volatility_reset_ratio = volatility_reset_ratio
        self.data_frequency = data_frequency
        self.tokens_outside_reset = tokens_outside_reset
        self.default_width = default_width
        self.return_forecast_cutoff = return_forecast_cutoff
        self.days_ar_model = days_ar_model
        self.z_score_cutoff = z_score_cutoff
        self.window_size = 60

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        self.garch_data = self.clean_data_for_garch(model_data)
        self.xgboost_data = self.clear_data_for_xgboost(model_data)
        if load_model:
            mlflow.set_tracking_uri("http://localhost:5050")
            self.model = mlflow.xgboost.load_model(model_link)
        else:
            self.model = self.train_model()

    #####################################
    # Estimate ARX model at current timepoint
    #####################################

    def train_model(self):
        X_train = self.xgboost_data.drop(['quotePrice'], axis=1)
        y_train = self.xgboost_data['quotePrice'].copy()
        parameters = {'gamma': self.gamma, 'learning_rate': self.learning_rate,
                      'max_depth': self.max_depth, 'n_estimators': self.n_estimators}
        model = xgb.XGBRegressor(**parameters, objective='reg:squarederror')
        model.fit(X_train, y_train, verbose=False)

        return model

    def clean_data_for_garch(self, data_in):
        data_filled = ActiveStrategyFramework.fill_time(data_in)

        # Filter according to Median Absolute Deviation
        # 1. Generate rolling median
        data_filled_rolling = data_filled.quotePrice.rolling(window=self.window_size)
        data_filled['roll_median'] = data_filled_rolling.median()

        # 2. Compute rolling absolute deviation of current price from median under Gaussian
        roll_dev = np.abs(data_filled.quotePrice - data_filled.roll_median)
        data_filled['median_abs_dev'] = 1.4826 * roll_dev.rolling(window=self.window_size).median()

        # 3. Identify outliers using MAD
        outlier_indices = np.abs(data_filled.quotePrice - data_filled.roll_median) >= self.z_score_cutoff * data_filled[
            'median_abs_dev']

        # impute
        # data_filled['quotePrice']      = np.where(outlier_indices.values == 0,  data_filled['quotePrice'].values,data_filled['roll_median'].values)

        # drop
        data_filled = data_filled[~outlier_indices]

        return data_filled

    def clear_data_for_xgboost(self, data_in):
        data_filled = ActiveStrategyFramework.fill_time(data_in)
        data_filled = extractAll(data_filled)
        return data_filled

    def generate_model_forecast(self, timepoint):

        # Compute returns with data_frequency frequency starting at the current timepoint and looking backwards
        current_data = self.garch_data.loc[:timepoint].resample(self.resample_option, closed='right', label='right',
                                                                origin=timepoint).last()
        current_data['price_return'] = current_data['quotePrice'].pct_change()
        current_data = current_data.dropna(axis=0, subset=['price_return'])
        ar_model = arch.univariate.ARX(current_data.price_return[(
                    current_data.index >= (timepoint - pd.Timedelta(str(self.days_ar_model) + ' days')))].to_numpy(),
                                       lags=1, rescale=True)
        ar_model.volatility = arch.univariate.GARCH(p=1, q=1)

        res = ar_model.fit(update_freq=0, disp="off")
        scale = res.scale

        forecasts = res.forecast(horizon=1, reindex=False)

        return_forecast      = forecasts.mean.to_numpy()[0][-1] / scale
        sd_forecast = (forecasts.variance.to_numpy()[0][-1] / np.power(res.scale, 2)) ** 0.5 * self.annualization_factor

        # Return forecast XGBOOST
        current_data_xg = self.xgboost_data.loc[:timepoint].resample(self.resample_option, closed='right',
                                                                     label='right', origin=timepoint).last()

        X = current_data_xg.drop(['quotePrice'], axis=1)
        y = current_data_xg.iloc[-1]['quotePrice']
        features = X.iloc[-1:, :]
        predict_features = np.array(features).reshape(-1, 7)
        return_forecast = self.model.predict(predict_features)[0] / y - 1.0

        result_dict = {'return_forecast': return_forecast, 'sd_forecast': sd_forecast}

        return result_dict

    #####################################
    # Check if a rebalance is necessary.
    # If it is, remove the liquidity and set new ranges
    #####################################

    def check_strategy(self, current_strat_obs):

        model_forecast = None
        LIMIT_ORDER_BALANCE = current_strat_obs.liquidity_ranges[1]['token_0'] + current_strat_obs.liquidity_ranges[1][
            'token_1'] / current_strat_obs.price
        BASE_ORDER_BALANCE = current_strat_obs.liquidity_ranges[0]['token_0'] + current_strat_obs.liquidity_ranges[0][
            'token_1'] / current_strat_obs.price

        if not 'last_vol_check' in current_strat_obs.strategy_info:
            current_strat_obs.strategy_info['last_vol_check'] = current_strat_obs.time

        #####################################
        #
        # This strategy rebalances in three scenarios:
        # 1. Leave Reset Range
        # 2. Volatility has dropped           (volatility_reset_ratio)
        # 3. Tokens outside of pool greater than 5% of value of LP position
        #
        #####################################

        #######################
        # 1. Leave Reset Range
        #######################
        LEFT_RANGE_LOW = current_strat_obs.price < current_strat_obs.strategy_info['reset_range_lower']
        LEFT_RANGE_HIGH = current_strat_obs.price > current_strat_obs.strategy_info['reset_range_upper']

        #######################
        # 2. Volatility has dropped
        #######################
        # Rebalance if volatility has gone down significantly
        # When volatility increases the reset range will be hit
        # Check every hour (60  minutes)

        ar_check_frequency = 60
        time_since_reset = current_strat_obs.time - current_strat_obs.strategy_info['last_vol_check']

        VOL_REBALANCE = False
        if (time_since_reset.total_seconds() / 60) >= ar_check_frequency:

            current_strat_obs.strategy_info['last_vol_check'] = current_strat_obs.time
            model_forecast = self.generate_model_forecast(current_strat_obs.time)

            if model_forecast['sd_forecast'] / current_strat_obs.liquidity_ranges[0][
                'volatility'] <= self.volatility_reset_ratio:
                VOL_REBALANCE = True
            else:
                VOL_REBALANCE = False

        #######################
        # 3. Tokens outside of pool greater than 5% of value of LP position
        #######################

        left_over_balance = current_strat_obs.token_0_left_over + current_strat_obs.token_1_left_over / current_strat_obs.price

        if (left_over_balance > self.tokens_outside_reset * (LIMIT_ORDER_BALANCE + BASE_ORDER_BALANCE)):
            TOKENS_OUTSIDE_LARGE = True
        else:
            TOKENS_OUTSIDE_LARGE = False

        if 'force_initial_reset' in current_strat_obs.strategy_info:
            if current_strat_obs.strategy_info['force_initial_reset']:
                INITIAL_RESET = True
                current_strat_obs.strategy_info['force_initial_reset'] = False
            else:
                INITIAL_RESET = False
        else:
            INITIAL_RESET = False

        # if a reset is necessary
        if ((((LEFT_RANGE_LOW | LEFT_RANGE_HIGH) | VOL_REBALANCE) | TOKENS_OUTSIDE_LARGE) | INITIAL_RESET):
            current_strat_obs.reset_point = True

            if (LEFT_RANGE_LOW | LEFT_RANGE_HIGH):
                current_strat_obs.reset_reason = 'exited_range'
            elif VOL_REBALANCE:
                current_strat_obs.reset_reason = 'vol_rebalance'
            elif TOKENS_OUTSIDE_LARGE:
                current_strat_obs.reset_reason = 'tokens_outside_large'
            elif INITIAL_RESET:
                current_strat_obs.reset_reason = 'initial_reset'

            # Remove liquidity and claim fees
            current_strat_obs.remove_liquidity()

            # Reset liquidity
            liq_range, strategy_info = self.set_liquidity_ranges(current_strat_obs, model_forecast)
            return liq_range, strategy_info
        else:
            return current_strat_obs.liquidity_ranges, current_strat_obs.strategy_info

    def set_liquidity_ranges(self, current_strat_obs, model_forecast=None):

        ###########################################################
        # STEP 1: Do calculations required to determine base liquidity bounds
        ###########################################################

        # Fit model
        if model_forecast is None:
            model_forecast = self.generate_model_forecast(current_strat_obs.time)

        if current_strat_obs.strategy_info is None:
            strategy_info_here = dict()
        else:
            strategy_info_here = copy.deepcopy(current_strat_obs.strategy_info)

        # Limit return prediction to a return_forecast_cutoff % change
        if np.abs(model_forecast['return_forecast']) > self.return_forecast_cutoff:
            model_forecast['return_forecast'] = np.sign(model_forecast['return_forecast']) * self.return_forecast_cutoff

        # If error in volatility computation use last or overall standard deviation of returns
        if np.isnan(model_forecast['sd_forecast']):
            if hasattr(current_strat_obs, 'liquidity_ranges'):
                model_forecast['sd_forecast'] = current_strat_obs.liquidity_ranges[0]['volatility']
            else:
                model_forecast['sd_forecast'] = self.garch_data.quotePrice.pct_change().std()

        target_price = (1 + model_forecast['return_forecast']) * current_strat_obs.price

        # Set the base range
        base_range_lower = current_strat_obs.price * (
                    1 + model_forecast['return_forecast'] - self.alpha_param * model_forecast['sd_forecast'])
        base_range_upper = current_strat_obs.price * (
                    1 + model_forecast['return_forecast'] + self.alpha_param * model_forecast['sd_forecast'])

        # Set the reset range
        strategy_info_here['reset_range_lower'] = current_strat_obs.price * (
                    1 + model_forecast['return_forecast'] - self.tau_param * self.alpha_param * model_forecast[
                'sd_forecast'])
        strategy_info_here['reset_range_upper'] = current_strat_obs.price * (
                    1 + model_forecast['return_forecast'] + self.tau_param * self.alpha_param * model_forecast[
                'sd_forecast'])

        # If volatility is high enough reset range is less than zero, set at default_width of current price
        if strategy_info_here['reset_range_lower'] < 0.0:
            strategy_info_here['reset_range_lower'] = self.default_width * current_strat_obs.price

        save_ranges = []

        ###########################################################
        # STEP 2: Set Base Liquidity
        ###########################################################

        # Store each token amount supplied to pool
        total_token_0_amount = current_strat_obs.liquidity_in_0
        total_token_1_amount = current_strat_obs.liquidity_in_1

        # Lower Range
        if base_range_lower > 0.0:
            TICK_A_PRE = math.log(current_strat_obs.decimal_adjustment * base_range_lower, 1.0001)
            TICK_A = int(math.floor(TICK_A_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)
        else:
            # If lower end of base range is negative, fix at 0.0
            base_range_lower = 0.0
            TICK_A = math.ceil(
                math.log((2 ** -128), 1.0001) / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing

        # Upper Range
        TICK_B_PRE = math.log(current_strat_obs.decimal_adjustment * base_range_upper, 1.0001)
        TICK_B = int(math.floor(TICK_B_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)

        # Make sure Tick A < Tick B. If not make one tick
        if TICK_A == TICK_B:
            TICK_B = TICK_A + current_strat_obs.tickSpacing

        liquidity_placed_base = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current, TICK_A, TICK_B,
                                                               current_strat_obs.liquidity_in_0, \
                                                               current_strat_obs.liquidity_in_1,
                                                               current_strat_obs.decimals_0,
                                                               current_strat_obs.decimals_1))

        base_amount_0_placed, base_amount_1_placed = UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,
                                                                              TICK_A, TICK_B, liquidity_placed_base \
                                                                              , current_strat_obs.decimals_0,
                                                                              current_strat_obs.decimals_1)

        total_token_0_amount -= base_amount_0_placed
        total_token_1_amount -= base_amount_1_placed

        base_liq_range = {'price': current_strat_obs.price,
                          'target_price': target_price,
                          'lower_bin_tick': TICK_A,
                          'upper_bin_tick': TICK_B,
                          'lower_bin_price': base_range_lower,
                          'upper_bin_price': base_range_upper,
                          'time': current_strat_obs.time,
                          'token_0': base_amount_0_placed,
                          'token_1': base_amount_1_placed,
                          'position_liquidity': liquidity_placed_base,
                          'volatility': model_forecast['sd_forecast'],
                          'reset_time': current_strat_obs.time,
                          'return_forecast': model_forecast['return_forecast']}

        save_ranges.append(base_liq_range)

        ###########################
        # Step 3: Set Limit Position
        ############################

        limit_amount_0 = total_token_0_amount
        limit_amount_1 = total_token_1_amount

        token_0_limit = limit_amount_0 * current_strat_obs.price > limit_amount_1
        # Place singe sided highest value
        if token_0_limit:
            # Place Token 0
            limit_amount_1 = 0.0
            limit_range_lower = current_strat_obs.price
            limit_range_upper = base_range_upper
        else:
            # Place Token 1
            limit_amount_0 = 0.0
            limit_range_lower = base_range_lower
            limit_range_upper = current_strat_obs.price

        if limit_range_lower > 0.0:
            TICK_A_PRE = math.log(current_strat_obs.decimal_adjustment * limit_range_lower, 1.0001)
            TICK_A = int(math.floor(TICK_A_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)
        else:
            limit_range_lower = 0.0
            TICK_A = math.ceil(
                math.log((2 ** -128), 1.0001) / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing

        TICK_B_PRE = math.log(current_strat_obs.decimal_adjustment * limit_range_upper, 1.0001)
        TICK_B = int(math.floor(TICK_B_PRE / current_strat_obs.tickSpacing) * current_strat_obs.tickSpacing)

        if token_0_limit:
            # If token 0 in limit, make sure lower tick is above active tick
            if TICK_A <= current_strat_obs.price_tick_current:
                TICK_A = TICK_A + current_strat_obs.tickSpacing
        else:
            # In token 1 in limit, make sure upper tick is below active tick
            if TICK_B >= current_strat_obs.price_tick_current:
                TICK_B = TICK_B - current_strat_obs.tickSpacing

        # Make sure Tick A < Tick B. If not make one tick
        if TICK_A == TICK_B:
            if token_0_limit:
                TICK_A += current_strat_obs.tickSpacing
            else:
                TICK_B -= current_strat_obs.tickSpacing

        liquidity_placed_limit = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current, TICK_A, TICK_B, \
                                                                limit_amount_0, limit_amount_1,
                                                                current_strat_obs.decimals_0,
                                                                current_strat_obs.decimals_1))
        limit_amount_0_placed, limit_amount_1_placed = UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,
                                                                                TICK_A, TICK_B, \
                                                                                liquidity_placed_limit,
                                                                                current_strat_obs.decimals_0,
                                                                                current_strat_obs.decimals_1)

        limit_liq_range = {'price': current_strat_obs.price,
                           'target_price': target_price,
                           'lower_bin_tick': TICK_A,
                           'upper_bin_tick': TICK_B,
                           'lower_bin_price': limit_range_lower,
                           'upper_bin_price': limit_range_upper,
                           'time': current_strat_obs.time,
                           'token_0': limit_amount_0_placed,
                           'token_1': limit_amount_1_placed,
                           'position_liquidity': liquidity_placed_limit,
                           'volatility': model_forecast['sd_forecast'],
                           'reset_time': current_strat_obs.time,
                           'return_forecast': model_forecast['return_forecast']}

        save_ranges.append(limit_liq_range)

        # Update token amount supplied to pool
        total_token_0_amount -= limit_amount_0_placed
        total_token_1_amount -= limit_amount_1_placed

        # How much liquidity is not allcated to ranges
        current_strat_obs.token_0_left_over = max([total_token_0_amount, 0.0])
        current_strat_obs.token_1_left_over = max([total_token_1_amount, 0.0])

        # Since liquidity was allocated, set to 0
        current_strat_obs.liquidity_in_0 = 0.0
        current_strat_obs.liquidity_in_1 = 0.0

        return save_ranges, strategy_info_here

    ########################################################
    # Extract strategy parameters
    ########################################################
    def dict_components(self, strategy_observation):
        this_data = dict()

        # General variables
        this_data['time'] = strategy_observation.time
        this_data['price'] = float(strategy_observation.price)
        this_data['reset_point'] = int(strategy_observation.reset_point)
        this_data['reset_reason'] = strategy_observation.reset_reason
        this_data['volatility'] = float(strategy_observation.liquidity_ranges[0]['volatility'])
        this_data['return_forecast'] = float(strategy_observation.liquidity_ranges[0]['return_forecast'])

        # Range Variables
        this_data['base_range_lower'] = float(strategy_observation.liquidity_ranges[0]['lower_bin_price'])
        this_data['base_range_upper'] = float(strategy_observation.liquidity_ranges[0]['upper_bin_price'])
        this_data['limit_range_lower'] = float(strategy_observation.liquidity_ranges[1]['lower_bin_price'])
        this_data['limit_range_upper'] = float(strategy_observation.liquidity_ranges[1]['upper_bin_price'])
        this_data['reset_range_lower'] = float(strategy_observation.strategy_info['reset_range_lower'])
        this_data['reset_range_upper'] = float(strategy_observation.strategy_info['reset_range_upper'])
        this_data['price_at_reset'] = float(strategy_observation.liquidity_ranges[0]['price'])

        # Fee Varaibles
        this_data['token_0_fees'] = float(strategy_observation.token_0_fees)
        this_data['token_1_fees'] = float(strategy_observation.token_1_fees)
        this_data['token_0_fees_uncollected'] = float(strategy_observation.token_0_fees_uncollected)
        this_data['token_1_fees_uncollected'] = float(strategy_observation.token_1_fees_uncollected)

        # Asset Variables
        this_data['token_0_left_over'] = float(strategy_observation.token_0_left_over)
        this_data['token_1_left_over'] = float(strategy_observation.token_1_left_over)

        total_token_0 = 0.0
        total_token_1 = 0.0
        for i in range(len(strategy_observation.liquidity_ranges)):
            total_token_0 += strategy_observation.liquidity_ranges[i]['token_0']
            total_token_1 += strategy_observation.liquidity_ranges[i]['token_1']

        this_data['token_0_allocated'] = float(total_token_0)
        this_data['token_1_allocated'] = float(total_token_1)
        this_data[
            'token_0_total'] = float(total_token_0 + strategy_observation.token_0_left_over + strategy_observation.token_0_fees_uncollected)
        this_data[
            'token_1_total'] = float(total_token_1 + strategy_observation.token_1_left_over + strategy_observation.token_1_fees_uncollected)

        # Value Variables
        this_data['value_position_in_token_0'] = this_data['token_0_total'] + this_data['token_1_total'] / this_data[
            'price']
        this_data['value_allocated_in_token_0'] = this_data['token_0_allocated'] + this_data['token_1_allocated'] / \
                                                  this_data['price']
        this_data['value_left_over_in_token_0'] = this_data['token_0_left_over'] + this_data['token_1_left_over'] / \
                                                  this_data['price']

        this_data['base_position_value_in_token_0'] = strategy_observation.liquidity_ranges[0]['token_0'] + \
                                                      strategy_observation.liquidity_ranges[0]['token_1'] / this_data[
                                                          'price']
        this_data['limit_position_value_in_token_0'] = strategy_observation.liquidity_ranges[1]['token_0'] + \
                                                       strategy_observation.liquidity_ranges[1]['token_1'] / this_data[
                                                           'price']

        return this_data

