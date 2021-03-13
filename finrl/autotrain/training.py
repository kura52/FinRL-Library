import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime
import pickle

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats


def train_one():
    """
    train an agent
    """
    logging.info("==============Start Fetching Data===========")

    do_download = True
    do_feature = True
    do_train = True

    if do_download:
        df = YahooDownloader(
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            ticker_list=config.TICKERS,
        ).fetch_data()
        df.to_csv('df.txt')
    else:
        df = pd.read_csv('df.txt')

    logging.info("==============Start Feature Engineering===========")
    if do_feature:
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=True,
            user_defined_feature=True,
        )
        with open('fe.pickle', 'wb') as f: pickle.dump(fe, f)
        #processed = fe.preprocess_data(df)

        df_group = df.groupby(df.tic)
        processed = pd.concat([fe.preprocess_data(df_group.get_group(tic)) for tic in config.TICKERS]).sort_index()
        with open('processed.pickle', 'wb') as f: pickle.dump(processed, f)
    else:
        with open('fe.pickle', 'rb') as f: fe = pickle.load(f)
        with open('processed.pickle', 'rb') as f: processed = pickle.load(f)

    # Training & Trading data split
    train = data_split(processed, config.START2_DATE, config.START_TRADE_DATE)
    # trade = data_split(processed, config.START2_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)

    train.to_csv('train.txt')
    trade.to_csv('trade.txt')

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        len(config.TECHNICAL_INDICATORS_LIST + config.USER_DEFINED_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 1000000,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST + config.USER_DEFINED_LIST,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
        }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    logging.info("==============Model Training===========")

    #for i in range(1, 100):
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    model_sac = agent.get_model("ppo")

    if do_train:
        trained_sac = agent.train_model(
            model=model_sac, tb_log_name="ppo", total_timesteps=100000
        )
        # trained_sac.save_replay_buffer("trained_sac")
    else:
        trained_sac = model_sac
        trained_sac.load_replay_buffer("trained_sac")

    logging.info("==============Start Trading===========")
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, is_prediction=True, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, environment=e_trade_gym
    )
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    logging.info("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")
    # backtest_plot(account_value=df_account_value, baseline_ticker=config.TECHNICAL_INDICATORS_LIST[0])
