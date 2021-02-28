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
# from finrl.trade.backtest import BackTestStats


def train_one():
    """
    train an agent
    """
    logging.info("==============Start Fetching Data===========")

    # df = YahooDownloader(
    #     start_date=config.START_DATE,
    #     end_date=config.END_DATE,
    #     ticker_list=config.DOW_30_TICKER,
    # ).fetch_data()
    # df.to_csv('df.csv')

    df = pd.read_csv('df.txt')

    logging.info("==============Start Feature Engineering===========")
    # fe = FeatureEngineer(
    #     use_technical_indicator=True,
    #     tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    #     use_turbulence=True,
    #     user_defined_feature=False,
    # )
    # with open('fe.pickle', 'wb') as f: pickle.dump(fe, f)
    with open('fe.pickle', 'rb') as f: fe = pickle.load(f)

    # processed = fe.preprocess_data(df)
    # with open('processed.pickle', 'wb') as f: pickle.dump(processed, f)
    with open('processed.pickle', 'rb') as f: processed = pickle.load(f)

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)

    train.to_csv('train.txt')
    trade.to_csv('trade.txt')

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
        }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    logging.info("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    model_sac = agent.get_model("sac")
    trained_sac = agent.train_model(
        model=model_sac, tb_log_name="sac", total_timesteps=80000
    )
    with open('trained_sac.pickle', 'wb') as f: pickle.dump(trained_sac, f)
    with open('trained_sac.pickle', 'rb') as f: trained_sac = pickle.load(f)

    logging.info("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, environment=e_trade_gym
    )
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    logging.info("==============Get Backtest Results===========")
    # perf_stats_all = BackTestStats(df_account_value)
    # perf_stats_all = pd.DataFrame(perf_stats_all)
    # perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")
