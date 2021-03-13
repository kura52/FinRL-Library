import logging
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 stock_dim,
                 hmax,
                 initial_amount,
                 buy_cost_pct,
                 sell_cost_pct,
                 reward_scaling,
                 state_space,
                 action_space,
                 tech_indicator_list,
                 turbulence_threshold=None,
                 make_plots=False,
                 print_verbosity=10,
                 day=0,
                 initial=True,
                 previous_state=[],
                 model_name='',
                 mode='',
                 iteration='',
                 is_prediction=False):
        self.day = day
        self.df = df

        self.tic_len = len(self.df.tic.unique())
        self.ind_len = len(self.df.index.unique())

        self.data_dic = {}
        self.data_date_dic = {}
        self.data_close_dic = {}
        self.state_dic = {}
        for ind in self.df.index.unique():
            self.data_dic[ind] = self.df.loc[ind]
            if self.tic_len > 1:
                self.data_date_dic[ind] = self.df.loc[ind].date.values[0]
                self.data_close_dic[ind] = self.data_dic[ind].close.to_numpy()
                self.state_dic[ind] = np.concatenate([self.data_dic[ind][tech].values for tech in tech_indicator_list])
            else:
                self.data_date_dic[ind] = self.df.loc[ind].date
                self.data_close_dic[ind] = self.data_dic[ind].close
                self.state_dic[ind] = np.array([self.data_dic[ind][tech] for tech in tech_indicator_list])

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.data = self.data_dic[self.day]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.previous_position = []
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.is_prediction = is_prediction

        # initalize state
        self.state, self.position = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount + sum(self.position[1:(self.stock_dim+1)]*self.position[(self.stock_dim+1):(self.stock_dim*2+1)])]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.reset()
        self._seed()

        self.future_num = 22

        self.sum_dic = {}
        for ind in self.df.index.unique():
            d = df.loc[ind, 'date'].values[0] if self.tic_len > 1 else df.loc[ind, 'date']
            self.sum_dic[d] = df.loc[ind+1:ind + self.future_num, ['tic', 'close']].groupby(
                'tic').apply(np.average).to_numpy()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.position[index+1]>0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.position[index+self.stock_dim+1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action // self.position[index+1]),self.position[index+self.stock_dim+1])
                    sell_amount = self.position[index+1] * sell_num_shares * (1- self.sell_cost_pct)
                    #update balance
                    self.position[0] += sell_amount

                    self.position[index+self.stock_dim+1] -= sell_num_shares
                    self.cost +=self.position[index+1] * sell_num_shares * self.sell_cost_pct
                    self.trades+=1
                    # logger.info(f"sell: {self.trades}")
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence>=self.turbulence_threshold:
                if self.position[index+1]>0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.position[index+self.stock_dim+1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = min(abs(action // self.position[index+1]),self.position[index+self.stock_dim+1])
                        sell_amount = self.position[index+1]*sell_num_shares* (1- self.sell_cost_pct)
                        #update balance
                        self.position[0] += sell_amount
                        self.position[index+self.stock_dim+1] =0
                        self.cost += self.position[index+1]*self.position[index+self.stock_dim+1]* \
                                    self.sell_cost_pct
                        self.trades+=1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares


    def _buy_stock(self, index, action):

        def _do_buy():
            if self.position[index+1]>0:
                #買える量 Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.position[0] // self.position[index+1]
                # logging.info('available_amount:{}'.format(available_amount))

                #update balance
                buy_num_shares = min(available_amount, action // self.position[index+1])
                buy_amount = self.position[index+1] * buy_num_shares * (1+ self.buy_cost_pct)
                self.position[0] -= buy_amount

                self.position[index+self.stock_dim+1] += buy_num_shares

                self.cost+=self.position[index+1] * buy_num_shares * self.buy_cost_pct
                self.trades+=1
                # logger.info(f"buy: {self.trades}")
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence< self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory,'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def step(self, actions):
        if self.is_prediction:
            self.terminal = self.day >= self.ind_len-1
        else:
            self.terminal = self.day >= self.ind_len-1-self.future_num

        if self.terminal:
            # logging.info(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.position[0]+ \
                sum(np.array(self.position[1:(self.stock_dim+1)])*np.array(self.position[(self.stock_dim+1):(self.stock_dim*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = self.position[0]+sum(np.array(self.position[1:(self.stock_dim+1)])*np.array(self.position[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.initial_amount
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)
            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                logging.info(f"day: {self.day}, episode: {self.episode}")
                logging.info(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                logging.info(f"end_total_asset: {end_total_asset:0.2f}")
                logging.info(f"total_reward: {tot_reward:0.2f}")
                logging.info(f"total_cost: {self.cost:0.2f}")
                logging.info(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() != 0:
                    logging.info(f"Sharpe: {sharpe:0.3f}")
                logging.info("=================================")

            if (self.model_name!='') and (self.mode!=''):
                df_actions = self.save_action_memory()
                df_actions.to_csv('results/actions_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration))
                df_total_value.to_csv('results/account_value_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                df_rewards.to_csv('results/account_rewards_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.plot(self.asset_memory,'r')
                plt.savefig('results/account_value_{}_{}_{}.png'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.close()

            # Add outputs to logger interface
            logger.record("environment/portfolio_value", end_total_asset)
            logger.record("environment/total_reward", tot_reward)
            logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            logger.record("environment/total_cost", self.cost)
            logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}
        else:

            actions = actions * self.hmax #actions initially is scaled between 0 to 1
            actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-self.hmax]*self.stock_dim)

            # 現在価格っぽい
            begin_total_asset = self.position[0] + \
                                sum(self.position[1:(self.stock_dim + 1)] *
                                    self.position[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            current_pos = self.position[(self.stock_dim+1):(self.stock_dim*2+1)].copy()

            #logging.info("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # logging.info(f"Num shares before: {self.position[index+self.stock_dim+1]}")
                # logging.info(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # logging.info(f'take sell action after : {actions[index]}')
                # logging.info(f"Num shares after: {self.position[index+self.stock_dim+1]}")
            for index in buy_index:
                # logging.info('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])
            self.actions_memory.append(actions)

            self.day += 1
            self.data = self.data_dic[self.day]
            if self.turbulence_threshold is not None:
                if isinstance(self.data['turbulence'], np.float64):
                    self.turbulence = self.data['turbulence']
                else:
                    self.turbulence = self.data['turbulence'].values[0]
            self.state, self.position = self._update_state()

            after_pos = self.position[(self.stock_dim+1):(self.stock_dim*2+1)]
            change_pos = np.array(after_pos) - np.array(current_pos)

            my_reward = self.calc_reward(change_pos)

            logger.info(f"current_pos: {current_pos}, after_pos: {after_pos}, change_pos: {change_pos}, my_reward: {my_reward}")

            end_total_asset = self.position[0] + \
                              sum(self.position[1:(self.stock_dim + 1)] *
                                  self.position[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.reward = my_reward
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling

        logger.info(f"end_total_asset: {end_total_asset}, reward: {self.reward}, position: {self.position}, state: {self.state}")
        return self.state, self.reward, self.terminal, {}

    def calc_reward(self, change_pos):
        # 将来の利益を計算
        if self.is_prediction:
            my_reward = 0
        else:
            if self.tic_len > 1:
                future_vals = self.sum_dic[self._get_date()]
                close = self.data_close_dic[self.day]
                my_reward = ((future_vals - close) / close * 100 * change_pos).sum()
            else:
                future_vals = self.sum_dic[self._get_date()][0]
                my_reward = (future_vals - self.data.close) / self.data.close * 100 * change_pos[0]
        return my_reward

    def reset(self):
        #initiate state
        self.state, self.position = self._initiate_state()

        # if self.initial:
        #     self.asset_memory = [self.initial_amount]
        # else:
        #     previous_total_asset = self.previous_state[0]+ \
        #     sum(np.array(self.position[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))
        #     self.asset_memory = [previous_total_asset]
        self.asset_memory = [self.initial_amount + sum(self.position[1:(self.stock_dim+1)]*self.position[(self.stock_dim+1):(self.stock_dim*2+1)])]

        self.day = 0
        self.data = self.data_dic[self.day]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]

        self.episode+=1

        return self.state

    def render(self, mode='human',close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if self.tic_len > 1:
                # for multiple stock
                # 1つ目が保持数？、次にN株分の価格(a,b)、次にN株の0(a,b)、テクニカルの列挙（sumはflatten)(a_t1,b_t1, a_t2,b_t2,...)
                state = self.state_dic[self.day]

                position = np.concatenate([[self.initial_amount],
                                          self.data.close.values,
                                          (0 // self.data.close.values // self.stock_dim)])
            else:
                # for single stock
                state = self.state_dic[self.day]

                position = np.array([self.initial_amount,
                                          self.data.close,
                                          (0 // self.data.close // self.stock_dim)])
                          # [self.initial_amount // self.data.close // self.stock_dim]
        # else:
        #     # Using Previous State
        #     if self.tic_len > 1:
        #         # for multiple stock
        #         state = [self.previous_state[0]] + \
        #                 self.data.close.values.tolist() + \
        #                 self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] + \
        #                 sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
        #     else:
        #         # for single stock
        #         state = [self.previous_state[0]] + \
        #                 [self.data.close] + \
        #                 self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] + \
        #                 sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
        return state, position

    def _update_state(self):
        state = self.state_dic[self.day]

        if self.tic_len > 1:
            position = np.concatenate([[self.position[0]],
                                      self.data.close.values,
                                      self.position[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]])
        else:
            position = np.array([self.position[0],
                                self.data.close,
                                self.position[2]])
        return state, position

    def _get_date(self):
        return self.data_date_dic[self.day]

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #logging.info(len(date_list))
        #logging.info(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if self.tic_len>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs