if __name__ == "__main__":
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 

import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 1
        self.action_size = 2
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action_a = None
        self.last_action_b = None
        self.last_portfolio_value = None
        self.last_price_a = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.vote_map = {
            Vote.SELL : 0,
            Vote.HOLD : 1,
            Vote.BUY : 2,
        }

        self.vote_map_inverse = {v: k for k, v in self.vote_map.items()}

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        # INPUT layer:  1 (buy or sell A?)
        # output layer: 2 ([buy_A, sell_A])

        # TODO Compute the current state
        stock_data_a = stock_market_data[Company.A]
        vote_a_for_a = self.expert_a.vote(stock_data_a)
        vote_b_for_a = self.expert_b.vote(stock_data_a)
        #stock_data_b = stock_market_data[Company.B]
        #vote_b = self.expert_a.vote(stock_data_b)
        state = np.array([[self.vote_map[vote_a_for_a] + self.vote_map[vote_b_for_a]]])#, self.vote_map[vote_b]])

        # do action 0 or 1?
        predictions = self.model.predict(state)
        action = np.argmax(predictions)

        current_price_a = stock_market_data.get_most_recent_price(Company.A)
        order_list = []

        if action == 0:
            # buy all A
            amount_to_buy = portfolio.cash // current_price_a
            if amount_to_buy > 0:
                order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy))
        elif action == 1:
            # sell all A
            amount_to_sell = portfolio.get_stock(Company.A)
            if amount_to_sell > 0:
                order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell))
        else:
            assert False

        if self.last_state is not None:
            # train
            diff_a = (current_price_a / self.last_price_a - 1)
            if self.last_action_a == 0:
                rec_vec = np.array([[diff_a, -diff_a]])
            elif self.last_action_a == 1:
                rec_vec = np.array([[diff_a, -diff_a]])
            else:
                assert False # wtf
            #reward_vec = np.array([[portfolio.get_value(stock_market_data)]])
            self.model.fit(self.last_state, rec_vec)
        
        self.last_state = state
        self.last_action_a = action
        self.last_portfolio_value = portfolio.get_value(stock_market_data)
        self.last_price_a = current_price_a

        return order_list

        # TODO Store state as experience (memory) and train the neural network only if trade() was called before at least once

        # TODO Create actions for current state and decrease epsilon for fewer random actions

        # TODO Save created state, actions and portfolio value for the next call of trade()


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 1
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    #training_data = training_data.deepcopy_first_n_items(6000)
    # 12588
    #print(f'training_data[Company.A].get_row_count(): {training_data[Company.A].get_row_count()}')

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()
