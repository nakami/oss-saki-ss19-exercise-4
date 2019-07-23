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
from framework.stock_data import StockData
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
        self.state_size = 2
        self.action_size = 10
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

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def choose_actions(self, stock_data_a: StockData, stock_data_b: StockData, portfolio: Portfolio, order_list: List[Order], epsilon=None, model_choice=None):
        assert epsilon is not None
        assert model_choice is not None

        action_a = None
        action_b = None

        if random.random() < self.epsilon:
            action_comb = random.randrange(10)
        else:
            #action_comb = model_choice
            pass

        potential_buy_a = int(portfolio.cash // stock_data_a.get_last()[-1])
        potential_buy_b = int(portfolio.cash // stock_data_b.get_last()[-1])

        potential_sell_a = portfolio.get_stock(Company.A)
        potential_sell_b = portfolio.get_stock(Company.B)

        """10 action combinations:
        - buy  100% A, buy    0% B  # buy only A completely
        - buy  100% A, sell 100% B  # sell all B, buy all A
        - buy   50% A, buy   50% B  # buy both
        - buy    0% A, buy  100% B  # buy only B completely
        - sell 100% A, sell   0% B  # sell only A completely
        - sell 100% A, sell 100% B  # sell both completely
        - sell 100% A, buy  100% B  # sell all A, buy all B
        - sell  50% A, sell  50% B  # sell both half
        - sell   0% A, sell 100% B  # sell only B completely
        - hold                      # do nothing
        """     
        logger.debug(f"{self.get_name()}: chooses action comb {action_comb}")
        if action_comb == 0:
            # buy  100% A, buy    0% B  # buy only A completely
            action_a = OrderType.BUY
            action_b = 0
            order_list.append(Order(OrderType.BUY, Company.A, potential_buy_a))
        elif action_comb == 1:
            # buy  100% A, sell 100% B  # sell all B, buy all A
            action_a = OrderType.BUY
            action_b = OrderType.SELL
            order_list.append(Order(OrderType.BUY, Company.A, potential_buy_a))
            order_list.append(Order(OrderType.SELL, Company.B, potential_sell_b))
        elif action_comb == 2:
            # buy   50% A, buy   50% B  # buy both
            action_a = OrderType.BUY
            action_b = OrderType.SELL
            order_list.append(Order(OrderType.BUY, Company.A, potential_buy_a // 2))
            remaining_cash = portfolio.cash - (potential_buy_a // 2) * stock_data_a.get_last()[-1]
            potential_buy_b = int(remaining_cash // stock_data_b.get_last()[-1])
            order_list.append(Order(OrderType.SELL, Company.B, potential_buy_b))
        elif action_comb == 3:
            # buy    0% A, buy  100% B  # buy only B completely
            action_a = 0
            action_b = OrderType.BUY
            order_list.append(Order(OrderType.BUY, Company.B, potential_buy_b))
        elif action_comb == 4:
            # sell 100% A, sell   0% B  # sell only A completely
            action_a = OrderType.SELL
            action_b = 0
            order_list.append(Order(OrderType.SELL, Company.A, potential_sell_a))
        elif action_comb == 5:
            # sell 100% A, sell 100% B  # sell both completely
            action_a = OrderType.SELL
            action_b = OrderType.SELL
            order_list.append(Order(OrderType.SELL, Company.A, potential_sell_a))
            order_list.append(Order(OrderType.SELL, Company.B, potential_sell_b))
        elif action_comb == 6:
            # sell 100% A, buy  100% B  # sell all A, buy all B
            action_a = OrderType.SELL
            action_b = OrderType.BUY
            order_list.append(Order(OrderType.SELL, Company.A, potential_sell_a))
            order_list.append(Order(OrderType.BUY, Company.B, potential_buy_b))
        elif action_comb == 7:
            # sell  50% A, sell  50% B  # sell both half
            action_a = OrderType.SELL
            action_b = OrderType.SELL
            order_list.append(Order(OrderType.SELL, Company.A, potential_sell_a // 2))
            order_list.append(Order(OrderType.SELL, Company.B, potential_sell_b // 2))
        elif action_comb == 8:
            # sell   0% A, sell 100% B  # sell only B completely
            action_a = 0
            action_b = OrderType.SELL
            order_list.append(Order(OrderType.SELL, Company.B, potential_sell_b))
        elif action_comb == 9:
            # hold                      # do nothing
            action_a = 0
            action_b = 0
        return action_a, action_b, order_list

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

        # TODO Compute the current state
        stock_data_a = stock_market_data[Company.A]
        vote_a = self.expert_a.vote(stock_data_a)
        stock_data_b = stock_market_data[Company.B]
        vote_b = self.expert_a.vote(stock_data_b)

        curr_state = np.asarray([
            #portfolio.cash,
            #portfolio.stocks[Company.A],
            #portfolio.stocks[Company.B],
            #stock_market_data.get_most_recent_price(Company.A),
            #stock_market_data.get_most_recent_price(Company.B),
            vote_a,
            vote_b,
        ])



        # TODO Store state as experience (memory) and train the neural network only if trade() was called before at least once
        # <s, a, r, s'>
        trade_called_once_before = (self.last_state is not None)
        experience = (self.last_state, self.last_action_a, self.last_action_b, self.last_portfolio_value, curr_state)
        self.memory.append(experience)
        if trade_called_once_before and self.min_size_of_memory_before_training < len(self.memory):
            # create self.batch_size-times random numbers
            # between 0 and length of queue
            # reverse the list, so pop() with indices works out
            selected_mems_ind = random.sample(range(0, len(self.memory)), self.batch_size)[::-1]
            selected_mems = [self.memory.pop(i) for i in selected_mems_ind]
            # mem[:2] -> s, a
            # mem[2] -> r
            X = [np.asarray(mem[:2]) for mem in selected_mems]
            Y = [np.asarray(mem[2]) for mem in selected_mems]
            self.model.fit(X, Y, batch_size=self.batch_size)

        # TODO Create actions for current state and decrease epsilon for fewer random actions
        # Order(OrderType.SELL, company, amount_to_sell)
        # model get suggested action
        print(f"curr_state: {curr_state}")
        print(f"curr_state.shape: {curr_state.shape}")
        predicted_actions_matrix = self.model.predict(curr_state)
        model_choice = np.argmax(predicted_actions_matrix)
        
        order_list = []
        curr_action_a, curr_action_b, order_list = self.choose_actions(stock_data_a, stock_data_b, portfolio, order_list, epsilon=self.epsilon, model_choice=model_choice)
        curr_portfolio_value = portfolio.get_value(stock_market_data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # TODO Save created state, actions and portfolio value for the next call of trade()
        self.last_state = curr_state
        self.last_action_a = curr_action_a
        self.last_action_b = curr_action_b
        self.last_portfolio_value = curr_portfolio_value

        return order_list

# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

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
