# A

```
# Parameters for neural network
self.state_size = 2
self.action_size = 4
self.hidden_size = 50

# Parameters for deep Q-learning
self.learning_rate = 0.001
self.epsilon_min = 0.01
self.batch_size = 16
self.min_size_of_memory_before_training = 1000
self.memory = deque(maxlen=2000)
```

## baseline

name                                        final portfolio value (thousand)
----------------------------------------------------------------------------
buy_and_hold_trader                         22.3
trusting_trader                              3.7

## parameter exploration

name                                        final portfolio value (thousand)
----------------------------------------------------------------------------
deep_q_learning_trader_eps0_gamma0           4.4
deep_q_learning_trader_eps0_gamma10         47.5
deep_q_learning_trader_eps0_gamma50         51.8
deep_q_learning_trader_eps0_gamma90         13.6
deep_q_learning_trader_eps0_gamma100        17.1