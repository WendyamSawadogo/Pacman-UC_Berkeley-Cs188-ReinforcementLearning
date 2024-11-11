# Project: Value Iteration and Q-Learning in Pacman's World

## Introduction

In this project, we implement the **Value Iteration** algorithm and the **Q-Learning** algorithm to enable Pacman to make optimal decisions in various environments. We first test our agents on the **Gridworld**, then apply them to a simulated robot controller (**Crawler**), and finally to **Pacman**.

The primary goal is to understand and implement reinforcement learning methods for solving sequential decision-making problems under uncertainty.

## Files Modified

- **`valueIterationAgents.py`**: Contains the implementation of the Value Iteration agent for solving known MDPs.
- **`qlearningAgents.py`**: Contains the implementation of Q-Learning agents for Gridworld, Crawler, and Pacman.
- **`analysis.py`**: Contains answers to analysis questions posed in the project.

## Implementation Details

### Value Iteration Agent

- **File**: `valueIterationAgents.py`
- **Class**: `ValueIterationAgent`
- **Description**: An agent that performs value iteration to compute the optimal policy for a given MDP.

- **Methods Implemented**:
  - `__init__(self, mdp, discount=0.9, iterations=100)`: Initializes the agent and runs value iteration.
  - `computeQValueFromValues(self, state, action)`: Computes the Q-value of an action in a state.
  - `computeActionFromValues(self, state)`: Computes the best action to take in a state based on current values.

- **Approach**:
  - Use the **batch** version of value iteration.
  - For a given number of iterations, update the value of each state based on the Bellman equation.

- **Code Snippet**:

  ```python
  class ValueIterationAgent(ValueEstimationAgent):
      def __init__(self, mdp, discount=0.9, iterations=100):
          self.mdp = mdp
          self.discount = discount
          self.iterations = iterations
          self.values = util.Counter()  # A Counter is a dict with default 0

          # Run value iteration
          for _ in range(iterations):
              new_values = util.Counter()
              for state in mdp.getStates():
                  if mdp.isTerminal(state):
                      new_values[state] = 0
                  else:
                      q_values = []
                      for action in mdp.getPossibleActions(state):
                          q_value = sum(
                              prob * (mdp.getReward(state, action, next_state) +
                              self.discount * self.values[next_state])
                              for next_state, prob in mdp.getTransitionStatesAndProbs(state, action)
                          )
                          q_values.append(q_value)
                      new_values[state] = max(q_values)
              self.values = new_values
  ```

# Q-Learning Agent

**File**: `qlearningAgents.py`

**Class**: `QLearningAgent`

**Description**: An agent that learns Q-values through interaction with the environment.

**Methods Implemented**:

- `getQValue(self, state, action)`: Returns the Q-value for a state-action pair.
- `computeValueFromQValues(self, state)`: Returns the maximum Q-value for a state.
- `computeActionFromQValues(self, state)`: Computes the best action to take in a state.
- `getAction(self, state)`: Selects an action using an epsilon-greedy policy.
- `update(self, state, action, nextState, reward)`: Updates the Q-value based on observed transition.

**Approach**:

- Initialize Q-values to zero.
- Use an epsilon-greedy policy for action selection.
- Update Q-values using the Q-learning update rule.

**Code Snippet**:

```python
class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.qValues = {}

    def getQValue(self, state, action):
        return self.qValues.get((state, action), 0.0)

    def computeValueFromQValues(self, state):
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            return max(self.getQValue(state, action) for action in possibleActions)
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        possibleActions = self.getLegalActions(state)
        if not possibleActions:
            return None
        max_value = self.computeValueFromQValues(state)
        best_actions = [action for action in possibleActions if self.getQValue(state, action) == max_value]
        return random.choice(best_actions)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
  ```
# Approximate Q-Learning Agent

**File**: `qlearningAgents.py`

**Class**: `ApproximateQAgent`

**Description**: An agent that approximates Q-values using feature extraction.

**Methods Implemented**:

- `getQValue(self, state, action)`: Returns the Q-value as a dot product of weights and features.
- `update(self, state, action, nextState, reward)`: Updates the weights based on the observed transition.

**Approach**:

- Use feature extractors to represent state-action pairs.
- Maintain weights for each feature.
- Update weights based on the difference between estimated and actual rewards.

**Code Snippet**:

```python
class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        super().__init__(**args)
        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[feat] * features[feat] for feat in features)

    def update(self, state, action, nextState, reward):
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feat in features:
            self.weights[feat] += self.alpha * difference * features[feat]
```
**Analysis Questions**
**File**: analysis.py

**Description**: Contains answers to specific questions about MDP parameters.

**Sample Answers**:
```python
def question2():
answerDiscount = 0.9
answerNoise = 0.0
return answerDiscount, answerNoise

def question3a():
answerDiscount = 0.5
answerNoise = 0.2
answerLivingReward = -4
return answerDiscount, answerNoise, answerLivingReward

def question6():
answerEpsilon = 0.5
answerLearningRate = 10
return answerEpsilon, answerLearningRate
```


# Observations and Results
Value Iteration Agent
- Successfully computes optimal policies for known MDPs
- Uses the batch version of value iteration to ensure correctness

Q-Learning Agent
- Learns optimal policies through exploration and exploitation
- Implements epsilon-greedy action selection to balance exploration and exploitation

**Approximate Q-Learning Agent
- Generalizes across states using feature representation
- Efficiently learns in larger state spaces where tabular Q-learning is infeasible

**Running the Code
**Value Iteration in Gridworld
```bash
python gridworld.py -a value -i 100 -k 10
```

**Q-Learning in Gridworld
```bash
python gridworld.py -a q -k 50
```

**Pacman with Q-Learning
```bash
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

**Approximate Q-Learning in Pacman
```bash
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

**Conclusion
By completing this project, we have:
- Implemented the Value Iteration algorithm to compute optimal policies in known MDPs
- Developed a Q-Learning Agent that learns optimal policies through interaction with the environment
- Extended our agent to an Approximate Q-Learning Agent that uses feature-based representations to handle larger state spaces

