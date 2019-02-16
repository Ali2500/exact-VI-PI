# Exact Value and Policy Iteration

A basic implementation of exact value and policy iteration for a stochastic agent in a 2D grid world.

The grid world is defined in `world_map.txt`. The agent can execute one of five actions: move left, up, right, down, or stay idle. Depending on the state in which the agent is, one or more of these actions may not be permitted due to the presence of walls.

### Stochasticity Model

For any permitted action, the agent may move to the target location, or to locations adjascent to the target location. This is illustrated below:

![alt text](https://i.imgur.com/Hw8XPtH.png "Stochasticity model of agent")

If both locations adjascent to the target location are free, then the agent moves with probability 0.8 to the target, and with 0.1 to each of the adjascent locations. If one of the two adjascent locations is occupied by a wall, then the probability of transitioning to the target increases to 0.9. Similarly, if both adjascent locations are occupied by walls, then the transition becomes deterministic.

## Value Iteration

The implementation of exact value iteration is done using the optimal Bellman operator:

![alt text](https://i.imgur.com/sfQ5clD.gif)

where the following notation is used:

- `V(s)` is the value of state `s` which will be iteratively refined
- `u` is one of the five actions described earlier
- `g(s,u,s')` is the one-step cost. In other words, it is the cost incurred when the agent is in state `s`, performs action `u` and transitions to state `s'`.
- &#611; is the discount factor.
- `p(s,u,s')` is the probability of transitioning from state `s` to `s'` under action `u`.

The value function is initialized to all zeros, and then iteratively refined. The above defined Bellman operator is repeatedly applied to every state in the 2D grid world. Refer to the code in `value_iteration.py` for the implementation. Theoretically, the Bellman operator is guaranteed to converge to the optimal value function after infinitely many iterations. Practically however, we run the algorithm until the different in values for all states drops below a certain threshold and then terminate.

## Policy Iteration

The implementation of exact policy iteration is as follows: we first initialize a random (but valid) policy. Thereafter, the following two steps are repeatedly executed in a loop:

**1) Policy Evaluation:** The value function for the current policy is evaluated by solving a linear system of equations. Consider the following Bellman operator (same as the optimal Bellman operator, but without the minimization):

![alt text](https://i.imgur.com/h4dx9FN.gif)

Here, &#956; denotes the current policy. Plainly put, &#956; can be viewed as a function which is given a state and produces an action. So &#956;(s) gives us an action that should be executed when the agent is in state `s`.

In the above equation, we want to determine the value function `V`. This can be done efficiently by formulating the problem as a system of linear equations. This allows us to evaluate the policy for all states by creating a matrix equation. Refer to the `evaluate` method in `policy_iteration.py` for the implementation of this.

**2) Policy Improvement:** The current policy is revised based on the updated value function:

![alt text](https://i.imgur.com/j6ljqtE.gif)

This is basically a greedily induced policy: given our current estimate of the value function, we choose a policy that minimizes the expected future cost for every state `s`. Refer to the `improve` method in `policy_iteration.py` for implementation of this step.

Similar to value iteration, policy iteration is also guaranteed to converge to the optimal policy which minimizes the future cost for every state. However, policy iteration has an added advantage: since the total number of policies is finite (because the set of states and set of actions are both finite), it is possible to exactly determine the optimal policy. This is in contrast to value iteration, where we set a convergence threshold to determine when to terminate.

The above mentioned loop of evaluating and improving the policy thus terminates when the updated policy at `t` is the same as the policy at `t-1`.

## Execution

Simply run the the `main.py` script (no options required). It will use the world definition in `world_map.txt` and a discount factor (&#611;) of 0.9. It will execute both value and policy iteration for the given problem, and create two plots at the end (one for each). These plots illustrate the grid world with the arrows denoting the final policy for each state, and the colors of the cells denoting the final costs for each state. Cells which are occupied by walls are colored in black.

I have provided two choices for the one-step cost function (see `main.py`). You can also define your own methods for the one-step cost and just pass them as arguments to `PolicyIteration` and `ValueIteration`.