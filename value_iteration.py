import numpy as np


class ValueIteration(object):
    def __init__(self, world, one_step_cost_fn, discount_factor=0.9, eps=10e-4):
        """
        Ctor for value iteration algorithm implementation.

        :param world: Instance of class 'World'
        :param one_step_cost_fn: Method that returns the one step cost given an instance of World, the current position.
        :param discount_factor: In range [0, 1)
        :param eps: Threshold for value function convergence
        """
        self.world = world
        self.discount_factor = discount_factor
        self.eps = eps
        self.one_step_cost_fn = lambda current_pos, next_pos: one_step_cost_fn(self.world, current_pos, next_pos)

        self.value_fn = np.zeros(self.world.map.shape, np.float32)
        self.value_fn_history = list()

    def execute(self, max_iterations=int(10e6)):
        """
        Starts the value iteration algorithm
        :param max_iterations: Maximum allowable number of iterations
        :return: None
        """

        for k in range(max_iterations):
            prev_value_fn = np.copy(self.value_fn)
            self.value_fn_history.append(prev_value_fn)

            for y in range(self.world.world_height):
                for x in range(self.world.world_width):
                    state = np.array([x, y], np.int32)
                    if self.world.is_wall(state):
                        continue

                    min_cost = 10e6

                    for action in self.world.actions:
                        if not self.world.is_action_allowed(state, action):
                            continue

                        transitions = self.world.get_transitions(state, action)

                        cost_fn = sum([transition_prob * (self.one_step_cost_fn(state, next_state) +
                                                          (self.discount_factor * prev_value_fn[next_state[1], next_state[0]]))
                                      for next_state, transition_prob in transitions])

                        min_cost = min(min_cost, cost_fn)

                    if min_cost == 10e6:
                        raise ValueError("No action could be applied on state (%d, %d). This state is probably "
                                         "surrounded by walls on all sides.")

                    self.value_fn[y, x] = min_cost

            if np.sum((np.absolute(self.value_fn - prev_value_fn) > self.eps).astype(np.int32)) == 0:
                print("Value iteration has converged after %d iterations" % (k + 1))
                break

        self.value_fn_history.append(self.value_fn)

    def extract_policy(self):
        """
        Computes the greedily induced policy from the current value function
        :return: The greedy policy for the current value function.
        """

        optimal_policy = np.zeros([self.world.world_height, self.world.world_width, 2], np.float32)

        for y in range(self.world.world_height):
            for x in range(self.world.world_width):
                state = np.array([x, y], np.int32)
                if self.world.is_wall(state):
                    continue

                min_cost = 10e6
                min_cost_action = None

                for action in self.world.actions:
                    if not self.world.is_action_allowed(state, action):
                        continue

                    transitions = self.world.get_transitions(state, action)

                    cost_fn = sum([transition_prob * (self.one_step_cost_fn(state, next_state) +
                                                      (self.discount_factor * self.value_fn[next_state[1], next_state[0]]))
                                   for next_state, transition_prob in transitions])

                    if cost_fn < min_cost:
                        min_cost = cost_fn
                        min_cost_action = action

                if min_cost_action is None:
                    raise ValueError("No action found for state (%d, %d)", x, y)
                else:
                    optimal_policy[y, x] = min_cost_action

        return optimal_policy
