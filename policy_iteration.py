import numpy as np


class PolicyIteration(object):
    def __init__(self, world, one_step_cost_fn, discount_factor=0.9):
        """
        Ctor for policy iteration algorithm implementation.

        :param world: Instance of class 'World'
        :param one_step_cost_fn: Method that returns the one step cost given an instance of World, the current position.
        and the next position
        :param discount_factor: In range [0, 1)
        """
        self.world = world
        self.discount_factor = discount_factor
        self.one_step_cost_fn = lambda current_pos, next_pos: one_step_cost_fn(self.world, current_pos, next_pos)
        self.policy = None

        self.value_fn_history = list()

    def init_random_policy(self):
        """
        Initializes an arbitrary but valid policy
        :return: Correctly initialized policy
        """

        policy = np.zeros((self.world.world_height, self.world.world_width, 2), np.int32)
        for y in range(self.world.world_height):
            for x in range(self.world.world_width):
                state = np.array([x, y], np.int32)
                if self.world.is_wall(state):
                    continue

                # iterate through list of actions and set the policy to the first permitted action
                for action in self.world.actions:
                    if self.world.is_action_allowed(state, action):
                        policy[y, x] = action
                        break
        return policy

    def has_policy_changed(self, policy):
        return not np.allclose(self.policy, policy, atol=10e-8)

    def evaluate(self):
        """
        Evaluates the current policy using a linear system of equations
        :return: Cost of the current policy
        """

        A = np.zeros((self.state_cardinality, self.state_cardinality), np.float32)
        b = np.zeros(self.state_cardinality, np.float32)

        next_free_idx = 0
        idx_mapping = list()

        def mapping_exists(state_to_find):
            for i, s in enumerate(idx_mapping):
                if np.array_equal(s, state_to_find):
                    return i
            return -1

        for y in range(self.world.world_height):
            for x in range(self.world.world_width):
                state = np.array([x, y], np.int32)
                if self.world.is_wall(state):
                    continue

                # check if idx mapping for next_state already exists
                retrieved_idx = mapping_exists(state)
                if retrieved_idx != -1:
                    state_idx = retrieved_idx
                else:
                    state_idx = next_free_idx
                    idx_mapping.append(state)
                    next_free_idx += 1

                action = self.policy[y, x]
                transitions = self.world.get_transitions(state, action)
                assert transitions  # non-empty
                A[state_idx, state_idx] = 1.

                for next_state, transition_prob in transitions:
                    # check if idx mapping for next_state already exists
                    retrieved_idx = mapping_exists(next_state)
                    if retrieved_idx != -1:
                        next_state_idx = retrieved_idx
                    else:
                        next_state_idx = next_free_idx
                        idx_mapping.append(next_state)
                        next_free_idx += 1

                    b[state_idx] += transition_prob * self.one_step_cost_fn(state, next_state)
                    A[state_idx, next_state_idx] -= transition_prob * self.discount_factor

        # clip A and b
        A = A[:next_free_idx, :next_free_idx]
        b = b[:next_free_idx]

        # solve the linear system
        solution = np.linalg.solve(A, b)

        # convert the flattened state vector back into a 2D array corresponding to the world map
        value_fn = np.zeros(self.world.map.shape, np.float32)
        for i, (x, y) in enumerate(idx_mapping):
            value_fn[y, x] = solution[i]

        self.value_fn_history.append(value_fn)
        return value_fn

    def improve(self, value_fn):
        """
        Performs a single policy improvement step given the value function of the current policy.
        :param value_fn: Value of the current policy (as computed by the 'evaluate' method
        :return: None
        """

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
                                                      (self.discount_factor * value_fn[next_state[1], next_state[0]]))
                                   for next_state, transition_prob in transitions])

                    if cost_fn < min_cost:
                        min_cost = cost_fn
                        min_cost_action = action

                if min_cost_action is None:
                    raise ValueError("No feasible action found for state (%d, %d)" % (x, y))

                self.policy[y, x] = min_cost_action

    def execute(self, max_iterations=int(10e6)):
        """
        Starts the policy iteration
        :param max_iterations: Maximum allowable number of iterations
        :return: The value function of the policy at the last iteration
        """

        self.policy = self.init_random_policy()
        value_fn = None

        has_converged = False
        for i in range(max_iterations):
            # policy evaluation
            value_fn = self.evaluate()

            # policy improvement
            prev_policy = np.copy(self.policy)
            self.improve(value_fn)

            # check for convergence
            if not self.has_policy_changed(prev_policy):
                print("Policy iteration has converged after %d iterations" % (i + 1))
                has_converged = True
                break

        if not has_converged:
            print("ERROR: Policy iteration did not converge after %d iterations" % max_iterations)

        return value_fn

    state_cardinality = property(fget=lambda self: self.world.world_width * self.world.world_height)
