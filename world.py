import numpy as np


class World(object):
    def __init__(self, world_map_path):
        """
        Ctor for World class.

        :param world_map_path: Full path to text file containing definition of the 2D world maze.
        """
        self.map          = None

        self.__start_x    = -1
        self.__start_y    = -1
        self.__goal_x     = -1
        self.__goal_y     = -1
        self.__trap_x     = -1
        self.__trap_y     = -1

        self.WALL_TOKEN   = '1'
        self.FREE_TOKEN   = '0'
        self.GOAL_TOKEN   = 'G'
        self.TRAP_TOKEN   = 'T'
        self.START_TOKEN  = 'S'

        self.ACTION_UP    = np.array([0, -1], dtype=np.int32)
        self.ACTION_RIGHT = np.array([1, 0],  dtype=np.int32)
        self.ACTION_DOWN  = np.array([0, 1],  dtype=np.int32)
        self.ACTION_LEFT  = np.array([-1, 0], dtype=np.int32)
        self.ACTION_IDLE  = np.array([0, 0],  dtype=np.int32)

        self.action_stochasticity = 0.1

        self.read_world_map(world_map_path)

    def read_world_map(self, filepath):
        """
        Parses the world maze definition.

        :param filepath: Full path to text file containing definition of the 2D world maze.
        :return: None
        """
        with open(filepath) as readfile:
            content = readfile.readlines()

        map_width = -1
        world_map = list()
        content_start_row = -1

        for i, line in enumerate(content):
            line = line.rstrip()
            if line.startswith('#') or line is None:  # skip commented and empty lines at the start of file
                continue

            if content_start_row == -1:
                content_start_row = i

            tokens = line.split(' ')
            if map_width != len(tokens) and map_width != -1:
                raise ValueError("All lines in the file should have the same number of tokens")

            map_width = len(tokens)
            world_map.append(tokens)

            for j, token in enumerate(tokens):
                if token == self.START_TOKEN:
                    self.__start_x = j
                    self.__start_y = i - content_start_row
                elif token == self.GOAL_TOKEN:
                    self.__goal_x = j
                    self.__goal_y = i - content_start_row
                elif token == self.TRAP_TOKEN:
                    self.__trap_x = j
                    self.__trap_y = i - content_start_row

        self.map = np.array(world_map)

    def get_residual_transitions(self, action):
        assert (action != self.ACTION_IDLE).all()
        perturbation = (action == 0).astype(np.int32)
        return action + perturbation, action - perturbation

    def is_wall(self, pos):
        return self.map[pos[1], pos[0]] == self.WALL_TOKEN

    def is_within_bounds(self, pos):
        return 0 <= pos[0] < self.map.shape[1] and 0 <= pos[1] < self.map.shape[0]

    def is_action_allowed(self, current_pos, action):
        new_pos = current_pos + action
        return self.is_within_bounds(new_pos) and not self.is_wall(new_pos)

    def get_transitions(self, current_pos, action):
        if self.is_wall(current_pos):
            raise ValueError("The provided current position (%d, %d) is occupied by a wall" %
                             (current_pos[0], current_pos[1]))

        new_positions = list()
        if self.is_action_allowed(current_pos, action):
            new_pos = current_pos + action
            transition_prob = 1.

            if (action != self.ACTION_IDLE).all():
                residual_action_1, residual_action_2 = self.get_residual_transitions(action)
                residual_pos_1 = current_pos + residual_action_1
                residual_pos_2 = current_pos + residual_action_2

                if self.is_within_bounds(residual_pos_1) and not self.is_wall(residual_pos_1):
                    new_positions.append([residual_pos_1, self.action_stochasticity])
                    transition_prob -= self.action_stochasticity

                if self.is_within_bounds(residual_pos_2) and not self.is_wall(residual_pos_2):
                    new_positions.append([residual_pos_2, self.action_stochasticity])
                    transition_prob -= self.action_stochasticity

            new_positions.append([new_pos, transition_prob])

        return new_positions

    def get_transition_prob(self, current_pos, next_pos, action):
        if not self.is_action_allowed(current_pos, action):
            return 0.

        transition_prob = 1.
        new_pos = current_pos + action
        residual_action_1, residual_action_2 = self.get_residual_transitions(action)
        residual_pos_1 = current_pos + residual_action_1

        if (residual_pos_1 == next_pos).all():
            return self.action_stochasticity

        residual_pos_2 = current_pos + residual_action_2
        if (residual_pos_2 == next_pos).all():
            return self.action_stochasticity

        if (new_pos == next_pos).all():
            if self.is_within_bounds(residual_pos_1) and not self.is_wall(residual_pos_1):
                transition_prob -= self.action_stochasticity

            if self.is_within_bounds(residual_pos_2) and not self.is_wall(residual_pos_2):
                transition_prob -= self.action_stochasticity

            return transition_prob
        else:
            return 0.

    goal_pos     = property(fget=lambda self: np.array([self.__goal_x, self.__goal_y], np.int32))
    start_pos    = property(fget=lambda self: np.array([self.__start_x, self.__start_y], np.int32))
    trap_pos     = property(fget=lambda self: np.array([self.__trap_x, self.__trap_y], np.int32))
    world_width  = property(fget=lambda self: self.map.shape[1])
    world_height = property(fget=lambda self: self.map.shape[0])
    actions      = property(fget=lambda self: [self.ACTION_UP, self.ACTION_RIGHT, self.ACTION_DOWN, self.ACTION_LEFT,
                                               self.ACTION_IDLE])
