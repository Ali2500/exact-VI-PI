from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from visualizer import Visualizer
from world import World

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

np.set_printoptions(precision=2)

# Two possible implementations for the one-step cost are provided


def one_step_cost_v1(world, current_pos, next_pos):
    # Reaching the goal yields -1 cost, falling into the trap yields a cost of 50, and all other transitions give
    # zero cost.
    if (next_pos == world.goal_pos).all():
        return -1.
    elif (next_pos == world.trap_pos).all():
        return 50.
    else:
        return 0.


def one_step_cost_v2(world, current_pos, next_pos):
    # Reaching the goal yields -1 cost, falling into the trap yields a cost of 50, and all other transitions give
    # a cost of 1.
    if (current_pos == world.goal_pos).all() and (next_pos == world.goal_pos).all():
        return 0.
    elif (next_pos == world.trap_pos).all():
        return 50.
    else:
        return 1.


def main(args):
    # resolve path to world map definition
    if not args.world:
        world_map_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'world_map.txt')
    else:
        world_map_path = args.world

    print("Reading world from %s" % world_map_path)
    if not os.path.exists(world_map_path):
        raise IOError("World map definition not found at its expected path: %s" % world_map_path)

    world = World(world_map_path)
    visualizer = Visualizer(world)

    # Value Iteration
    value_iteration = ValueIteration(world, one_step_cost_v1, discount_factor=args.gamma, eps=10e-10)
    value_iteration.execute()
    optimal_policy = value_iteration.extract_policy()

    fig_vi = plt.figure()
    visualizer.draw(fig_vi, optimal_policy, value_iteration.value_fn, "Value Iteration (gamma = %.2f)" % args.gamma)

    # Policy Iteration
    policy_iteration = PolicyIteration(world, one_step_cost_v1, discount_factor=args.gamma)
    value_fn = policy_iteration.execute()

    fig_pi = plt.figure()
    visualizer.draw(fig_pi, policy_iteration.policy, value_fn, "Policy Iteration (gamma = %.2f)" % args.gamma)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', '-w', required=False)
    parser.add_argument('--gamma', type=float, default=0.9)
    main(parser.parse_args())
