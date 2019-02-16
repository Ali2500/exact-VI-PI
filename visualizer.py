from copy import copy

import numpy as np
import matplotlib.pyplot as plt


class Visualizer(object):
    def __init__(self, world):
        self.world = world

    def draw(self, fig, policy, value_fn, title):
        value_fn_flipped = np.flipud(value_fn)
        arrow_dirs = np.flip(policy, axis=0)
        arrow_dirs[:, :, 1] *= -1

        arrow_loc_x = list()
        arrow_loc_y = list()
        arrow_dir_x = list()
        arrow_dir_y = list()

        wall_value = np.min(value_fn_flipped) - 1.

        for y in range(self.world.world_height):
            for x in range(self.world.world_width):
                arrow_loc_x.append(x + 0.5)
                arrow_loc_y.append(y + 0.5)
                arrow_dir_x.append(arrow_dirs[y, x, 0])
                arrow_dir_y.append(arrow_dirs[y, x, 1])

                if self.world.is_wall((x, self.world.world_height - y - 1)):
                    value_fn_flipped[y, x] = wall_value

        ax = fig.add_subplot(111)
        ax.set_title(title)

        color_map = copy(plt.cm.viridis)
        color_map.set_over('k', 1.0)
        color_map.set_under('k', 1.0)
        color_map.set_bad('k', 1.0)
        pcolor = ax.pcolor(value_fn_flipped, vmin=wall_value + 1., vmax=np.max(value_fn_flipped), cmap=color_map,
                           edgecolors='k')

        cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(pcolor, cax=cax, orientation='vertical')

        ax.quiver(arrow_loc_x, arrow_loc_y, arrow_dir_x, arrow_dir_y, angles='xy', scale_units='xy', scale=1.4)
        fig.subplots_adjust(right=0.8)
