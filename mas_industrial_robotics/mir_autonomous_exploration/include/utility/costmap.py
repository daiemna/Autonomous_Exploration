import random
import rospy
from rospy import Subscriber, loginfo, logdebug
from nav_msgs.msg import OccupancyGrid
from  map_msgs.msg import OccupancyGridUpdate
import numpy as np
import itertools as it


class GlobalCostmap(object):
    """Read global costmap and update info."""

    def costmap_init_cb(self, occupancy_grid):
        self.costmap_data_ = np.asarray(occupancy_grid.data)
        self.map_resolution = occupancy_grid.info.resolution

        # map(self.costmap_data_, lambda x : known_cells = known_cells + 1 if x >= 0)


        loginfo("recived occupancy_grid of size %d min : %d , max : %d, res : %f", len(self.costmap_data_), min(self.costmap_data_),
                max(self.costmap_data_),
                self.map_resolution)

    def costmap_updates_cb(self, updates):
        ind_pairs = it.product(range(updates.x, updates.x + updates.width), range(updates.y, updates.y + updates.height))
        for (x,y),value in zip(ind_pairs, updates.data):
            self.costmap_data_[x * y] = value

        known_cells = 0.0
        for i in self.costmap_data_:
            known_cells += 1 if i > 0 else 0
        self.known_cells = known_cells
        logdebug("know area size %s", known_cells * (self.map_resolution)**2)
    def get_known_map_area(self, in_meters=False):
        if in_meters:
            return self.known_cells * (self.map_resolution)**2
        return self.known_cells

    def __init__(self,topic_name, update_topic_name):
        self.costmap_data_ = None
        self.map_resolution = 0.0
        self.known_cells = 0
        Subscriber(topic_name, OccupancyGrid, self.costmap_init_cb)
        Subscriber(update_topic_name, OccupancyGridUpdate, self.costmap_updates_cb)
