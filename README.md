This is the repository for Research and Development Project "Autonomous Exploration for Mobile Robots"

### Branch frontier_exploration
## notes about hector exploration :
- start from doExploration()
- Frontier Cell: the cell has to be free but at least three of its neighbors need
  to be unknown.
- assign cost to all the frontiers and all the free cells in explored region (buildexploration_trans_array_)
  - assign cost using breath first sequence in clock wise manner.
  - cost of each frontier is 0 (can be euclidean distance from robot.)
  - cost of each neighbor is cost of the parent cell + distance to nearest obstacle
  - each free cell that is 4-connected to parent cell has less cost than
    each free diagonal cell. reason for this is to avoid much turns during exploration
- to find the trajectory to the frontier start from robots position.
  - select a cell from 8-connected neighbor cells which reduces the cost most.
  - add the cell to trajectory.
  - repeat until you reach a frontier cell.
- to do exploration near previous trajectory.
  - add each pose of trajectory as frontier and the call buildexploration_trans_array_ to assign cost.
  - this will cause cost to increase outwards from trajectory.

## TODO(s):
- robot voilated the obstacle wile path following boundaries change that.
  - possible solution is to check unified range sensor data for proximity and abandon the plan.
  -
