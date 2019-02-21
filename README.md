This is the repository for Research and Development Project "Autonomous Exploration for Mobile Robots". The complete system is based on ROS and Hochschule BRS Robocup @Work repository, we implemented some map exploration methods mentioned below:

### Methods of Exploration:
---
- Nearest Frontier [2] [Hector_navigation](https://github.com/tu-darmstadt-rospkg/hector_navigation)
- Nearest Cluster
- Cost-Utility [1]
- Deep Q-Network

For running these algorithms look at launch file  "mas_industrial_robotics/mir_autonomous_exploration/launch/autonomous_exploration_sim.launch" . Also, run *rqt_reconfigure* and go to *hector_exploration_planner* settings for changing exploration behavior. You can change exploration behavior from frontier exploration to Cluster Exploration or to cost-utility exploration.

For deep q-network implementation you need to know anaconda. environment file of anaconda is environment.yml . Also, docker files are available.


### References
[1] Yan, Z., Fabresse, L., Laval, J., & Bouraqadi, N. (2015). Metrics for performance benchmarking of multi-robot exploration. IEEE International Conference on Intelligent Robots and Systems, 2015–Decem, 3407–3414.

[2] Yamauchi, B. (1998). Frontier-based exploration using multiple robots. Proc. of the Second International Conference on Autonomous Agents, 47–53.

### Citation

```
@misc{daiem2018,
  author = {Ali, Daiem Nadir},
  title = {Autonommous Exploration for Mobile Robots},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/daiemna/Autonomous_Exploration}}
}
```
