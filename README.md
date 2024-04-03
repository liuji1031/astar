# A* path planning for mobile robot
## Usage
Source ROS
```
source install/setup.bash
```

Build the workspace
```
cd ~\project3_ws
colcon build --packages-select turtlebot3_project3
```

Launch Environment
```
ros2 launch turtlebot3_project3 competition_world.launch.py
```

You can run the [astar_sim.py](/scripts/astar_sim.py) using

```sh
ros2 run turtlebot3_project3 astar_sim.py
```
