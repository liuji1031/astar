#!/usr/bin/env python3

import argparse
import rclpy
import rclpy.logging
from rclpy.node import Node
from geometry_msgs.msg import Twist
from astar_planner.astar_sim import Action, Turtlebot3Waffle, State, Map, VisTree, Astar

class VelocityPublisher(Node):
    def __init__(self, publish_freq, astar:Astar):
        super().__init__('astar_node')
        self.timer_period = 1.0/publish_freq
        self.cmd_vel_pub = self.create_publisher(Twist,'/cmd_vel', 10)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.i = 0
        self.astar = astar
        self.action_list = astar.retrieve_actions()
        self.completed = False
        
    def timer_callback(self):
        if self.completed:
            return
        
        msg = Twist()

        if self.i >= len(self.action_list):
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            self.completed=True
        else:
            a:Action = self.action_list[self.i]
            msg.linear.x = a.lin_vel/1000
            msg.angular.z = a.ang_vel
            self.get_logger().info(f"lin vel: {msg.linear.x:.2f}, ang vel: {msg.angular.z:.2f}")
            self.cmd_vel_pub.publish(msg)
            self.astar.visualize_path(ind=self.i)
            self.i+=1

def main(args=None):
    
    dt = 5.0
    cogw=1.00
    State.xy_res = Turtlebot3Waffle.robot_radius/10.0

    # create map object
    custom_map = Map(inflate_radius=1.5*Turtlebot3Waffle.robot_radius,
                     width=6000,height=2000)

    # define the corners of all the convex obstacles
    obs_corners = []
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(1500,2000),
                                            w=250,h=1000))
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(2500,1000),
                                            w=250,h=1000))
    obs_corners.append(custom_map.get_corners_circ(
                                            center=(4200,1200),
                                            circle_radius=600,n=30))
    
    # add all obstacles to map
    for c in obs_corners:
        custom_map.add_obstacle(corners_tuple=c)

    # get the inflated obstacle corners
    corners = custom_map.get_obstacle_corners_array()

    # # ask user for init and goal position
    # init_coord,init_ori = ask_for_coord(custom_map, mode="initial")
    # goal_coord,goal_ori = ask_for_coord(custom_map, mode="goal")

    init_coord = (500,1000)
    init_ori = 0

    goal_coord = (5750,1000)

    vt = VisTree(corners=corners,goal_coord=goal_coord,
             boundary=custom_map.obstacle_boundary_inflate,
             inflate_coef=cogw)
    
    # create Astar solver
    a = Astar(init_coord=init_coord,
              init_ori=init_ori,
              goal_coord=goal_coord,
              rpms=[10,20],
              wheel_radius=Turtlebot3Waffle.wheel_radius,
              wheel_distance=Turtlebot3Waffle.wheel_distance,
              map=custom_map,
              vis_tree=vt,
              savevid=False,
              vid_res=300,
              dt=dt,
              )

    # run the algorithm
    a.run()
    
    rclpy.init(args=None)
    node = VelocityPublisher(publish_freq=1.0/dt, astar=a)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()