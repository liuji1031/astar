#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
from pynput import keyboard
import argparse
import numpy as np
import heapq
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

DEBUG=False

radius_wheel_tb = 33 # diameter 66 mm
D_tb = 287 # distance between two wheels

radius_wheel_sim = 5
D_sim = 10

class Action:
    
    def __init__(self, rpm_left, rpm_right,
                 dt=0.1,
                 in_gazebo=True,
                 ) -> None:
        if in_gazebo:
            # use actual turtlebot dimension
            radius_wheel = radius_wheel_tb
            D = D_tb
        else:
            radius_wheel = radius_wheel_sim
            D = D_sim

        self.rpm_left = rpm_left
        self.rpm_right = rpm_right
        self.dt = dt

        self.w_l = 2*np.pi/60.0*rpm_left
        self.w_r = 2*np.pi/60.0*rpm_right # angular vel of left and
                                                       # right wheel
        self.v_l = self.w_l*radius_wheel
        self.v_r = self.w_r*radius_wheel
        vel_sum = self.v_l+self.v_r
        vel_diff = self.v_l-self.v_r
        if vel_diff != 0:
            self.ang_vel = -2*vel_diff/D
            self.dyaw = self.ang_vel*self.dt
            self.turn_radius = D/2*vel_sum/np.abs(vel_diff)
            self.cost = self.turn_radius*np.abs(self.dyaw)
        else:
            self.ang_vel = 0.
            self.dyaw = 0.
            self.turn_radius = None
            self.cost = self.v_l*self.dt

        # generate array of points for plotting
        if vel_diff == 0:
            L = self.cost
            xs = np.linspace(0,L,2,endpoint=True)[:,np.newaxis]
            ys = np.zeros_like(xs)
            self.traj = np.hstack((xs,ys))
            self.cx=None
            self.cy=None
        else:
            if self.v_l > self.v_r: # turn right
                phi0 = np.pi/2
                cx=0.0
                cy=-1*self.turn_radius
            else:
                phi0 = -np.pi/2
                cx=0.0
                cy=self.turn_radius

            phi1 = phi0+self.dyaw
            ts = np.linspace(phi0,phi1,10)
            xs = cx+self.turn_radius*np.cos(ts)
            ys = cy+self.turn_radius*np.sin(ts)

            self.traj = np.hstack((xs[:,np.newaxis],ys[:,np.newaxis]))
            self.cx=cx
            self.cy=cy
        
        self.plt_pts = np.hstack((self.traj,
                                  np.ones(self.traj.shape[0])[:,np.newaxis]))
        
    def apply(self, init_pose):
        """apply the action to get final pose

        Args:
            init_pose (_type_): tuple of x, y, yaw of the robot
        """
        
        if self.v_l != self.v_r: # traveling on a curve
            new_pose, center_rot = self.curve_motion(init_pose)
        else:
            new_pose = self.straight_motion(init_pose)
            center_rot = None
        
        return new_pose, center_rot, self.cost
    
    def curve_motion(self, init_pose):
        """apply curved motion to the init pose

        Args:
            init_pose (_type_): _description_
        """
        x0,y0,yaw0 = init_pose

        cyaw0 = np.cos(yaw0)
        syaw0 = np.sin(yaw0)

        x1b,y1b = self.traj[-1,:]

        x1 = cyaw0*x1b-syaw0*y1b+x0
        y1 = syaw0*x1b+cyaw0*y1b+y0
        yaw1 = wrap(yaw0+self.dyaw)

        cx = cyaw0*self.cx-syaw0*self.cy+x0
        cy = syaw0*self.cx+cyaw0*self.cy+y0

        return (x1,y1,yaw1),(cx,cy)
    
    def straight_motion(self, init_pose):
        """apply straight motion

        Args:
            init_pose (_type_): _description_
        """
        x0,y0,yaw0 = init_pose

        x1 = x0+self.cost*np.cos(yaw0)
        y1 = y0+self.cost*np.sin(yaw0)
        yaw1 = wrap(yaw0)

        return (x1,y1,yaw1)
    
    def get_plot_pts(self, init_pose):
        x0,y0,yaw0 = init_pose

        cyaw0 = np.cos(yaw0)
        syaw0 = np.sin(yaw0)

        T = np.array([[cyaw0, -syaw0, x0],
                      [syaw0,  cyaw0, y0],
                      [   0.,     0., 1.]])
        
        plt_pts = T.dot(self.plt_pts.T)[:2,:]
        return np.transpose(plt_pts)

def wrap(rad):
    """wrap a angle between 0 to 2pi

    Args:
        rad (_type_): _description_
    """
    twopi = 2*np.pi
    while rad >= twopi:
        rad-=twopi
    
    while rad<0:
        rad+=twopi
    return rad

def collision_check_curve(pose,
                          action:Action,
                          center_rot,
                          boundaries,
                          debug=DEBUG):
    """this function check if the curved path the robot is on will collide 
    with obstacles

    Args:
        start_xy (_type_): the starting position of the robot, tuple of (x,y)
        rpm (_type_): tuple of left and right wheel rpm
        boundaries (_type_): a list of list, containing boundary normal and 
        projection value defining the boundary
    """
    x,y,yaw = pose
    if debug:
        print("+++++ Current Pose +++++ ",x,y,yaw)
    
    cx,cy = center_rot
    min_val = 0.0
    max_val = np.abs(action.dyaw)
    if debug:
        print("Initial")
        print((min_val,max_val))

    if action.v_r>action.v_l: # turn towards left
        phi0 = yaw-np.pi/2
    else: # turn towards right
        phi0 = yaw+np.pi/2
    # compute center of rotation
    
    if action.v_l>action.v_r:
        phi0-=max_val
    cp = np.cos(phi0)
    sp = np.sin(phi0)
    # print(cx,cy,phi0/np.pi*180)

    for ib,b in enumerate(boundaries):
        if debug:
            print(f"=== boundary {ib+1}")
        b:dict
        if b["type"]=="polygon": # obstacle defined by a polygon
            normal, proj = b["normal"], b["proj"]
            nx = normal[0]
            ny = normal[1]
            # convert to frame defined by phi0, cx, cy
            proj_ = proj-nx*cx-ny*cy
            
            # print("theta hat: ",theta_hat/np.pi*180)
            rho = proj_/action.turn_radius
            
            if rho>=1:
                if debug:
                    print("condition 1, full range")
                continue
            elif rho>=-1 and rho<1:
                nx_ = cp*nx+sp*ny
                ny_ = -sp*nx+cp*ny
                theta_hat = wrap(np.arctan2(ny_,nx_))
                cos_inv_rho = np.arccos(rho)
                theta1 = cos_inv_rho+theta_hat
                theta2 = 2*np.pi-cos_inv_rho+theta_hat
                if debug:
                    print("condition 2",rho,theta1,theta2,cos_inv_rho,theta_hat)
                if theta1>2*np.pi or theta2>2*np.pi:
                    theta1-=2*np.pi
                    theta2-=2*np.pi
                    if debug:
                        print("\tcorrection",theta1,theta2)
                min_val = max(min_val,theta1)
                max_val = min(max_val,theta2)
            elif rho<-1:
                if debug:
                    print("condition 3",rho)
                return False
            if debug:
                print((min_val, max_val))
        elif b["type"]=="circle": # circular obstacle
            # cx_obs, cy_obs = b["center"]
            # r_obs = b["radius"]

            # dist = np.linalg.norm((cx-cx_obs,cy-cy_obs),ord=2)
            # return dist<=(r_obs+)
            pass
            
    # if non empty intersection, the curve intersects with the obstacle
    return min_val<=max_val

def collison_check_line_seg(x_start,y_start,
                    x_end,y_end,
                    boundary,
                    exclude_boundary=False):
    """check if the line segment defined by (x_start,y_start),(x_end, y_end) is 
    within the obstacle defined by the corners

    Args:
        x (_type_): _description_
        y (_type_): _description_
        corners (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = len(boundary)
    intersection = []
    for i in range(n):
        # get normal direction
        normal,p = boundary[i]["normal"],boundary[i]["proj"]

        # find the meshgrid points whose projections are <= p
        p1 = np.inner((x_start,y_start), normal)
        p2 = np.inner((x_end,y_end), normal)

        # now see if (1-rho)*p1+rho*p2 for rho>=0 and rho<=1 has any range of
        # rho that makes the value < p+inflate_radius
        if not exclude_boundary:
            if p1 > p and p2 > p:
                return False
            elif p1 <= p and p2 <= p:
                intersection.append([0.0,1.0]) # full range
            elif p1 > p and p2 <= p:
                intersection.append([(p1-p)/(p1-p2), 1.0])
            elif p1 < p and p2 >= p:
                intersection.append([0.0, (p1-p)/(p1-p2)])
        else:
            if p1 >= p-1e-8 and p2 >= p-1e-8:
                return False
            elif p1 < p and p2 < p:
                intersection.append([0.0,1.0]) # full range
            elif p1 > p and p2 < p:
                intersection.append([(p1-p)/(p1-p2), 1.0])
            elif p1 < p and p2 > p:
                intersection.append([0.0, (p1-p)/(p1-p2)])

    # compute the final intersection
    min_val = 0.0
    max_val = 1.0
    for i in intersection:
        min_val = max(min_val, i[0])
        max_val = min(max_val, i[1])

    # if the final intersection is non empty, it means some of the line segment
    # is within the obstacle
    if not exclude_boundary:
        return min_val<=max_val
    else:
        # okay if min_val takes 0 and max_val takes 1
        return (min_val<max_val-1e-8)

def collison_check_point(coord,
                    boundary):
    """check if the line segment defined by (x_start,y_start),(x_end, y_end) is 
    within the obstacle defined by the corners

    Args:
        x (_type_): _description_
        y (_type_): _description_
        corners (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = len(boundary)
    out=None
    for i in range(n):
        # get normal direction
        normal,p = boundary[i]

        # find the meshgrid points whose projections are <= p
        p_ = np.inner(coord, normal.reshape((1,2)))
        if out is None:
            out = (p_<=p)
        else:
            out=np.logical_and(out, p_<=p)
    return out

class Map:
    """class representing the map
    """
    def __init__(self,
                 width=6000,#1200,
                 height=2000,#500,
                 inflate_radius=100,#5
                 res=0.5,
                 ):
        """create a map object to represent the discretized map

        Args:
            width (int, optional): the width of the map. Defaults to 1200.
            height (int, optional): the height of the map. Defaults to 500.
            inflate_radius (int, optional): the radius of the robot for inflat-
            ing the obstacles. Defaults to 5.
        """
        self.width = width
        self.height = height
        self.res=res
        self.map = np.zeros((height, width),dtype=np.int8) # 0: obstacle free
                                                           # 1: obstacle
        self.map_inflate = np.zeros_like(self.map)
        self.inflate_radius = inflate_radius
        self.obstacle_corners = []
        self.obstacle_corners_inflate = []
        self.obstacle_boundary = []
        self.obstacle_boundary_inflate = []
        self.obstacle_map = None

    def add_obstacle(self, corners_tuple):
        """add obstacle defined by the corner points. the corners should define
        a convex region, not non-convex ones. for non-convex obstacles, need to 
        define it in terms of the union of the convex parts. 

        Args:
            corners (_type_): the corners of the obstacles, defined in the
            clockwise direction. each row represents the (x,y) coordinate of a 
            corner

        Returns:
            _type_: _description_
        """
        corners = corners_tuple[0]
        corners_inflate = corners_tuple[1]
        obs_map = np.zeros((self.height, self.width),dtype=np.int8)
        obs_map_inflate = np.zeros_like(obs_map)

        # first get a meshgrid of map coordinates
        x, y = np.meshgrid(np.arange(0,self.width), np.arange(0,self.height))
        xy_all = np.hstack((x.flatten()[:,np.newaxis],
                            y.flatten()[:,np.newaxis]))

        if corners.shape[1] != 2:
            corners = corners.reshape((-1,2)) # make sure it's a 2D array

        # add to the list of obstacle corners
        self.obstacle_corners.append(corners)
        self.obstacle_corners_inflate.append(corners_inflate)

        n = corners.shape[0]
        boundary = []
        boundary_inflate = []
        for i in range(corners.shape[0]):
            j = int((i+1)%n) # the adjacent corner index in clockwise direction

            # get x, y
            x1,y1 = corners[i,:]
            x2,y2 = corners[j,:]

            # get normal direction
            normal_dir = np.arctan2(y2-y1, x2-x1) + np.pi/2
            normal = np.array([np.cos(normal_dir),np.sin(normal_dir)])

            # compute the projection of one of the corner point
            p = np.inner((x1,y1),normal)

            # find the meshgrid points whose projections are <= p
            proj_all = np.inner(xy_all, normal).reshape((self.height,
                                                         self.width))

            obs_map += np.where(proj_all<=p,1,0)
            obs_map_inflate += np.where(proj_all<=p+self.inflate_radius,1,0)

            # record the boundary and projection value
            boundary.append(dict(type="polygon",normal=normal,proj=p))
            boundary_inflate.append(dict(type="polygon",
                                         normal=normal,
                                         proj=p+self.inflate_radius))
        
        self.obstacle_boundary.append(boundary)
        self.obstacle_boundary_inflate.append(boundary_inflate)
        
        # find points that meet all half plane conditions
        obs_map = np.where(obs_map==n,1,0)
        obs_map_inflate = np.where(obs_map_inflate==n,1,0)

        # add to the existing map
        self.map = np.where(obs_map==1,obs_map,self.map)
        self.map_inflate = np.where(obs_map_inflate==1,
                                    obs_map_inflate,self.map_inflate)

    def compute_obstacle_map(self):
        """compute the obstacle map at the resolution defined
        """

        nh = int(self.height/self.res+1)
        nw=int(self.width/self.res+1)
        xs,ys = np.meshgrid(
                            np.linspace(0,self.width,nw),
                            np.linspace(0,self.height,nh)
                            )

        self.obstacle_map = np.zeros((nh,nw),dtype=bool)
        p = multiprocessing.Pool()

        xy_array = np.hstack((xs.reshape((-1,1)),ys.reshape((-1,1))))
        out = None
        for b in self.obstacle_boundary_inflate:
            tmp = collison_check_point(xy_array, b)
            if out is None:
                out = tmp
            else:
                out = np.logical_or(out, tmp)
        self.obstacle_map = np.reshape(out, (nh,nw))

    def plot(self,show=True):
        """show the map

        Args:
            show (bool, optional): _description_. Defaults to True.
        """
        ax = plt.gca()
        for c in self.obstacle_corners_inflate:
            ax.add_patch(plt.Polygon(c,facecolor=(0.8,0.8,0.8)))
        ax.set_aspect("equal",adjustable="box")
        for c in self.obstacle_corners:
            ax.add_patch(plt.Polygon(c,facecolor=(0.4,0.4,0.4)))
        
        plt.xlim([0,self.width])
        plt.ylim([0,self.height])
        if show:
            plt.show()

    def in_range(self, x, y):
        """return true if (x, y) within the range of the map

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        if x>=0 and x<=self.width and y>=0 and y<=self.height:
            return True
        else:
            return False

    def check_obstacle_line_seg(self,
                                x_start,
                                y_start,
                                x_end,
                                y_end,
                                pool=None):
        """go through all obstacles to check if x,y is within any one of them

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        if pool is None:
            for boundary in self.obstacle_boundary_inflate:
                # returns true if within any obstacle
                if collison_check_line_seg(x_start,
                                   y_start,
                                   x_end,
                                   y_end,
                                   boundary):
                    return True
            return False
        else:
            # use parallel pool to speed things up
            nobs = len(self.obstacle_corners)
            inputs = zip((x_start,)*nobs,
                         (y_start,)*nobs,
                         (x_end,)*nobs,
                         (y_end,)*nobs,
                         self.obstacle_boundary_inflate)
            outputs = pool.starmap(collison_check_line_seg,inputs)

            for out in outputs:
                if out:
                    return True
            return False
    
    def check_obstacle_curve(self,
                            curr_pose,
                            action:Action,
                            center_rot,
                            pool=None):
        """go through all obstacles to check if x,y is within any one of them

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        if pool is None:
            for boundary in self.obstacle_boundary_inflate:
                # returns true if within any obstacle
                if collision_check_curve(pose=curr_pose,
                                         action=action,
                                         center_rot=center_rot,
                                         boundaries=boundary):
                    return True
            return False
        else:
            # use parallel pool to speed things up
            nobs = len(self.obstacle_corners)
            inputs = zip((curr_pose,)*nobs,
                         (action,)*nobs,
                         (center_rot,)*nobs,
                         self.obstacle_boundary_inflate)
            outputs = pool.starmap(collision_check_curve,inputs)

            for out in outputs:
                if out:
                    return True
            return False
        
    def get_obstacle_corners_array(self,
                                   omit,
                                   correction):
        """returns an numpy array of all obstacle corners

        Returns:
            _type_: _description_
        """
        out = []
        for i, corners in enumerate(self.obstacle_corners_inflate):
            corners:np.ndarray
            for j in range(corners.shape[0]):
                skip=False
                for o in omit:
                    if (i,j)==o:
                        skip=True
                        break
                if not skip:
                    if (i,j) in correction:
                        out.append(corners[j,:]+np.array(correction[(i,j)]))
                    else:
                        out.append(corners[j,:])
        return np.array(out)
    
    def get_corners_hex(self,center, radius):
        """get the hexagon corner points

        Args:
            center (_type_): _description_
            radius (_type_): _description_

        Returns:
            _type_: _description_
        """
        theta = np.pi/2 + np.linspace(0., -2*np.pi, 50, endpoint=False)
        radius_inflate = radius + self.inflate_radius/np.sqrt(3)*2

        corners = np.hstack([
                    (center[0]+radius*np.cos(theta))[:,np.newaxis],#
                    (center[1]+radius*np.sin(theta))[:,np.newaxis],#
                    ])
        corners_inflate = np.hstack([
                    (center[0]+radius_inflate*np.cos(theta))[:,np.newaxis],#
                    (center[1]+radius_inflate*np.sin(theta))[:,np.newaxis],#
                    ])
        return (corners, corners_inflate)
    
    def get_corners_rect(self,
                         upper_left,
                         w,
                         h):
        """return the 4 corners of a rectangle in clockwise order

        Args:
            upper_left (_type_): _description_
            w (_type_): _description_
            h (_type_): _description_

        Returns:
            _type_: _description_
        """
        corners = np.array([[0,0],
                            [w,0],
                            [w,-h],
                            [0,-h]
                           ])
        corners += np.array([upper_left[0],upper_left[1]])[np.newaxis,:]

        r = self.inflate_radius
        corners_inflate = corners+np.array([[-r,+r],
                                            [+r,+r],
                                            [+r,-r],
                                            [-r,-r]
                                        ])
        for i in range(4):
            corners_inflate[i,0] = max(0,corners_inflate[i,0])
            corners_inflate[i,0] = min(self.width,corners_inflate[i,0])
            corners_inflate[i,1] = max(0,corners_inflate[i,1])
            corners_inflate[i,1] = min(self.height,corners_inflate[i,1])

        return (corners, corners_inflate)

def round_to_precision(data,
                       precision=0.5):
    """round the coordinate according to the requirement, e.g., 0.5

    Args:
        data (_type_): _description_
        precision (float, optional): _description_. Defaults to 0.5.
    """
    if isinstance(data, tuple):
        x_ = np.round(data[0]/precision)*precision
        y_ = np.round(data[1]/precision)*precision
        return (x_,y_)
    else:
        return np.round(data/precision)*precision

class State:
    """create a custom class to represent each map coordinate.
    attribute cost_to_come is used as the value for heap actions.
    for this purpose, the <, > and = operations are overridden

    """
    xy_res = 2.5
    rad_res = 30.0/180.0*np.pi
    def __init__(self,
                 coord,
                 orientation,
                 cost_to_come,
                 cost_to_go=None,
                 parent=None,
                 parent_action=None,
                 vt_node=None) -> None:

        self.coord = coord
        self.orientation = wrap(orientation)
        self.cost_to_come = cost_to_come
        self.cost_to_go = cost_to_go
        self.estimated_cost = self.cost_to_come+self.cost_to_go
        self.parent = parent
        self.parent_action=parent_action # the action taken by the parent to
                                         # arrive at the child node
        self.vt_node = vt_node

    def __lt__(self, other):
        return self.estimated_cost < other.estimated_cost
    
    def __gt__(self, other):
        return self.estimated_cost > other.estimated_cost
    
    def __eq__(self, other):
        return self.estimated_cost == other.estimated_cost
    
    def set_parent(self, parent):
        self.parent = parent

    def same_state_as(self, other, ignore_ori=False):
        dx = self.x-other.x
        dy = self.y-other.y
        rad1 = self.orientation/180*np.pi
        v1 = np.array([np.cos(rad1),np.sin(rad1)])

        rad2 = other.orientation/180*np.pi
        v2 = np.array([np.cos(rad2),np.sin(rad2)])

        proj = np.inner(v1,v2).item()

        if not ignore_ori:
            return (np.linalg.norm((dx,dy))<1e-3) & (proj>0.95)
        else:
            return np.linalg.norm((dx,dy))<1e-3
    
    def update(self, cost_to_come, parent, parent_action):
        self.cost_to_come = cost_to_come
        self.estimated_cost = self.cost_to_come+self.cost_to_go
        self.parent = parent
        self.parent_action = parent_action
    
    @property
    def x(self):
        return self.coord[0]
    
    @property
    def y(self):
        return self.coord[1]
    
    @property
    def index(self):
        ideg = int(self.orientation/State.rad_res)
        iw = int(self.x/State.xy_res)
        ih = int(self.y/State.xy_res)
        return ideg, ih, iw
    
def cost_to_go_l2(state1, state2):
    """calculate the optimistic cost to go estimate

    Args:
        state1 (_type_): _description_
        state2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(state1,tuple):
        x1,y1 = state1
    elif isinstance(state1, State):
        x1 = state1.x
        y1 = state1.y

    if isinstance(state2,tuple):
        x2,y2 = state2
    elif isinstance(state2, State):
        x2 = state2.x
        y2 = state2.y
    return np.linalg.norm([x1-x2, y1-y2],ord=2)

def motion_model_proj3(curr_coord,
                       curr_ori,
                       L,
                       dtheta=30,
                       deg_coef=0.0):
    """returns the action set defined in proj3

    Returns:
        _type_: _description_
    """
    x0,y0 = curr_coord
    next_state = []
    action_cost = [L]*5

    deg = np.arange(-2,3,1)*dtheta
    new_ori = (curr_ori+deg)%360
    rad = new_ori/180*np.pi
    x1 = round_to_precision(x0+L*np.cos(rad))
    y1 = round_to_precision(y0+L*np.sin(rad))

    next_state = np.hstack((x1[:,np.newaxis],
                            y1[:,np.newaxis],
                            new_ori[:,np.newaxis]))

    return next_state, action_cost

class VisTreeNode:
    """node of the visibility tree

    Returns:
        _type_: _description_
    """
    def __init__(self,
                 coord,
                 dist_to_goal,
                 parent=None) -> None:
        self.coord = coord
        self.dist_to_goal = dist_to_goal
        self.parent = parent
        self.children = []

    def __lt__(self, other):
        return self.dist_to_goal < other.dist_to_goal
    
    def __gt__(self, other):
        return self.dist_to_goal > other.dist_to_goal
    
    def __eq__(self, other):
        return self.dist_to_goal == other.dist_to_goal
    
class VisTree:
    def __init__(self,
                 corners:np.ndarray,
                 goal_coord,
                 boundary,
                 map_w=1200,
                 map_h=500,
                 inflate_coef=1.0) -> None:
        """build the visibility tree from the corner points

        Args:
            corners (_type_): _description_
            goal_coord (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.boundary = boundary
        self.rho = inflate_coef # scale the cog to prefer nodes close to goal
        # first create the root node
        root = VisTreeNode(coord=goal_coord,
                           dist_to_goal=0)
        
        q = [root]
        heapq.heapify(q)
        n = corners.shape[0]

        closed = dict()
        open_nodes = dict()
        while len(q)>0:
            t:VisTreeNode = heapq.heappop(q)
            
            closed[t.coord] = t

            # go through remaining corners to see which one the node t can see
            # directly
            
            for i in range(n):
                c = corners[i,:]
                in_obstacle=False

                # skip if in closed list
                if (c[0],c[1]) in closed:
                    continue
                
                # make sure not on the edge of the map
                if c[0]==0.0 and t.coord[0]==0.0:
                    continue

                if c[0]==map_w and t.coord[0]==map_w:
                    continue

                if c[1]==0.0 and t.coord[1]==0.0:
                    continue

                if c[1]==map_h and t.coord[1]==map_h:
                    continue

                # check obstacles
                for ib,b in enumerate(boundary):
                    out = collison_check_line_seg(x_start=t.coord[0],
                                                  y_start=t.coord[1],
                                                x_end=c[0],y_end=c[1],
                                                boundary=b,
                                                exclude_boundary=True)
                    if out:
                        in_obstacle=True
                        break
                
                if in_obstacle:
                    continue
                
                dist = np.linalg.norm((t.coord[0]-c[0],t.coord[1]-c[1]))
                if (c[0],c[1]) not in open_nodes:
                    # initialize
                    node = VisTreeNode(coord=(c[0],c[1]),
                                       dist_to_goal=t.dist_to_goal+dist,
                                       parent=t)
                    
                    # add to open list
                    open_nodes[(c[0],c[1])] = node

                    # push to heap
                    heapq.heappush(q, node)
                else:
                    node:VisTreeNode = open_nodes[(c[0],c[1])]

                    if node.dist_to_goal > t.dist_to_goal+dist:
                        node.dist_to_goal = t.dist_to_goal+dist
                        node.parent = t
                        heapq.heapify(q)

        # build the tree
        for v in closed.values():
            v:VisTreeNode
            p = v.parent
            if not p:
                continue
            p.children.append(v)
            
        self.root = root

        # store coord and cost to goal
        self.coord_array = []
        self.dist_to_goal_array = []

        q = [self.root]
        while len(q)>0:
            t:VisTreeNode = q.pop(0)
            self.coord_array.append(t.coord)
            self.dist_to_goal_array.append(t.dist_to_goal)
            for c in t.children:
                q.append(c)
            
        self.coord_array = np.array(self.coord_array)[np.newaxis,:,:]
        self.dist_to_goal_array = np.array(self.dist_to_goal_array)

    def compute_cost_to_go_from_root(self,
                           x_start,y_start,
                           pool):
        """_summary_

        Returns:
            _type_: _description_
        """

        q = [self.root]
        heapq.heapify(q)
        dist=0
        vt_node=None
        while len(q)>0:
            t:VisTreeNode = heapq.heappop(q)
            x_end,y_end = t.coord

            # traverse starting from the root
            nobs = len(self.boundary)
            inputs = zip((x_start,)*nobs,
                        (y_start,)*nobs,
                        (x_end,)*nobs,
                        (y_end,)*nobs,
                        self.boundary,
                        (True,)*nobs)
            outputs = pool.starmap(collison_check_line_seg,inputs)
            in_obs=False
            for out in outputs:
                in_obs = in_obs|out
            
            if not in_obs: # not cross any obstacle
                dist = np.linalg.norm((x_start-t.coord[0],y_start-t.coord[1]))
                dist += t.dist_to_goal
                vt_node = t
                break
            else:
                for c in t.children:
                    heapq.heappush(q,c)
        return dist*self.rho,vt_node
    
    def compute_cost_to_go_from_current(self,
                                        curr_node:VisTreeNode,
                                        x_start,
                                        y_start,
                                        pool):
        """_summary_

        Returns:
            _type_: _description_
        """

        # check if we can see parent
        c=curr_node
        t:VisTreeNode = c.parent
        
        dist=0
        while t is not None:
            t:VisTreeNode
            x_end,y_end = t.coord

            # traverse starting from the root
            nobs = len(self.boundary)
            inputs = zip((x_start,)*nobs,
                        (y_start,)*nobs,
                        (x_end,)*nobs,
                        (y_end,)*nobs,
                        self.boundary,
                        (True,)*nobs)
            outputs = pool.starmap(collison_check_line_seg,inputs)
            in_obs=False
            for out in outputs:
                if out:
                    in_obs=True
                    break
            
            if not in_obs: # not cross any obstacle              
                # go back one level
                c = t
                t = t.parent
            else:
                # cannot see parent, return dist to current
                break

        dist = np.linalg.norm((x_start-c.coord[0],y_start-c.coord[1]))
        dist += c.dist_to_goal
        vt_node = c

        return dist*self.rho,vt_node

class Astar:
    # implement the Astar search algorithm

    def __init__(self,
                 init_coord,
                 init_ori,
                 goal_coord,
                 map : Map,
                 vis_tree:VisTree,
                 rpms=[0.25,0.5],
                 savevid=False,
                 vid_res=72,
                 goal_ori=0,
                 dt=0.1,
                 gazebo=False):
        # use multi processing to check for obstacles
        self.check_pool = multiprocessing.Pool()

        init_dist, init_node = vis_tree.compute_cost_to_go_from_root(
                                                    init_coord[0],
                                                    init_coord[1],
                                                    pool=self.check_pool)
        self.init_coord = State(init_coord,
                                init_ori,
                                cost_to_come=0.0,
                                cost_to_go=init_dist,
                                vt_node=init_node)
        
        self.goal_coord = State(goal_coord,
                                goal_ori,
                                cost_to_come=np.inf,
                                cost_to_go=0.0)
        self.map = map
        self.vis_tree:VisTree = vis_tree
        self.savevid = savevid

        self.open_list = [self.init_coord]
        heapq.heapify(self.open_list)
        # use a dictionary to track which coordinate has been added to the open
        # list
        ndeg = int(2*np.pi/State.rad_res)
        nw = int(map.width/State.xy_res)+1
        nh = int(map.height/State.xy_res)+1
        self.open_list_added = [ [[None]*nw for j in range(nh)] \
                                  for k in range(ndeg)
                               ]

        # use a dictionary to store the visited map coordinates;
        # None means not visited. otherwise, stores the actual State obj
        self.closed_list = [ [[None]*nw for j in range(nh)] \
                                  for k in range(ndeg)
                            ]

        self.goal_reached = False
        self.path_to_goal = None

        # create the list of actions
        self.actions = []
        rpms = [0.0,*rpms]
        for rpm_l in rpms:
            for rpm_r in rpms:
                if rpm_l > 0  or rpm_r >0:
                    self.actions.append(Action(rpm_left=rpm_l,
                                            rpm_right=rpm_r,
                                            dt=dt,
                                            in_gazebo=gazebo)
                                        )
        assert(len(self.actions)==8)
        
        # create the handles for the plots
        self.fig = plt.figure(figsize=(12,6))
        self.ax = self.fig.add_subplot()
        self.ax.invert_yaxis()
        # show the map
        self.map.plot(show=False)

        # plot goal location
        self.ax.plot(self.goal_coord.x, self.goal_coord.y, marker="*",ms=10)
        # plot robot location
        self.robot_plot = self.ax.plot(self.init_coord.x, self.init_coord.y,
                                       marker="o",ms=5,c="r")[0]

        # handle for plotting explored states
        self.closed_plot_data_x = None
        self.closed_plot_data_y = None
        self.closed_plot = None
        self.fig.show()

        # create movie writer
        if self.savevid:
            self.writer = FFMpegWriter(fps=15, metadata=dict(title='Astar',
                                                        artist='Matplotlib',
                                                        comment='Path search'))
            self.writer.setup(self.fig, outfile="./animation.mp4",dpi=vid_res)

    @property
    def map_plot_data(self):
        return 3*(self.map.map+self.map.map_inflate)

    def add_to_closed(self, c : State):
        """add the popped coordinate to the closed list

        Args:
            c (State): _description_
        """
        ideg, ih, iw = c.index
        ideg = ideg%len(self.closed_list)
        self.closed_list[ideg][ih][iw] = c

    def at_goal(self, c : State):
        """return true if c is at goal coordinate

        Args:
            c (State): _description_
        """
        dx = c.x-self.goal_coord.x
        dy = c.y-self.goal_coord.y

        return np.linalg.norm((dx,dy))<0.5*self.map.inflate_radius

    def initiate_coord(self,
                       coord,
                       ori,
                       parent : State,
                       parent_action : Action,
                       edge_cost):
        """initiate new coordinate to be added to the open list

        Args:
            coord (_type_): _description_
            parent (_type_): _description_
        """
        # create new State obj
        
        cog,vt_node = self.vis_tree.compute_cost_to_go_from_current(
                                               curr_node=parent.vt_node,
                                               x_start=coord[0],
                                               y_start=coord[1],
                                               pool=self.check_pool)
        
        new_c = State(coord=coord,
                    orientation=ori,
                    cost_to_come=parent.cost_to_come+edge_cost,
                    cost_to_go=cog,
                    parent=parent,
                    parent_action=parent_action,
                    vt_node=vt_node)
        
        # push to open list heaqp
        heapq.heappush(self.open_list, new_c)
        
        # mark as added
        ideg, ih, iw = new_c.index
        ideg = ideg%len(self.closed_list)
        self.open_list_added[ideg][ih][iw] = new_c

    def print_open_len(self):
        print("current open list length: ", len(self.open_list))

    def update_coord(self, c : State, new_cost_to_come, parent):
        """update the coordinate with new cost to come and new parent

        Args:
            c :  the state to be updated
            new_cost_to_come (_type_): _description_
            parent (_type_): _description_
        """
        ideg, ih, iw = c.index
        ideg = ideg%len(self.closed_list)
        self.open_list_added[ideg][ih][iw].update(new_cost_to_come,parent)
    
    def on_obstacle(self, x, y):
        """check if coord (x,y) is on the obstacle
        return true if there is obstacle

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        return self.map[y,x]>0
    
    def add_to_closed_plot(self, state : State):
        """plot the state's coord and orientation, along with the search
        directions

        Args:
            state (_type_): _description_
        """
        plt.sca(self.ax)

        # plot the path from the parent to the current node
        p:State = state.parent
        if not p:
            return
        a:Action = state.parent_action

        # n by 2 array
        plt_pts = a.get_plot_pts(init_pose=(p.x,p.y,p.orientation))

        xarr = np.concatenate([plt_pts[:,0],np.array([None])])
        yarr = np.concatenate([plt_pts[:,1],np.array([None])])

        if self.closed_plot_data_x is None:
            self.closed_plot_data_x = xarr
            self.closed_plot_data_y = yarr
        else:
            self.closed_plot_data_x = np.concatenate(
                (self.closed_plot_data_x, xarr))
            self.closed_plot_data_y = np.concatenate(
                (self.closed_plot_data_y, yarr))

    def visualize_search(self):
        """visualize the search process
        """
        if self.closed_plot_data_x is None:
            return
        if not self.closed_plot:
            self.closed_plot = plt.plot(self.closed_plot_data_x,
                                        self.closed_plot_data_y,
                                        color=(3/255, 198/255, 252/255),
                                        linewidth=0.5)[0]
        else:
            # update only
            self.closed_plot.set_data(self.closed_plot_data_x,
                                      self.closed_plot_data_y)
        # plt.xlim([50,500])
        # plt.ylim([50,90])
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        if self.savevid:
            self.writer.grab_frame()

        # plt.pause(0.2)
        # plt.pause(0.0001)

    def visualize_path(self):
        """visualize the result of backtrack
        """
        nodes = self.path_to_goal
        scatter_pts_x = []
        scatter_pts_y = []
        for node in nodes:
            x,y = node[0]
            ori = node[1]
            a:Action = node[2]
            if a:
                plt_pts = a.get_plot_pts(init_pose=(x,y,ori))
                plt.plot(plt_pts[:,0],plt_pts[:,1],color='r',
                        linewidth=1.5)
            
            scatter_pts_x.append(x)
            scatter_pts_y.append(y)

        plt.scatter(scatter_pts_x,scatter_pts_y,s=3,c='r')
        plt.pause(10.0)
        # add some more static frames with the robot at goal
        for _ in range(40):
            
            if self.savevid:
                self.writer.grab_frame()
            else:
                plt.pause(0.05)

        # finish writing video
        if self.savevid:
            self.writer.finish()

    def run(self, step=False):
        """run the actual Astar algorithm
        """
        i = 0
        while self.goal_reached is False and len(self.open_list) > 0:
            # pop the coord with the min cost to come
            c = heapq.heappop(self.open_list)
            curr_pose = (c.x,c.y,c.orientation)
            # print("======= New closed node: =======")
            # print(c.coord, c.orientation)
            # if c.parent:
            #     print("parent pose:",c.parent.coord, c.parent.orientation)
            # if c.parent_action:
            #     print("parent action: ",c.parent_action.v_l, c.parent_action.v_r)
            # print("================================")
            self.add_to_closed(c)
            self.add_to_closed_plot(c)

            if self.at_goal(c):
                self.goal_reached = True
                print("Path found!")
                self.backtrack(goal_coord=c)
                break
            
            # not at goal, go through reachable point from c
            # apply the motion model
            for a in self.actions:
                a : Action
                next_pose,center_rot,cost = a.apply(init_pose=(c.x,
                                                          c.y,
                                                          c.orientation)
                                            )
                
                # check within the map or not
                x,y,ori = next_pose
                if not self.map.in_range(x,y):
                    continue
                
                # check if in closed list
                
                ideg = int(ori/State.rad_res)%len(self.closed_list)
                ih = int(y/State.xy_res)
                iw = int(x/State.xy_res)
                if self.closed_list[ideg][ih][iw]:
                    continue

                # check obstacle
                if a.v_l == a.v_r:
                    # check line segment
                    collide = self.map.check_obstacle_line_seg(
                                                        x_start=c.x,
                                                        y_start=c.y,
                                                        x_end=x,
                                                        y_end=y,
                                                        pool=self.check_pool)
                else:
                    # check curves
                    collide = self.map.check_obstacle_curve(curr_pose=curr_pose,
                                                        action=a,
                                                        center_rot=center_rot,
                                                        pool=self.check_pool)
                    
                    # print(f"checking: {a.v_l, a.v_r}, {collide}")

                if collide:
                    continue
                if not self.open_list_added[ideg][ih][iw]:
                    # not added to the open list, do initialization first
                    self.initiate_coord(coord=(x,y),
                                        ori=ori,
                                        parent=c,
                                        parent_action=a,
                                        edge_cost=cost)
                else:
                    # update the coordinate
                    new_cost_to_come = c.cost_to_come + cost
                    next_s : State = self.open_list_added[ideg][ih][iw]
                    if new_cost_to_come < next_s.cost_to_come:
                        next_s.update(cost_to_come=new_cost_to_come,
                                      parent=c,
                                      parent_action=a)
                        heapq.heapify(self.open_list)
            
            if step:
                break

            # visualize the result at some fixed interval
            i+=1
            if i%50==0:
                self.visualize_search()
        
        if self.goal_reached:
            # show the path to the goal
            self.visualize_path()
            
    def backtrack(self, goal_coord : State):
        """backtrack to get the path to the goal from the initial position

        """
        self.path_to_goal = []
        c = goal_coord
        action = None
        while not c.same_state_as(self.init_coord,ignore_ori=True):
            c : State
            self.path_to_goal.append([c.coord,c.orientation,action])
            action = c.parent_action
            c = c.parent
            
        self.path_to_goal.append([c.coord,c.orientation,action])
        self.path_to_goal.reverse()

def ask_for_coord(map:Map, mode="initial"):
    """function for asking user input of init or goal coordinate; if user input
    is not valid, ask again

    Args:
        msg (_type_): _description_
    """
    while True:
        x = float(input(f"Please input {mode} coordinate x: "))
        y = float(input(f"Please input {mode} coordinate y: "))

        if x<0 or x>=map.width or y<0 or y>=map.height:
            print("Coordinate out of range of map, please try again")
            continue

        if map.map_inflate[int(y),int(x)] > 0:
            print("Coordinate within obstacle, please try again")
            continue
        
        if mode=="initial":
            ori=float(input(f"Please input {mode} orientation (degrees): "))
        else:
            ori=None
        
        break
    return (x,y), ori

class velocity_publisher(Node):
    def __init__(self, f, action_list):
        super().__init__('astar_planning_node')
        self.timer_period = f
        self.cmd_vel_pub = self.create_publisher(Twist,'/cmd_vel', 10)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.i = 0
        self.action_list = action_list
        

    def timer_callback(self):
        msg = Twist()

        if self.i >= len(self.action_list):
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            return
        
        a:Action = self.action_list[self.i]
        
        ur = a.v_r
        ul = a.v_l

        linear_vel = 0.5 * (ur+ul)
        ang_vel = radius_wheel_tb * (ur-ul) / D_tb

        msg.linear.x = linear_vel
        msg.angular.z = ang_vel

        # publish

        self.i+=1
        
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--savevid", type=bool, default=False,
                        help="whether to save the demo as video")
    parser.add_argument("--dpi", type=int, default=300,
                        help="resolution of the video saved")
    parser.add_argument("--gazebo", type=bool, default=False,
                        help="in gazebo sim or not")
    parser.add_argument("--rr", type=int, default=20,
                        help="robot radius")
    parser.add_argument("--dt", type=float, default=20.0,
                        help="step time")
    parser.add_argument("--cogw", type=float, default=1.0,
                        help="additional weight of cost to go, default to 1.0")
    parser.add_argument("--rpm1", type=float, default=0.5,
                        help="rpm 1")
    parser.add_argument("--rpm2", type=float, default=1.0,
                        help="rpm 2")
    args = parser.parse_args()

    if args.gazebo:
        State.xy_res = D_tb/4.0
    else:
        State.xy_res = D_sim/4.0

    # create map object
    custom_map = Map(inflate_radius=args.rr)

    # define the corners of all the convex obstacles
    obs_corners = []
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(1500,2000),w=250,h=1000))
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(2500,1000),w=250,h=1000))
    obs_corners.append(custom_map.get_corners_hex(
                                            center=(4200,1200),radius=600))
    
    # add all obstacles to map
    for c in obs_corners:
        custom_map.add_obstacle(corners_tuple=c)

    # get the inflated obstacle corners
    rr = args.rr
    corners = custom_map.get_obstacle_corners_array()

    # # ask user for init and goal position
    # init_coord,init_ori = ask_for_coord(custom_map, mode="initial")
    # goal_coord,goal_ori = ask_for_coord(custom_map, mode="goal")

    init_coord = (500,1000)
    init_ori = 0

    goal_coord = (5750,1000)

    vt = VisTree(corners=corners,goal_coord=goal_coord,
             boundary=custom_map.obstacle_boundary_inflate,
             inflate_coef=args.cogw)
    
    # create Astar solver
    a = Astar(init_coord=init_coord,
              init_ori=init_ori,
              goal_coord=goal_coord,
              rpms=[args.rpm1,args.rpm2],
              map=custom_map,
              vis_tree=vt,
              savevid=args.savevid,
              vid_res=args.dpi,
              dt=args.dt,
              gazebo=args.gazebo
              )

    # run the algorithm
    a.run()
    
    goal_path = a.path_to_goal

    rclpy.init(args=None)
    node = velocity_publisher(f=1, action_list=goal_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



    