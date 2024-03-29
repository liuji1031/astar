#########################################################
## repo link: https://github.com/liuji1031/astar       ##
#########################################################

import argparse
import numpy as np
import heapq
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class Action:
    radius_wheel = 33 # diameter 66 mm
    D = 287 # distance between two wheels

    def __init__(self, rpm_left, rpm_right,
                 dt=0.1) -> None:
        self.rpm_left = rpm_left
        self.rpm_right = rpm_right
        self.dt = dt

        self.w_l = 2*np.pi/60.0*rpm_left
        self.w_r = 2*np.pi/60.0*rpm_right # angular vel of left and
                                                       # right wheel
        self.v_l = self.w_l*Action.radius_wheel
        self.v_r = self.w_r*Action.radius_wheel
        vel_sum = self.v_l+self.v_r
        vel_diff = self.v_l-self.v_r
        if vel_diff != 0:
            self.turn_radius = Action.D/2*vel_sum/np.abs(vel_diff)

        else:
            self.turn_radius = None
        self.ang_vel = 2*np.abs(vel_diff)/Action.D
        self.dyaw = self.ang_vel*self.dt
    
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
        
        return new_pose, center_rot
    
    def curve_motion(self, init_pose):
        """apply curved motion to the init pose

        Args:
            init_pose (_type_): _description_
        """
        x0,y0,yaw0 = init_pose

        if self.v_r>self.v_l: # turn towards left
            phi0 = yaw0-np.pi/2
        else: # turn towards right
            phi0 = yaw0+np.pi/2
        # compute center of rotation
        cp = np.cos(phi0)
        sp = np.sin(phi0)
        cx = x0-self.turn_radius*cp
        cy = y0-self.turn_radius*sp

        phi1 = phi0+self.dyaw*(self.v_r>self.v_l)
        x1 = cx+self.turn_radius*np.cos(phi1)
        y1 = cy+self.turn_radius*np.sin(phi1)
        yaw1 = yaw0+self.dyaw*(self.v_r>self.v_l)

        return (x1,y1,yaw1),(cx,cy)
    
    def straight_motion(self, init_pose):
        """apply straight motion

        Args:
            init_pose (_type_): _description_
        """
        x0,y0,yaw0 = init_pose

        x1 = x0+self.v_l*self.dt*np.cos(yaw0)
        y1 = y0+self.v_l*self.dt*np.sin(yaw0)
        yaw1 = yaw0

        return (x1,y1,yaw1)

def wrap(rad):
    """wrap a angle between 0 to 2pi

    Args:
        rad (_type_): _description_
    """
    twopi = 2*np.pi
    while rad > twopi:
        rad-=twopi
    
    while rad<0:
        rad+=twopi

    return rad

def collision_check_curve(pose,
                          v_l,
                          v_r,
                          cx,
                          cy,
                          turn_radius,
                          dyaw,
                          boundaries):
    """this function check if the curved path the robot is on will collide 
    with obstacles

    Args:
        start_xy (_type_): the starting position of the robot, tuple of (x,y)
        rpm (_type_): tuple of left and right wheel rpm
        boundaries (_type_): a list of list, containing boundary normal and 
        projection value defining the boundary
    """
    x,y,yaw = pose
    
    min_val = 0.0
    max_val = dyaw

    if v_r>v_l: # turn towards left
        phi0 = yaw-np.pi/2
    else: # turn towards right
        phi0 = yaw+np.pi/2
    # compute center of rotation
    cp = np.cos(phi0)
    sp = np.sin(phi0)
    cx = x-turn_radius*cp
    cy = y-turn_radius*sp

    if v_l>v_r:
        phi0-=dyaw
    # print(cx,cy,phi0/np.pi*180)

    for ib,b in enumerate(boundaries):
        # print(f"=== boundary {ib+1}")
        b:dict
        if b["type"]=="polygon": # obstacle defined by a polygon
            normal, proj = b["normal"], b["proj"]
            nx = normal[0]
            ny = normal[1]
            # convert to frame defined by phi0, cx, cy
            proj_ = proj-nx*cx-ny*cy
            nx_ = cp*nx+sp*ny
            ny_ = -sp*nx+cp*ny
            theta_hat = wrap(np.arctan2(ny_,nx_))
            # print("theta hat: ",theta_hat/np.pi*180)
            rho = proj_/turn_radius
            
            if rho>=1:
                # print("condition 1, full range")
                continue
            elif rho>=-1 and rho<1:
                cos_inv_rho = np.arccos(rho)
                theta1 = cos_inv_rho+theta_hat
                theta2 = 2*np.pi-cos_inv_rho+theta_hat
                # print("condition 2",rho,theta1,theta2)
                if theta1>2*np.pi or theta2>2*np.pi:
                    theta1-=2*np.pi
                    theta2-=2*np.pi
                # print("\tcorrection",theta1,theta2)
                min_val = max(min_val,theta1)
                max_val = min(max_val,theta2)
            elif rho<-1:
                # print("condition 3",rho)
                return False
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
        normal,p = boundary[i]

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
                 width=1200,
                 height=500,
                 inflate_radius=5,
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
            boundary.append([normal,p])
            boundary_inflate.append([normal,p+self.inflate_radius])
        
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
        plt.imshow(self.map+self.map_inflate,cmap="gray_r",vmax=8)
        plt.gca().invert_yaxis()
        # plt.colorbar(shrink=0.3)
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

    def check_obstacle(self, x_start, y_start,
                       x_end, y_end,
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
        theta = np.pi/2 + np.linspace(0., -2*np.pi, 6, endpoint=False)
        radius_inflate = radius + self.inflate_radius/np.sqrt(3)*2

        corners = np.hstack([
                    (center[0]+radius*np.cos(theta))[:,np.newaxis],
                    (center[1]+radius*np.sin(theta))[:,np.newaxis],
                    ])
        corners_inflate = np.hstack([
                    (center[0]+radius_inflate*np.cos(theta))[:,np.newaxis],
                    (center[1]+radius_inflate*np.sin(theta))[:,np.newaxis],
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
    xy_res = 0.5
    rad_res = 3.0/180.0*np.pi
    def __init__(self,
                 coord,
                 orientation,
                 cost_to_come,
                 cost_to_go=None,
                 parent=None,
                 vt_node=None) -> None:

        self.coord = round_to_precision(coord,precision=State.xy_res)
        self.orientation = round_to_precision(wrap(orientation),
                                              precision=State.rad_res)
        self.cost_to_come = cost_to_come
        self.cost_to_go = cost_to_go
        self.estimated_cost = self.cost_to_come+self.cost_to_go
        self.parent = parent
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
    
    def update(self, cost_to_come, parent):
        self.cost_to_come = cost_to_come
        self.estimated_cost = self.cost_to_come+self.cost_to_go
        self.parent = parent
    
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
                 rpms=[50,100],
                 step_size=10,
                 dtheta=3,
                 coord_res=0.5,
                 savevid=False,
                 vid_res=72,
                 goal_ori=None,
                 dt=0.1):
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
        self.step_size = step_size
        self.dtheta = dtheta

        self.open_list = [self.init_coord]
        heapq.heapify(self.open_list)
        # use a dictionary to track which coordinate has been added to the open
        # list
        ndeg = int(360/self.dtheta)
        nw = int(map.width/coord_res)+1
        nh = int(map.height/coord_res)+1
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
                self.actions.append(Action(rpm_left=rpm_l,
                                           rpm_right=rpm_r,
                                           dt=dt)
                                    )
        
        # create the handles for the plots
        self.fig = plt.figure(figsize=(12,6))
        self.ax = self.fig.add_subplot()
        self.ax.invert_yaxis()
        # show the map
        self.map_plot = self.ax.imshow( self.map_plot_data,
                                        cmap='bone_r',vmin=0,vmax=6,
                                        extent=(0,self.map.width,0,
                                               self.map.height),
                                        resample=False,
                                        aspect='equal',
                                        origin='lower',
                                        interpolation='none')
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
        self.closed_list[ideg][ih][iw] = c

    def at_goal(self, c : State):
        """return true if c is at goal coordinate

        Args:
            c (State): _description_
        """
        dx = c.x-self.goal_coord.x
        dy = c.y-self.goal_coord.y
        rad1 = c.orientation/180*np.pi
        v1 = np.array([np.cos(rad1),np.sin(rad1)])

        rad2 = self.goal_coord.orientation/180*np.pi
        v2 = np.array([np.cos(rad2),np.sin(rad2)])

        proj = np.inner(v1,v2).item()

        return (np.linalg.norm((dx,dy))<1.5*self.map.inflate_radius)&(proj>=0.86)

    
    def initiate_coord(self,
                       coord,
                       ori,
                       parent : State,
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
                    vt_node=vt_node)
        
        # push to open list heaqp
        heapq.heappush(self.open_list, new_c)
        
        # mark as added
        ideg, ih, iw = new_c.index
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
        x,y = state.x,state.y
        rad = state.orientation/180*np.pi
        # print(x,y,rad,state.orientation)
        for i in range(0,1):
            if i==0:
                c='b'
                lw=1
            else:
                c=(0.8,0.8,0.8)
                lw=0.5
            x_ = x+5*np.cos(rad)
            y_ = y+5*np.sin(rad)
            xarr = np.array([x,x_,None])
            yarr = np.array([y,y_,None])
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
        if not self.closed_plot:
            self.closed_plot = plt.plot(self.closed_plot_data_x,
                                        self.closed_plot_data_y,
                                        color=(3/255, 198/255, 252/255),
                                        linewidth=0.5)[0]
        else:
            # update only
            self.closed_plot.set_data(self.closed_plot_data_x,
                                      self.closed_plot_data_y)

        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        if self.savevid:
            self.writer.grab_frame()
        plt.pause(0.0001)

    def visualize_path(self):
        """visualize the result of backtrack
        """
        path = np.array(self.path_to_goal)
        plt.plot(path[:,0],path[:,1],color='r',marker="o",linewidth=1.5,ms=3)
        plt.pause(3.0)
        # add some more static frames with the robot at goal
        for _ in range(40):
            
            if self.savevid:
                self.writer.grab_frame()
            else:
                plt.pause(0.05)

        # finish writing video
        if self.savevid:
            self.writer.finish()

    def run(self):
        """run the actual Astar algorithm
        """
        i = 0
        while self.goal_reached is False and len(self.open_list) > 0:
            # pop the coord with the min cost to come
            c = heapq.heappop(self.open_list)
            self.add_to_closed(c)
            self.add_to_closed_plot(c)

            if self.at_goal(c):
                self.goal_reached = True
                print("Path found!")
                self.backtrack(goal_coord=c)
                break
            
            # not at goal, go through reachable point from c
            # apply the motion model
            next_states, action_costs = motion_model_proj3(curr_coord=[c.x,c.y],
                                                        curr_ori=c.orientation,
                                                        L=self.step_size)
            for next_state, cost in zip(next_states,action_costs):
                x,y,ori = next_state

                if not self.map.in_range(x,y):
                    continue

                ideg = int(ori/self.dtheta)
                ih = int(y/0.5)
                iw = int(x/0.5)
                self.map : Map

                # skip if new coord in closed list already
                if self.closed_list[ideg][ih][iw]:
                    continue

                # skip if overlap with obstacle
                if self.map.check_obstacle(x_start=c.x,
                                            y_start=c.y,
                                            x_end=x,
                                            y_end=y,
                                            pool=self.check_pool):
                    continue

                if not self.open_list_added[ideg][ih][iw]:
                    # not added to the open list, do initialization first
                    self.initiate_coord(coord=(x,y),
                                        ori=ori,
                                        parent=c,
                                        edge_cost=cost)
                else:
                    # update the coordinate
                    new_cost_to_come = c.cost_to_come + cost
                    next_s : State = self.open_list_added[ideg][ih][iw]
                    if new_cost_to_come < next_s.cost_to_come:
                        next_s.update(new_cost_to_come,c)
                        heapq.heapify(self.open_list)

            # visualize the result at some fixed interval
            i+=1
            if i%10==0:
                self.visualize_search()
        
        if self.goal_reached:
            # show the path to the goal
            self.visualize_path()
            
    def backtrack(self, goal_coord : State):
        """backtrack to get the path to the goal from the initial position

        """
        self.path_to_goal = []
        c = goal_coord
        while not c.same_state_as(self.init_coord,ignore_ori=True):
            self.path_to_goal.append(c.coord)
            c = c.parent

        self.path_to_goal.append(c.coord)
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--savevid", type=bool, default=False,
                        help="whether to save the demo as video")
    parser.add_argument("--dpi", type=int, default=72,
                        help="resolution of the video saved")
    parser.add_argument("--rr", type=int, default=220,
                        help="robot radius")
    parser.add_argument("--step", type=int, default=20,
                        help="robot radius")
    parser.add_argument("--cogw", type=float, default=1.0,
                        help="additional weight of cost to go, default to 1.0")
    args = parser.parse_args()
    savevid = args.savevid
    rr = args.rr
    # create map object
    custom_map = Map(inflate_radius=args.rr)

    # define the corners of all the convex obstacles
    obs_corners = []
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(100,500),w=75,h=400))
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(275,400),w=75,h=400))
    obs_corners.append(custom_map.get_corners_hex(
                                            center=(650,250),radius=150))
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(900,450),w=200,h=75))
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(1020,375),w=80,h=250))
    obs_corners.append(custom_map.get_corners_rect(
                                            upper_left=(900,125),w=200,h=75))

    # add all obstacles to map
    for c in obs_corners:
        custom_map.add_obstacle(corners_tuple=c)

    # get the inflated obstacle corners
    corners = custom_map.get_obstacle_corners_array(omit=[(3,2),
                                                          (4,1),
                                                          (4,2),
                                                          (5,1)],
                                                correction={(4,0):[0,-rr*2],
                                                (4,3):[0,rr*2]})

    # ask user for init and goal position
    init_coord,init_ori = ask_for_coord(custom_map, mode="initial")
    goal_coord,goal_ori = ask_for_coord(custom_map, mode="goal")
        
    # init_coord = (5,100)
    # init_ori = -90
    # goal_coord = (800,200)
    # goal_ori = -90

    vt = VisTree(corners=corners,goal_coord=goal_coord,
             boundary=custom_map.obstacle_boundary_inflate,
             inflate_coef=args.cogw)
    
    # create Astar solver
    a = Astar(init_coord=init_coord,
              init_ori=init_ori,
              goal_coord=goal_coord,
              goal_ori=goal_ori,
              map=custom_map,
              vis_tree=vt,
              step_size=args.step,
              savevid=savevid,
              vid_res=args.dpi,
              )

    # run the algorithm
    a.run()
    