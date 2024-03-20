import argparse
import numpy as np
import heapq
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def within_obstacle(x_start,y_start,
                    x_end,y_end,
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
    intersection = []
    for i in range(n):
        # get normal direction
        normal,p = boundary[i]

        # find the meshgrid points whose projections are <= p
        p1 = np.inner((x_start,y_start), normal)
        p2 = np.inner((x_end,y_end), normal)

        # now see if (1-rho)*p1+rho*p2 for rho>=0 and rho<=1 has any range of
        # rho that makes the value < p+inflate_radius
        if p1 > p and p2 > p:
            return False
        elif p1 <= p and p2 <= p:
            intersection.append([0.0,1.0]) # full range
        elif p1 > p and p2 <= p:
            intersection.append([(p1-p)/(p1-p2), 1.0])
        elif p1 < p and p2 >= p:
            intersection.append([0.0, (p1-p)/(p1-p2)])

    # compute the final intersection
    min_val = 0.0
    max_val = 1.0
    for i in intersection:
        min_val = max(min_val, i[0])
        max_val = min(max_val, i[1])

    # if the final intersection is non empty, it means some of the line segment
    # is within the obstacle
    return min_val<=max_val


class Map:
    """class representing the map
    """
    def __init__(self,
                 width=1200,
                 height=500,
                 inflate_radius=5
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
        self.map = np.zeros((height, width),dtype=np.int8) # 0: obstacle free
                                                           # 1: obstacle
        self.map_inflate = np.zeros_like(self.map)
        self.inflate_radius = inflate_radius
        self.obstacle_corners = []
        self.obstacle_boundary = []

    def add_obstacle(self, corners : np.ndarray):
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

        n = corners.shape[0]
        boundary = []
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
            boundary.append([normal,p+self.inflate_radius])
        
        self.obstacle_boundary.append(boundary)
        
        # find points that meet all half plane conditions
        obs_map = np.where(obs_map==n,1,0)
        obs_map_inflate = np.where(obs_map_inflate==n,1,0)

        # add to the existing map
        self.map = np.where(obs_map==1,obs_map,self.map)
        self.map_inflate = np.where(obs_map_inflate==1,
                                    obs_map_inflate,self.map_inflate)

    def plot(self,show=True):
        """show the map

        Args:
            show (bool, optional): _description_. Defaults to True.
        """
        plt.imshow(self.map+self.map_inflate)
        plt.gca().invert_yaxis()
        plt.colorbar()
        if show:
            plt.show()

    def in_range(self, x, y):
        """return true if (x, y) within the range of the map

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        if x>=0 and x<self.width and y>=0 and y<self.height:
            return True
        else:
            return False

    def on_obstacle(self, x, y, use_inflate=True):
        """check if x, y coord is a valid point on the map, i.e., within range
        and obstacle free

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        if use_inflate:
            if self.map_inflate[y,x] < 1:
                return False
            else:
                return True
        else:
            if self.map[y,x] < 1:
                return False
            else:
                return True

    def check_obstacle(self, x_start, y_start,
                       x_end, y_end,
                       pool=None):
        """go through all obstacles to check if x,y is within any one of them

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        if pool is None:
            for boundary in self.obstacle_boundary:
                # returns true if within any obstacle
                if within_obstacle(x_start,
                                   y_start,
                                   x_end,
                                   y_end,
                                   boundary):
                    return True
            return False
        else:
            nobs = len(self.obstacle_corners)
            inputs = zip((x_start,)*nobs,
                         (y_start,)*nobs,
                         (x_end,)*nobs,
                         (y_end,)*nobs,
                         self.obstacle_boundary)
            outputs = pool.starmap(within_obstacle,inputs)

            for out in outputs:
                if out:
                    return True
            return False
    
    @staticmethod
    def get_corners_hex(center, radius):
        """get the hexagon corner points

        Args:
            center (_type_): _description_
            radius (_type_): _description_

        Returns:
            _type_: _description_
        """
        theta = np.pi/2 + np.linspace(0., -2*np.pi, 6, endpoint=False)
        return np.hstack([
            (center[0]+radius*np.cos(theta))[:,np.newaxis],
            (center[1]+radius*np.sin(theta))[:,np.newaxis],
        ])
    
    @staticmethod
    def get_corners_rect(upper_left,
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
        return corners

def round_to_precision(data,
                        precision=0.5):
    """round the coordinate according to the requirement, e.g., 0.5

    Args:
        data (_type_): _description_
        precision (float, optional): _description_. Defaults to 0.5.
    """
    if isinstance(data, tuple):
        x_ = np.round(data[0]/0.5)*0.5
        y_ = np.round(data[1]/0.5)*0.5
        return (x_,y_)
    else:
        return np.round(data/0.5)*0.5

class State:
    """create a custom class to represent each map coordinate.
    attribute cost_to_come is used as the value for heap actions.
    for this purpose, the <, > and = operations are overridden

    """
    def __init__(self,
                 coord,
                 orientation,
                 cost_to_come,
                 cost_to_go=None,
                 parent=None) -> None:
        self.coord = round_to_precision(coord)
        # orientation: discrete value for this problem, i.e., 30, 60 etc
        self.orientation = orientation
        self.cost_to_come = cost_to_come
        self.cost_to_go = cost_to_go
        self.estimated_cost = self.cost_to_come+self.cost_to_go
        self.parent = parent

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

        if not ignore_ori:
            return (np.linalg.norm((dx,dy))<1.5) & \
                (np.abs(self.orientation-other.orientation) < 1e-8)
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
        ideg = int(self.orientation/30)
        iw = int(self.x/0.5)
        ih = int(self.y/0.5)
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
    action_cost = []

    for i in range(-2,3):
        deg = i*dtheta
        new_ori = (curr_ori+deg)%360
        rad = new_ori/180*np.pi
        x1,y1 = round_to_precision((x0 + L*np.cos(rad),
                                     y0 + L*np.sin(rad))
                                    )
        
        next_state.append((x1,y1,new_ori))
        action_cost.append(L+deg_coef*np.abs(deg))
    
    return next_state, action_cost

class Astar:
    # implement the Astar search algorithm

    def __init__(self,
                 init_coord,
                 init_ori,
                 goal_coord,
                 goal_ori,
                 map : Map,
                 step_size=10,
                 dtheta=30,
                 coord_res=0.5,
                 savevid=False):
        
        self.init_coord = State(init_coord,
                                   init_ori,
                                   cost_to_come=0.0,
                                   cost_to_go=cost_to_go_l2(init_coord,
                                                            goal_coord))
        self.goal_coord = State(goal_coord,
                                   goal_ori,
                                   cost_to_come=np.inf,
                                   cost_to_go=0.0)
        self.map = map
        self.savevid = savevid
        self.step_size = step_size
        self.dtheta = dtheta

        # use multi processing to check for obstacles
        self.check_pool = multiprocessing.Pool()

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
            self.writer.setup(self.fig, outfile="./animation.mp4",dpi=72)

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
        # self.closed_plot_data[c.y][c.x] = 1

    def at_goal(self, c : State):
        """return true if c is at goal coordinate

        Args:
            c (State): _description_
        """
        return self.goal_coord.same_state_as(c)
    
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
        new_c = State(coord=coord,
                         orientation=ori,
                         cost_to_come=parent.cost_to_come+edge_cost,
                         cost_to_go=cost_to_go_l2(coord,self.goal_coord),
                         parent=parent)
        
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
            x_ = x+8*np.cos(rad)
            y_ = y+8*np.sin(rad)
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
                                        color='b',
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
        plt.plot(path[:,0],path[:,1],color='r',linewidth=1)

        n = path.shape[0]
        ind = np.linspace(0,n-1,50).astype(int)
        for i in ind:
            self.robot_plot.set_data([path[i,0],],[path[i,1],])
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            if self.savevid:
                self.writer.grab_frame()
            plt.pause(0.001)
        
        # add some more static frames with the robot at goal
        for _ in range(40):
            plt.pause(0.005)
            if self.savevid:
                self.writer.grab_frame()

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
                ideg = int(ori/self.dtheta)
                ih = int(y/0.5)
                iw = int(x/0.5)
                self.map : Map
                # skip if new coord not valid
                if not self.map.in_range(x,y) or \
                    self.map.check_obstacle(x_start=c.x,
                                            y_start=c.y,
                                            x_end=x,
                                            y_end=y,
                                            pool=self.check_pool):
                    continue
                
                # skip if new coord in closed list already
                if self.closed_list[ideg][ih][iw]:
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
            if i%100==0:
                self.visualize_search()
        
        if self.goal_reached:
            # show the path to the goal
            self.visualize_path()
            
    def backtrack(self, goal_coord : State):
        """backtrack to get the path to the goal from the initial position

        """
        print("running back track")
        self.path_to_goal = []
        c = goal_coord
        print(c.coord)
        print(self.init_coord.coord)
        while not c.same_state_as(self.init_coord,ignore_ori=True):
            self.path_to_goal.append(c.coord)
            c = c.parent

        self.path_to_goal.append(c.coord)
        self.path_to_goal.reverse()
        print(self.path_to_goal)

def ask_for_coord(map:Map, mode="initial"):
    """function for asking user input of init or goal coordinate; if user input
    is not valid, ask again

    Args:
        msg (_type_): _description_
    """
    while True:
        x = input(f"Please input {mode} coordinate x: ")
        y = input(f"Please input {mode} coordinate y: ")

        if x<0 or x>=map.width or y<0 or y>=map.height:
            print("Coordinate out of range of map, please try again")
            continue

        if map.map_inflate[int(y),int(x)] > 0:
            print("Coordinate within obstacle, please try again")
            continue

        ori = input(f"Please input {mode} orientation (degrees): ")
        ori = int(ori/30)*30
        
        break
    return (x,y), ori

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--savevid", type=bool, default=False,
                        help="whether to save the demo as video")
    args = parser.parse_args()
    savevid = args.savevid

    # create map object
    custom_map = Map()

    # define the corners of all the convex obstacles
    obs_corners = []
    obs_corners.append(Map.get_corners_rect(upper_left=(100,500),w=75,h=400))
    obs_corners.append(Map.get_corners_rect(upper_left=(275,400),w=75,h=400))
    obs_corners.append(Map.get_corners_hex(center=(650,250),radius=150))
    obs_corners.append(Map.get_corners_rect(upper_left=(900,450),w=200,h=75))
    obs_corners.append(Map.get_corners_rect(upper_left=(1020,375),w=80,h=250))
    obs_corners.append(Map.get_corners_rect(upper_left=(900,125),w=200,h=75))

    # add all obstacles to map
    for c in obs_corners:
        custom_map.add_obstacle(corners=c)

    # ask user for init and goal position
    # init_coord,init_ori = ask_for_coord(custom_map, mode="initial")
    # goal_coord,goal_ori = ask_for_coord(custom_map, mode="goal")
        
    init_coord = (10,10)
    init_ori = 0
    goal_coord = (600,100)
    goal_ori = 30

    # create Astar solver
    d = Astar(init_coord=init_coord,
              init_ori=init_ori,
              goal_coord=goal_coord,
              goal_ori=goal_ori,
              map=custom_map,
              step_size=40,
              savevid=savevid)

    # run the algorithm
    d.run()
    