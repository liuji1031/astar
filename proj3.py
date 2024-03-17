import argparse
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

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
        n = corners.shape[0]
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
        
class MapCoord:
    """create a custom class to represent each map coordinate.
    attribute cost_to_come is used as the value for heap actions.
    for this purpose, the <, > and = operations are overridden

    """
    def __init__(self,
                 coord,
                 cost_to_come,
                 cost_to_go=None,
                 parent=None) -> None:
        self.coord = coord
        self.cost_to_come = cost_to_come
        self.cost_to_go = cost_to_go
        self.parent = parent

    def __lt__(self, other):
        return self.estimated_cost < other.estimated_cost
    
    def __gt__(self, other):
        return self.estimated_cost > other.estimated_cost
    
    def __eq__(self, other):
        return self.estimated_cost == other.estimated_cost
    
    def set_parent(self, parent):
        self.parent = parent

    def same_coord(self, other):
        return self.coord == other.coord
    
    def update(self, cost_to_come, parent):
        self.cost_to_come = cost_to_come
        self.parent = parent
    
    @property
    def x(self):
        return self.coord[0]
    
    @property
    def y(self):
        return self.coord[1]
    
    @property
    def estimated_cost(self):
        return self.cost_to_come+self.cost_to_go
    
    @staticmethod
    def calculate_cost_to_go(coord1, coord2):
        """calculate the optimistic cost to go estimate

        Args:
            coord1 (_type_): _description_
            coord2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.linalg.norm([coord1.x-coord2.x, coord1.y-coord2.y],ord=2)

class Astar:
    # implement the Astar search algorithm

    def __init__(self,
                 init_coord,
                 goal_coord,
                 map : Map,
                 savevid=False):
        
        self.init_coord = MapCoord(init_coord, cost_to_come=0.0)
        self.goal_coord = MapCoord(goal_coord, cost_to_come=np.inf)
        self.map = map
        self.savevid = savevid

        self.open_list = [self.init_coord]
        heapq.heapify(self.open_list)
        # use an array to track which coordinate has been added to the open list
        self.open_list_added = \
            [[None for i in range(map.width)] for j in range(map.height)]

        # use an array to store the visited map coordinates; 
        # None means not visited. otherwise, stores the actual MapCoord obj
        self.closed_list = \
            [[None for i in range(map.width)] for j in range(map.height)]

        self.goal_reached = False
        self.path_to_goal = None

        # create the list of possible actions
        self.actions = []
        for ax in np.arange(-1,2):
            for ay in np.arange(-1,2):
                if ax==0 and ay==0:
                    continue
                else:
                    self.actions.append((ax,ay))
        self.actions = np.array(self.actions)
        self.actions_cost = np.round(np.linalg.norm(self.actions,ord=2,axis=1),
                                     decimals=1)
        
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
        # create an array of 0s and 1s to track closed list, for plot purpose
        self.closed_plot_data = np.zeros_like(self.map.map)
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

    def add_to_closed(self, c : MapCoord):
        """add the popped coordinate to the closed list

        Args:
            c (MapCoord): _description_
        """
        self.closed_list[c.y][c.x] = c
        self.closed_plot_data[c.y][c.x] = 1

    def at_goal(self, c : MapCoord):
        """return true if c is at goal coordinate

        Args:
            c (MapCoord): _description_
        """
        return self.goal_coord.same_coord(c)
    
    def initiate_coord(self, coord, parent : MapCoord, edge_cost):
        """initiate new coordinate to be added to the open list

        Args:
            coord (_type_): _description_
            parent (_type_): _description_
        """
        # create new MapCoord obj
        new_c = MapCoord(coord=coord,
                         cost_to_come=parent.cost_to_come+edge_cost,
                         parent=parent)
        
        # push to open list heaqp
        heapq.heappush(self.open_list, new_c)
        
        # mark as added
        self.open_list_added[new_c.y][new_c.x] = new_c

    def print_open_len(self):
        print("current open list length: ", len(self.open_list))

    def update_coord(self, x, y, new_cost_to_come, parent):
        """update the coordinate with new cost to come and new parent

        Args:
            x (_type_): _description_
            y (_type_): _description_
            new_cost_to_come (_type_): _description_
            parent (_type_): _description_
        """
        self.open_list_added[y][x].update(new_cost_to_come,parent)
    
    def on_obstacle(self, x, y):
        """check if coord (x,y) is on the obstacle
        return true if there is obstacle

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        return self.map[y,x]>0

    def visualize_search(self):
        """visualize the search process
        """
        self.map_plot.set_data(self.map_plot_data+self.closed_plot_data)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        if self.savevid:
            self.writer.grab_frame()
        plt.pause(0.001)

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
        for _ in range(20):
            plt.pause(0.005)
            if self.savevid:
                self.writer.grab_frame()

        # finish writing video
        if self.savevid:
            self.writer.finish()

    def run(self):
        """run the actual Dikstra's algorithm
        """
        i = 0
        while self.goal_reached is False and len(self.open_list) > 0:
            # pop the coord with the min cost to come
            c = heapq.heappop(self.open_list)
            self.add_to_closed(c)

            if self.at_goal(c):
                self.goal_reached = True
                print("Path found!")
                self.backtrack(goal_coord=c)
                break
            
            # not at goal, go through reachable point from c
            for a, cost in zip(self.actions,self.actions_cost):
                x,y = c.coord[0]+a[0], c.coord[1]+a[1]
                
                self.map : Map
                # skip if new coord not valid
                if self.map.in_range(x,y) is False or \
                    self.map.on_obstacle(x,y) is True:
                    continue
                
                # skip if new coord in closed list already
                if self.closed_list[y][x] is not None:
                    continue

                if self.open_list_added[y][x] is None: 
                    # not added to the open list, do initialization first
                    self.initiate_coord(coord=(x,y),parent=c,edge_cost=cost)
                else:
                    # update the coordinate
                    cost_to_come_ = c.cost_to_come + cost
                    next_c : MapCoord = self.open_list_added[y][x]
                    if cost_to_come_ < next_c.cost_to_come:
                        next_c .update(cost_to_come_,c)
                        heapq.heapify(self.open_list)

            # visualize the result at some fixed interval
            i+=1
            if i%4000==0:
                self.visualize_search()
        
        if self.goal_reached:
            # show the path to the goal
            self.visualize_path()
            
    def backtrack(self, goal_coord : MapCoord):
        """backtrack to get the path to the goal from the initial position

        """
        self.path_to_goal = []
        c = goal_coord

        while c.same_coord(self.init_coord) is False:
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
        x = input(f"Please input {mode} coordinate x: ")
        y = input(f"Please input {mode} coordinate y: ")

        x = int(x)
        y = int(y)

        if x<0 or x>=map.width or y<0 or y>=map.height:
            print("Coordinate out of range of map, please try again")
            continue
        
        if map.map_inflate[y,x] > 0:
            print("Coordinate within obstacle, please try again")
            continue
        
        break
    return (x,y)

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
    init_coord = ask_for_coord(custom_map, mode="initial")
    goal_coord = ask_for_coord(custom_map, mode="goal")

    # create Astar solver
    d = Astar(init_coord=init_coord,goal_coord=goal_coord,map=custom_map,
                 savevid=savevid)

    # run the algorithm
    d.run()
    