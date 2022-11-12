from cmath import inf
import sys
import math
import numpy as np
import gym
import pygame
from pygame.locals import *
from racecar import Racecar

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout.



    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    MANUAL TRAINING LOOP WITHOUT THE TRAINING!

    Action vector is specified up front for the episode, then it is fed into the environment in a loop.
    This is for testing the environment.  The NN i not involved.

    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

"""

# Here are the actions to be taken at each time step (scaled) as [accel, LC cmd]
actions = [
    [0.0],
]

def main(argv):

    prng = np.random.default_rng()

    # Handle any args
    if len(argv) > 3  or  argv[1] == "--help":
        print("Usage is: {} [mode [starting lane]]".format(argv[0]))
        print("           mode h = hand-entered one at a time, mode p = pre-defined list")
        sys.exit(1)

    mode_auto = True
    start_lane = int(prng.random()*3)
    if len(argv) > 1:
        mode = argv[1]
        if mode == 'h':
            mode_auto = False

        if len(argv) > 2:
            lane = int(argv[1])
            if 0 <= lane <= 2:
                start_lane = lane

    env_config = {  "time_step_size":   0.5,
                    "debug":            0,
                    "init_ego_lane":    start_lane,
                    "training":         True        #forces vehicle to start at beginning of track
                }

    env = Racecar(env_config)
    print("///// Environment configured.")

    # Set up the graphic display
    if mode_auto:
        graphics = Graphics(env)

    # Run the agent through a complete episode
    episode_reward = 0
    done = False
    env.reset() #output is scaled obs
    step = 0
    accel = 0.0
    cont_accel = False
    while not done:
        if mode_auto:
            action = actions[step]
        else:
            action = [accel, 0.0]

        raw_obs, reward, done, info = env.step(action) #obs returned is UNSCALED
        episode_reward += reward

        # Display current status of the ego vehicle
        if mode_auto:
            graphics.update(action, raw_obs)
        #if step == 1:
        #   input("///// Press Enter to begin...")
        print("///// step {:3d}: scaled action = [{:5.2f}], speed = {:.2f}, dist = {:.3f}, r = {:7.4f} {}"
                .format(step, action[0], raw_obs[1], raw_obs[0], reward, info["reward_detail"]))
        if not cont_accel  and  not mode_auto:
            opt = input("Next scaled accel? ")
            if opt == 'c':
                cont_accel = True
            else:
                acc = float(opt)
                if -1.0 <= acc <= 1.0:
                    accel = acc

        step += 1

        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))
            input("///// Press Enter to close...")
            sys.exit()

            # Get user input before closing the window
            for event in pygame.event.get(): #this isn't working
                if event.type == pygame.QUIT:
                    graphics.close()
                    sys.exit()


######################################################################################################
######################################################################################################


class Graphics:

    # set up the colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (168, 168, 0)

    # Other graphics constants
    LANE_WIDTH = 30.0 #m (wider than reality for graphics aesthetics)
    WINDOW_SIZE_X = 1800
    WINDOW_SIZE_Y = 800
    REAL_TIME_RATIO = 8.0   #Factor faster than real time


    def __init__(self,
                 env    : gym.Env
                ):
        """Initializes the graphics and draws the roadway background display."""

        # Save the environment for future reference
        self.env = env

        # set up pygame
        pygame.init()
        self.pgclock = pygame.time.Clock()
        self.display_freq = Graphics.REAL_TIME_RATIO / env.time_step_size

        # set up the window
        self.windowSurface = pygame.display.set_mode((Graphics.WINDOW_SIZE_X, Graphics.WINDOW_SIZE_Y), 0, 32)
        pygame.display.set_caption('cda0')

        # set up fonts
        self.basicFont = pygame.font.SysFont(None, 16)

        # draw the background onto the surface
        self.windowSurface.fill(Graphics.BLACK)

        # Loop through all segments of all lanes and find the extreme coordinates to determine our bounding box
        x_min = inf
        y_min = inf
        x_max = -inf
        y_max = -inf
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                x_min = min(x_min, seg[0], seg[2])
                y_min = min(y_min, seg[1], seg[3])
                x_max = max(x_max, seg[0], seg[2])
                y_max = max(y_max, seg[1], seg[3])

        # Add a buffer all around to ensure we have room to draw the edge lines, which are 1/2 lane width away
        x_min -= 0.5*Graphics.LANE_WIDTH
        y_min -= 0.5*Graphics.LANE_WIDTH
        x_max += 0.5*Graphics.LANE_WIDTH
        y_max += 0.5*Graphics.LANE_WIDTH

        # Define the transform between roadway coords (x, y) and display viewport pixels (r, s).  Note that
        # viewport origin is at upper left, with +s pointing downward.  Leave a few pixels of buffer on all sides
        # of the display so the lines don't bump the edge.
        buffer = 8 #pixels
        display_width = Graphics.WINDOW_SIZE_X - 2*buffer
        display_height = Graphics.WINDOW_SIZE_Y - 2*buffer
        roadway_width = x_max - x_min
        roadway_height = y_max - y_min
        ar_display = display_width / display_height
        ar_roadway = roadway_width / roadway_height
        self.scale = display_height / roadway_height     #pixels/meter
        if ar_roadway > ar_display:
            self.scale = display_width / roadway_width
        self.roadway_center_x = x_min + 0.5*(x_max - x_min)
        self.roadway_center_y = y_min + 0.5*(y_max - y_min)
        self.display_center_r = Graphics.WINDOW_SIZE_X // 2
        self.display_center_s = Graphics.WINDOW_SIZE_Y // 2

        # Loop through the lane segments and draw the left and right edge lines of each
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                self._draw_segment(seg[0], seg[1], seg[2], seg[3], Graphics.LANE_WIDTH)

        pygame.display.update()
        #time.sleep(20) #debug only

        # Initialize the previous ego vehicle location at the beginning of a lane
        self.prev_ego_r = self.scale*(self.env.roadway.lanes[0].segments[0][0] - self.roadway_center_x) + self.display_center_r
        self.prev_ego_s = Graphics.WINDOW_SIZE_Y - \
                          self.scale*(self.env.roadway.lanes[0].segments[0][1] - self.roadway_center_y) + self.display_center_s
        self.veh_radius = int(0.25 * Graphics.LANE_WIDTH * self.scale) #radius of icon in pixels


    def update(self,
               action  : list,      #vector of actions for the ego vehicle for the current time step
               obs     : list       #vector of observations of the ego vehicle for the current time step
              ):
        """Paints the new motion of the ego vehicle on the display screen."""

        # Grab the background under where we want the vehicle to appear & erase the old vehicle
        pygame.draw.circle(self.windowSurface, Graphics.BLACK, (self.prev_ego_r, self.prev_ego_s), self.veh_radius, 0)

        # Display the vehicle in its new location.  Note that the obs vector is scaled for the NN.
        new_x, new_y = self._get_vehicle_coords(obs)
        new_r = int(self.scale*(new_x - self.roadway_center_x)) + self.display_center_r
        new_s = Graphics.WINDOW_SIZE_Y - int(self.scale*(new_y - self.roadway_center_y) + self.display_center_s)
        pygame.draw.circle(self.windowSurface, Graphics.YELLOW, (new_r, new_s), self.veh_radius, 0)
        pygame.display.update()

        # Update the previous location
        self.prev_ego_r = new_r
        self.prev_ego_s = new_s

        # Pause until the next time step
        self.pgclock.tick(self.display_freq)


    def close(self):
        pygame.quit()


    def _draw_segment(self,
                      x0        : float,
                      y0        : float,
                      x1        : float,
                      y1        : float,
                      w         : float
                     ):
        """Draws a single lane segment on the display, which consists of the left and right edge lines.
            ASSUMES that all segments are oriented with headings between 0 and 90 deg for simplicity.
        """

        # Find the scaled end-point pixel locations
        r0 = self.scale*(x0 - self.roadway_center_x) + self.display_center_r
        r1 = self.scale*(x1 - self.roadway_center_x) + self.display_center_r
        s0 = Graphics.WINDOW_SIZE_Y - (self.scale*(y0 - self.roadway_center_y) + self.display_center_s)
        s1 = Graphics.WINDOW_SIZE_Y - (self.scale*(y1 - self.roadway_center_y) + self.display_center_s)

        # Find the scaled width of the lane
        ws = 0.5 * w * self.scale

        angle = math.atan2(y1-y0, x1-x0) #radians in [-pi, pi]
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        # Find the screen coords of the left edge line
        left_r0 = r0 - ws*sin_a
        left_r1 = r1 - ws*sin_a
        left_s0 = s0 - ws*cos_a
        left_s1 = s1 - ws*cos_a

        # Find the screen coords of the right edge line
        right_r0 = r0 + ws*sin_a
        right_r1 = r1 + ws*sin_a
        right_s0 = s0 + ws*cos_a
        right_s1 = s1 + ws*cos_a

        # Draw the edge lines
        pygame.draw.line(self.windowSurface, Graphics.WHITE, (left_r0, left_s0), (left_r1, left_s1), 1)
        pygame.draw.line(self.windowSurface, Graphics.WHITE, (right_r0, right_s0), (right_r1, right_s1), 1)


    def _get_vehicle_coords(self,
                            obs:    list
                           ) -> tuple:
        """Returns the map coordinates of the ego vehicle based on lane ID and distance downtrack.

            CAUTION: these calcs are hard-coded to the specific roadway geometry in this code,
            it is not a general solution.
        """

        x = None
        y = None
        lane = obs[self.env.EGO_LANE_ID]
        if lane == 0:
            x = obs[self.env.EGO_X]
            y = self.env.roadway.lanes[0].segments[0][1]
        elif lane == 1:
            x = obs[self.env.EGO_X]
            y = self.env.roadway.lanes[1].segments[0][1]
        else:
            ddt = obs[self.env.EGO_X]
            if ddt < self.env.roadway.lanes[2].segments[0][4]: #vehicle is in seg 0
                seg0x0 = self.env.roadway.lanes[2].segments[0][0]
                seg0y0 = self.env.roadway.lanes[2].segments[0][1]
                seg0x1 = self.env.roadway.lanes[2].segments[0][2]
                seg0y1 = self.env.roadway.lanes[2].segments[0][3]

                factor = ddt / self.env.roadway.lanes[2].segments[0][4]
                x = seg0x0 + factor*(seg0x1 - seg0x0)
                y = seg0y0 + factor*(seg0y1 - seg0y0)

            else: #vehicle is in seg 1
                x = self.env.roadway.lanes[2].segments[1][0] + ddt - self.env.roadway.lanes[2].segments[0][4]
                y = self.env.roadway.lanes[2].segments[1][1]

        return x, y


######################################################################################################
######################################################################################################


if __name__ == "__main__":
   main(sys.argv)
