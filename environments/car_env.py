# Modules
import pygame
import time
import random
import numpy as np
def to_bool(s):
    return 1 if s else 0

# RGB Color
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,200,0)
# SPEED = 40
SPEED = 26.8 # m/s or 60 mph
ACCELERATION = 10 # m/s2
PEDESTRIAN_SPEED = 5 #1.42 #15 #1.42
TIME = 1000000
# Times in pygame are represented in milliseconds (1/1000 seconds)
# window with size of 500 x 400 pixels
CAR_POSITION = 500
# image
bg = pygame.image.load('./environments/bg_400_800.png')
bg_rect = bg.get_rect()

carimg = pygame.image.load('./environments/car.png')

humanimg = pygame.image.load('./environments/obstacle.png')
humanimg_rect = humanimg.get_rect()
wn_width = bg_rect.width
wn_height = bg_rect.height
ROAD_DIST = 40
# boundary
west_b = 10
east_b = 390 - humanimg_rect.width
      
class Hazard:
    def __init__(self,x,y,speedy):
        self.image = humanimg
        self.x = x
        self.y = y
        self.speedy = speedy
        self.dodged = 0
        self.stopped = False
        self.rect = self.image.get_rect()
        self.rect.x = self.x
        self.rect.y = self.y
        self.width = int(self.image.get_width()/1)
        self.height = int(self.image.get_height()/1)
        # self.speedx = 5
        self.speedx = PEDESTRIAN_SPEED
        
    def update(self):
        self.y = self.y + self.speedy
        
        if self.x < 100:
            self.x = self.x + int(2*np.random.binomial(1, 0.5, 1)-1)*(self.speedx+np.random.randint(-10,10)/10)
        else:
            self.x = self.x + int(2*np.random.binomial(1, 0.5, 1)-1)*(self.speedx+np.random.randint(-1,10)/10)
       
        # check boundary (block)
        if self.y > wn_height:
            self.y = 0 - self.height 
            self.rect.y = self.y
            
            self.x = random.randrange(west_b,(east_b-self.width))-int(self.width/2)
            # self.rect.x  = self.x
            self.dodged = self.dodged + 1
        if self.speedy <= 0:
            if not self.stopped:
                self.dodged = self.dodged + 1
            self.stopped = True
    
        self.rect.x = self.x 
        self.rect.y = self.y

    def draw(self,wn):
        pygame.draw.rect(wn, RED, [self.rect.x, 
                                   self.rect.y, 
                                   self.width, 
                                   self.height])
                  
        
class Player:
    def __init__(self):
        self.image = carimg
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.rect = self.image.get_rect()
        self.rect.x = int(wn_width/2 - self.rect.width/2)
        self.rect.y = int(CAR_POSITION)
        self.speedy = SPEED
        self.speedx = 0
        self.break_init = False
        
    def update(self, action):
        
        keystate = {}
        if action==0:
            keystate[pygame.K_SPACE] = True
        elif action==1:
            keystate[pygame.K_SPACE] = False
        
        if not self.break_init: 
            if keystate[pygame.K_SPACE]:
                # print(f'Stopping')
                self.break_init=True
                if self.speedy > 0:
                    self.speedy = max(0, self.speedy-ACCELERATION)
                elif self.speedy < 0:
                    self.speedy = min(0, self.speedy+ACCELERATION)
        elif self.break_init:
            if self.speedy > 0:
                self.speedy = max(0, self.speedy-ACCELERATION)
            elif self.speedy < 0:
                self.speedy = min(0, self.speedy+ACCELERATION)

        if self.rect.left < west_b:
            self.rect.left = west_b
        if self.rect.right > east_b:
            self.rect.right = east_b

    def draw(self,wn):
        pygame.draw.rect(wn, BLACK, [self.rect.x, 
                                   self.rect.y, 
                                   self.width, 
                                   self.height])

class CarEnv:
    def __init__(self, n_episodes=5,seed=None):
        if seed is None:
            np.random.seed()
            random.seed()
        else:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()
        self.n_episodes = n_episodes
        self.state = None
        self.reward = 0
        self.terminal = False
        self.action = 1
        self.crash = False
        
    def _init_game(self):
        block_x = random.randrange(0, bg_rect.width)
        block_y = 0
        self.action = 1
        self.terminal = False
        self.player = Player()
        self.block = Hazard(block_x,block_y,self.player.speedy)
        dist_o = self._dist_to_obstacle(self.block.y+self.block.height, self.player.rect.y)
        isp = self._impact_speed(dist_to_o=dist_o, init_speed=self.player.speedy, 
                                 acceleration=ACCELERATION)
        impact_sp = isp['impact_speed']
        impact_sp_sq = isp['impact_speed_sq']
        self.state = np.array([self.block.x+(self.block.width/2), 
                               impact_sp_sq, 
                               ])
        
        self.tick = 0
        self.action_hist = []
        left_bound_x, right_bound_x = self._stopping_region()
        
    def game_loop(self):
        self._init_game()
           
    def reset(self):
        pygame.quit()
        pygame.init() 
        # Clock
        self.clock = pygame.time.Clock()
        # self.wn = pygame.display.set_mode((wn_width,wn_height))
        # pygame.display.set_caption('Race car with road block')
        
        
    def _step(self, mode='expert'):
        if mode=='expert':
            left_bound_x, right_bound_x = self._stopping_region()
            if self.block.rect.topleft[0]<=right_bound_x and self.block.rect.topright[0]>=left_bound_x:
                if self.block.rect.bottomleft[1]>=430-self.block.rect.height and self.block.rect.topleft[1]<=self.player.rect.bottomleft[1]:
                    self.action = 0
                else:
                    self.action = 1
            else:
                self.action = 1
        elif mode=='test':
            self.action = 1
            
            left_bound_x, right_bound_x = self._stopping_region()
            if self.block.rect.topleft[0]<=right_bound_x and self.block.rect.topright[0]>=left_bound_x:
                if self.block.rect.bottomleft[1]>=430-self.block.rect.height and self.block.rect.topleft[1]<=self.player.rect.bottomleft[1]:
                    self.expert_action = 0
                else:
                    self.expert_action = 1
            elif self.expert_action==0:
                self.expert_action=0
            else:
                self.expert_action = 1
            # print(f'self.expert_action: {self.expert_action}')
        action = self.action
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
             pygame.quit()
        self.player.update(action)
        if mode=='expert':
            self.action_hist.append(action)
        else:
            self.action_hist.append(self.expert_action)
            # print(f'self.action_hist: {self.action_hist}')
        
        self.block.update()
        
        dist_o = self._dist_to_obstacle(self.block.y+self.block.height, self.player.rect.y)
        isp = self._impact_speed(dist_to_o=dist_o, init_speed=self.player.speedy, 
                                 acceleration=ACCELERATION)
        impact_sp = isp['impact_speed']
        impact_sp_sq = isp['impact_speed_sq']
        
        if self.tick>=1:
            if self.action_hist[self.tick-1]==0:
                self.reward = 0
            else:
                self.reward = self._reward(obstacle_loc_x=self.block.x+(self.block.width/2), 
                                            impact_speed=impact_sp, 
                                            impact_speed_sq=impact_sp_sq, 
                                            a=action)
        else:
           self.reward = self._reward(obstacle_loc_x=self.block.x+(self.block.width/2), 
                                            impact_speed=impact_sp, 
                                            impact_speed_sq=impact_sp_sq, 
                                            a=action)
        self.block.speedy = self.player.speedy
        self.state = np.array([self.block.x+(self.block.width/2), 
                               impact_sp_sq,
                               ])
        
        self.tick+=1
        # Car collision with block 
        if self.block.x+(self.block.width) >= wn_width/2-ROAD_DIST and self.block.x  <= wn_width/2+ROAD_DIST:
            if self.block.y+self.block.height >= self.player.rect.y and self.block.y <= self.player.rect.y:            
                    self.terminal = True   
                    # print(f'CRASH!!!')
                    self._crash()
                    self.crash=True
                    self.player.speedy = SPEED
                    self.tick=0
                    self.action_hist = []
        if self.player.speedy<=0:
            self.terminal = True
            self._success()
            self.tick=0
            self.action_hist = []
                
        self.clock.tick(TIME) 
        
    def step(self, mode='expert'):
        if int(self.action)==0:
            self.terminal = True
        else:
            self.terminal = False
        self._step(mode)
        
        return self.state, self.reward, self.terminal
    

    def _crash(self):
        self.terminal = True 
        self.crash = True
        self.tick=0
        self.n_episodes-=1
        self.game_loop()
    
    def _success(self):
        self.terminal = True 
        # print('Well done!')
        self.tick=0
        self.n_episodes-=1
        self.terminal = True
        self.game_loop()
        
    def _reward(self, obstacle_loc_x, impact_speed, 
                impact_speed_sq, lambda_speed=1, 
                lambda_loc=10, 
                distance_to_o=None,
                a=None):

        if a==1:
               total_rew = 0
        elif a==0:
           if impact_speed_sq<=0:
               total_rew = (1/np.sqrt(np.max([abs(impact_speed_sq),0.1])))**0.03*0.001
               total_rew -= (abs(200/2-obstacle_loc_x)**0.9/(40+self.block.width/2))*0.00005
               
               
           elif impact_speed_sq>0:
               
               total_rew = (10-1/np.sqrt(abs(impact_speed_sq)))**0.5*0.001
               total_rew -= (abs(200/2-obstacle_loc_x)**0.9/(40+self.block.width/2))*0.00005
               
           else:
               total_rew = 0

        return total_rew
    
    def _time_to_collision(self):
        pass
    
    
    def _dist_to_obstacle(self, obstacle_loc_y, position):
        # returns the absolute distance to the obstacle
        return (position-obstacle_loc_y)
    
    def _impact_speed(self, dist_to_o, init_speed, acceleration):
        speed_i_sq = init_speed**2 - 2*dist_to_o*acceleration
        # print(f'IMPACT_SPEED: {speed_i_sq}')
        return {'impact_speed': np.sqrt(max(0, speed_i_sq)), 
                'impact_speed_sq':speed_i_sq}

    def run_game(self):
        self.game_loop()
        pygame.quit()
        quit()
        
    def _stopping_region(self):
        x_p = self.block.rect.centerx
        y_p = self.block.rect.bottomleft[1]
        
        x_c = self.player.rect.centerx # x coordinate of the car (center of the car)
        y_c = self.player.rect.topleft[1]
        
        a = -ACCELERATION # deceleration
        v_c = SPEED # car speed
        v_p = PEDESTRIAN_SPEED # pedestrian speed
        d_p = y_p-y_c # distance from pedestrian to car
        b = v_c**2 - 2*d_p*a
        v_i = np.sqrt(max(0,b))
        tau_vti = (v_i-v_c)/a # time to collision
        d_pmax = v_p*tau_vti # max dist pedestrian makes in time-to-collision
        
        
        right_bound_x = (d_pmax+ROAD_DIST).copy()
        left_bound_x = (-ROAD_DIST-d_pmax).copy()
        return left_bound_x+wn_width/2-self.block.rect.width/2, right_bound_x+wn_width/2+self.block.rect.width/2
       
    def sim_test(self, episodes, max_path_length):
        res = self.sim_expert(episodes, max_path_length, mode='test')
        return res
        
    def sim_expert(self, episodes, max_path_length, mode='expert'):
        self.n_episodes = episodes
        
        STATE_MEM_SIM = []
        STATE2_MEM_SIM = []
        ACTION_MEM_SIM = []
        DONE_MEM_SIM = []
        IS_EXPERT = []
        INIT_STATES = []
        REWARD_MEM = []
        action_hist=[]
        PATH_IDS = []
        TIME_IDS = []
        for ep_num in range(episodes):
            self.expert_action=1
            self._init_game()
            init_state = self.state
            
            path_step=0
            self.crash=False
            self.terminal=False
            while max_path_length>path_step and not self.crash and not self.terminal:
                INIT_STATES.append(init_state)
                STATE_MEM_SIM.append(self.state)
                if mode=='expert':
                    ACTION_MEM_SIM.append(self.action)
                else:
                    ACTION_MEM_SIM.append(self.expert_action)
                PATH_IDS.append(ep_num)
                TIME_IDS.append(path_step)
                if int(self.action)==0:
                    self.terminal = True
                elif self.crash:
                    self.terminal=True
                else:
                    self.terminal = False
                DONE_MEM_SIM.append(to_bool(self.terminal))
                state_next, reward, terminal = self.step(mode)
                STATE2_MEM_SIM.append(state_next)
                REWARD_MEM.append(reward)
                path_step+=1
            ACTION_MEM_SIM[-1] = 0
                
        return {'state_mem': np.array(STATE_MEM_SIM), 
                'next_state_mem': np.array(STATE2_MEM_SIM), 
                'action_mem': np.array(ACTION_MEM_SIM), 
                'done_mem': np.array(DONE_MEM_SIM),
                'init_states_mem': np.array(INIT_STATES), 
                'reward_mem': np.array(REWARD_MEM),
                'path_ids':np.array(PATH_IDS),
                'time_ids':np.array(TIME_IDS)}
    

        
if __name__ == '__main__':
    
    env = CarEnv()
    res = env.sim_expert(10, 1000) #.run_game()
    