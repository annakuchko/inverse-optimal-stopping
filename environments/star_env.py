#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import matplotlib.pyplot as plt
from aleatory.processes import BESProcess, BrownianMotion
        
def check_if_contains(path, coords):
    return path.contains_point(coords)

from aleatory.processes import BESProcess, BrownianMotion

def generate_dense_star(num_points=5, outer_radius=1, inner_radius=0.5, density=20, rotation=0):
    """
    Generate a dense (x, y) scatter representation of a star shape.

    :param num_points: Number of star tips (default = 5)
    :param outer_radius: Radius of outer points (default = 1)
    :param inner_radius: Radius of inner points (default = 0.5)
    :param density: Number of interpolated points per star edge
    :param rotation: Rotation angle (in radians) to rotate the star
    :return: x, y coordinates of the star
    """
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False) + rotation  # Outer & inner points
    radii = np.array([outer_radius if i % 2 == 0 else inner_radius for i in range(len(angles))])

    # Convert polar to Cartesian coordinates
    x_raw = radii * np.cos(angles)
    y_raw = radii * np.sin(angles)

    # Interpolate more points along the star edges
    x_dense, y_dense = [], []
    for i in range(len(x_raw)):
        x_start, y_start = x_raw[i], y_raw[i]
        x_end, y_end = x_raw[(i + 1) % len(x_raw)], y_raw[(i + 1) % len(y_raw)]  # Wrap around
        
        # Create interpolated points between two key star points
        x_interp = np.linspace(x_start, x_end, density)
        y_interp = np.linspace(y_start, y_end, density)

        x_dense.extend(x_interp)
        y_dense.extend(y_interp)
    
    return x_dense, y_dense


def to_bool(s):
    return 1 if s == 'true' else 0

def shift_array(array, place=-1):
    new_arr = np.roll(array, place, axis=0)
    new_arr[place:,:,:] = 0
    new_arr[place:,0,np.isnan(new_arr[place-1,0,:])] = np.nan
    new_arr[place:,1,np.isnan(new_arr[place-1,1,:])] = np.nan
    new_arr[place:,2,np.isnan(new_arr[place-1,2,:])] = np.nan
    # new_arr[np.isnan(new_arr):] = np.nan
    return new_arr
import random
class STAR:
    def __init__(self, total_n=100):
        all_ids = [i for i in range(total_n)]
        self.training_ids = random.sample(all_ids, round(total_n*0.70))
        self.testing_ids = list(set(all_ids) - set(self.training_ids))
        # t_values = np.linspace(0, 50, 50)/50  # Generate 10 different t values
        t_values = np.arange(0,50)/50
        star_regions = {}
        
        for t, times in enumerate(t_values):
            # radius = 0.01 + t * 0.15  # Increase radius gradually
            rotation =  0.32 #t * 0.1  # Vary rotation based on t
            radius = 0.5 + t * 0.05  # Increase radius gradually
            # radius = 0.25 + (1/50)*t
            # rotation = t * 0.01
            
            x, y = generate_dense_star(num_points=5, outer_radius=radius,
                                       inner_radius=radius * 0.5, 
                                       density=1500, 
                                       rotation=rotation)
            star_path = path.Path(np.column_stack((x, y)))
            star_regions[t/50] = star_path
        self.star_regions = star_regions
    def sim_expert(self, episodes=None, max_path_length=None):
            # train_memory = self.memory #[self.memory[self.memory['path_ids']==self.train_test_paths[0]][i] for i in range(len(np.unique(self.memory['path_ids'])))]
            return self.load_transform_data(N=500, n=50, gamma=0.75, full_paths=False, ids=self.training_ids)
    
    def sim_test(self, episodes, max_path_length):
        return self.load_transform_data(N=1000, n=50, gamma=0.75, full_paths=False, ids=self.testing_ids)
    
    def load_transform_data(self,N=500, n=50, gamma=0.75, full_paths=False, ids=None):
        # def load_transform_data(self,N=500, n=50, gamma=0.75, full_paths=False, ids=None):
        N=len(ids)
        # bes = BrownianMotion(initial=0.0, T=1.0)
        bes = BrownianMotion(initial=0.0, T=1.0)
        paths = np.zeros((n,N, 2))
        X0 = np.zeros((n,N))
        X1 = np.zeros((n,N))
        T = np.zeros((n,N))
        STOPPING = np.zeros((n,N))

        for p in range(N):
            # print(f'path: {p}')
            ts=np.arange(n)/(n)
            x0,x1 = bes.sample(n), bes.sample(n)
            t=0
            st = np.full(n,False)
            while t<n:
                # if np.sqrt(x0[t]**2+x1[t]**2)<=t/50*2+0.05:
                if check_if_contains(self.star_regions[t/(n)], (x0[t], x1[t]) ):
                    pass
                else:

                    st[t] = True
                    x0[t+1:] = np.inf
                    x1[t+1:] = np.inf

                    break      
                t+=1      
            # if not np.isinf(x0[-1]):
            #     ts[-1] = True
            X0[:,p] = x0
            X1[:,p] = x1
            T[:,p] = ts 
            STOPPING[:,p] = st

            # plt.scatter(X0[(STOPPING==0)],
        #                     X1[(STOPPING==0)],
        #                     s=5, color='green')
        
        # X0[-1,:] = np.inf
        # X1[-1,:] = np.inf

        # plt.scatter(X0[(STOPPING==1)],
        #                     X1[(STOPPING==1)],
        #                     s=5, color='red')
        
        # plt.xlim(-1.5,1.5)
        # plt.ylim(-1.5,1.5)
        # plt.show()   

        
        data = np.stack([X0,X1],1)
        data = np.hstack([data.copy(), np.repeat(np.array([i/data.shape[0] for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data.copy(), np.repeat(np.array([i for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data, np.array([np.repeat(i, data.shape[0]) for i in range(data.shape[2])]).T.reshape(data.shape[0],1,-1)])

        inf_ids = np.isinf(data)[:,0,:,][:,None,:]
        data[:,2,:][:,None,:][inf_ids] = np.inf
        data[np.isinf(data)] = np.nan
        shifted_data = shift_array(data)

        # 0 - x1, 1-x2, 2-time, 3-time_ids, 4-path_ids
        nan_mask = np.isnan(data[:,:,:])
        nan_mask_next = np.isnan(shifted_data[:,0,:])
        nan_mask_last = nan_mask_next.copy()
        # nan_mask_last[-1,:] = True
        DONE_MEM_SIM = nan_mask_last.astype(int).T[~nan_mask[:,0,:].T].flatten()
        action_mem = (nan_mask_next.astype(int)-1)**2
        action_mem[-1,:] = 0
        # DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
        # if full_paths:
        #     st = action_mem.copy()
        #     first_st = np.argmax(st==0,axis=1)
        #     idxrow,idxcol=np.indices(st.shape)
        #     first_st2=first_st[:,None]
        #     mask = idxcol > first_st2
        #     st[mask]=0
        #     ACTION_MEM_SIM = st.T.flatten() 
        if True:
            st = action_mem.copy()
            # st = (nan_mask_last.astype(int)-1)**2
            first_st = np.argmax(st==0,axis=0)
            # first_st = []
            # for i in range(st.shape[0]):
            #     s = np.where(st[i,:]==0)[0]
            #     if s.size>0:
            #         first_st.append(s)
            #     else:
            #         first_st.append(10000000)
            # first_st =  np.where(st==0, axis=0)
            # first_st[first_st==0] = 10000000
            print(f'first_st: {first_st}')
            idxrow,idxcol=np.indices(st.shape)
            first_st2=first_st[None,:]
            mask = idxrow > first_st2
            st[mask]=0
            ACTION_MEM_SIM = st.T.flatten()[~nan_mask[:,0,:].T.flatten()].flatten()
            # ACTION_MEM_SIM = (ACTION_MEM_SIM
        # X1,X2,Time, DONE
        STATE_MEM_SIM = np.array([data[:,0,:].T[~nan_mask[:,0,:].T].flatten().T, data[:,1,:].T[~nan_mask[:,1,:].T].flatten().T, data[:,2,:].T[~nan_mask[:,2,:].T].flatten().T]).T
        STATE2_MEM_SIM = np.array([shifted_data[:,0,:].T[~nan_mask[:,0,:].T].flatten(), shifted_data[:,1,:].T[~nan_mask[:,1,:].T].flatten(), shifted_data[:,2,:].T[~nan_mask[:,2,:].T].flatten()]).T
        STATE2_MEM_SIM = np.nan_to_num(STATE2_MEM_SIM.copy())
        TIME_IDS = np.array(data[:,3,:].T[~nan_mask[:,0,:].T].flatten())
        PATH_IDS = np.array(data[:,4,:].T[~nan_mask[:,0,:].T].flatten())

        state_mem = np.array(STATE_MEM_SIM)
        INIT_STATES = np.zeros_like(TIME_IDS)
        
            
        return {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}

def generate_dense_circles(num_circles=10, base_radius=0.5, step=0.5, density=100):
    """
    Generate (x, y) coordinates for concentric circles.

    :param num_circles: Number of concentric circles
    :param base_radius: Radius of the smallest circle
    :param step: Incremental increase in radius for each circle
    :param density: Number of points per circle
    :return: x, y coordinates of the circles
    """
    x_all, y_all = [], []

    for i in range(num_circles):
        radius = base_radius + i * step  # Increasing radius
        angles = np.linspace(0, 2 * np.pi, density)  # Points evenly spaced around the circle

        # Convert polar to Cartesian coordinates
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        x_all.extend(x)
        y_all.extend(y)

    return x_all, y_all

class RADIAL:
    def __init__(self, total_n=100):
        all_ids = [i for i in range(total_n)]
        self.training_ids = random.sample(all_ids, round(total_n*0.70))
        self.testing_ids = list(set(all_ids) - set(self.training_ids))
        # t_values = np.linspace(0, 50, 50)/50  # Generate 10 different t values
        t_values = np.arange(0,50)/50
        star_regions = {}
        for t, times in enumerate(t_values):
            radius = 0.5 + t * 0.05  # Increase radius gradually
            # radius = 0.25 + (1/50)*t
            # rotation = t * 0.01  # Vary rotation based on t
            
            x, y = generate_dense_circles(num_circles=1, 
                                          base_radius=radius, 
                                          density=1500)
            star_path = path.Path(np.column_stack((x, y)))
            star_regions[t/50] = star_path
        self.star_regions = star_regions
    def sim_expert(self, episodes=None, max_path_length=None):
            # train_memory = self.memory #[self.memory[self.memory['path_ids']==self.train_test_paths[0]][i] for i in range(len(np.unique(self.memory['path_ids'])))]
            return self.load_transform_data(N=500, n=50, gamma=0.75, full_paths=False, ids=self.training_ids)
    
    def sim_test(self, episodes, max_path_length):
        return self.load_transform_data(N=1000, n=50, gamma=0.75, full_paths=False, ids=self.testing_ids)
    
    def load_transform_data(self,N=500, n=50, gamma=0.75, full_paths=False, ids=None):
        from aleatory.processes import BESProcess, BrownianMotion
        N=len(ids)
        # bes = BrownianMotion(initial=0.0, T=1.0)
        bes = BrownianMotion(initial=0.0, T=1.0)
        paths = np.zeros((n,N, 2))
        X0 = np.zeros((n,N))
        X1 = np.zeros((n,N))
        T = np.zeros((n,N))
        STOPPING = np.zeros((n,N))

        for p in range(N):
            # print(f'path: {p}')
            ts=np.arange(n)/(n)
            x0,x1 = bes.sample(n), bes.sample(n)
            t=0
            st = np.full(n,False)
            while t<n:
                if np.sqrt(x0[t]**2+x1[t]**2) < 0.5 + t*0.01:
                # if check_if_contains(self.star_regions[t/(n)], (x0[t], x1[t]) ):
                    pass
                else:
                    st[t] = True
                    x0[t+1:] = np.inf
                    x1[t+1:] = np.inf
                    break      
                t+=1      
            # if not np.isinf(x0[-1]):
            #     ts[-1] = True
            X0[:,p] = x0
            X1[:,p] = x1
            T[:,p] = ts 
            STOPPING[:,p] = st

            # plt.scatter(X0[(STOPPING==0)],
        #                     X1[(STOPPING==0)],
        #                     s=5, color='green')
        
        # X0[-1,:] = np.inf
        # X1[-1,:] = np.inf

        # plt.scatter(X0[(STOPPING==1)],
        #                     X1[(STOPPING==1)],
        #                     s=5, color='red')
        
        # plt.xlim(-1.5,1.5)
        # plt.ylim(-1.5,1.5)
        # plt.show()   

        
        data = np.stack([X0,X1],1)
        data = np.hstack([data.copy(), np.repeat(np.array([i/data.shape[0] for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data.copy(), np.repeat(np.array([i for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data, np.array([np.repeat(i, data.shape[0]) for i in range(data.shape[2])]).T.reshape(data.shape[0],1,-1)])

        inf_ids = np.isinf(data)[:,0,:,][:,None,:]
        data[:,2,:][:,None,:][inf_ids] = np.inf
        data[np.isinf(data)] = np.nan
        shifted_data = shift_array(data)

        # 0 - x1, 1-x2, 2-time, 3-time_ids, 4-path_ids
        nan_mask = np.isnan(data[:,:,:])
        nan_mask_next = np.isnan(shifted_data[:,0,:])
        nan_mask_last = nan_mask_next.copy()
        nan_mask_last[-1,:] = True
        DONE_MEM_SIM = nan_mask_last.astype(int).T[~nan_mask[:,0,:].T].flatten()
        action_mem = (nan_mask_next.astype(int)-1)**2
        action_mem[-1,:] = 0
        # DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
        # if full_paths:
        #     st = action_mem.copy()
        #     first_st = np.argmax(st==0,axis=1)
        #     idxrow,idxcol=np.indices(st.shape)
        #     first_st2=first_st[:,None]
        #     mask = idxcol > first_st2
        #     st[mask]=0
        #     ACTION_MEM_SIM = st.T.flatten() 
        if True:
            st = action_mem.copy()
            # st = (nan_mask_last.astype(int)-1)**2
            first_st = np.argmax(st==0,axis=0)
            
            # print(f'BEFORE first_st==0: {sum(first_st==0)}')
            first_st[first_st==0] = 100000
            # print(f'first_st==0: {sum(first_st==0)}')
            idxrow,idxcol=np.indices(st.shape)
            first_st2=first_st[None,:]
            mask = idxrow > first_st2
            st[mask]=0
            ACTION_MEM_SIM = st.T.flatten()[~nan_mask[:,0,:].T.flatten()].flatten()
            # ACTION_MEM_SIM = (ACTION_MEM_SIM
        # X1,X2,Time, DONE
        STATE_MEM_SIM = np.array([data[:,0,:].T[~nan_mask[:,0,:].T].flatten().T, data[:,1,:].T[~nan_mask[:,1,:].T].flatten().T, data[:,2,:].T[~nan_mask[:,2,:].T].flatten().T]).T
        STATE2_MEM_SIM = np.array([shifted_data[:,0,:].T[~nan_mask[:,0,:].T].flatten(), shifted_data[:,1,:].T[~nan_mask[:,1,:].T].flatten(), shifted_data[:,2,:].T[~nan_mask[:,2,:].T].flatten()]).T
        STATE2_MEM_SIM = np.nan_to_num(STATE2_MEM_SIM.copy())
        TIME_IDS = np.array(data[:,3,:].T[~nan_mask[:,0,:].T].flatten())
        PATH_IDS = np.array(data[:,4,:].T[~nan_mask[:,0,:].T].flatten())

        state_mem = np.array(STATE_MEM_SIM)
        INIT_STATES = np.zeros_like(TIME_IDS)
        
            
        return {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}


# if __name__=='__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
#     # plt.figure(figsize=(10, 10))
#     fig, ax = plt.subplots(figsize=(15,15))
    
#     # t_values = np.linspace(0, 50, 50)/50  # Generate 10 different t values
#     t_values = np.arange(0,50)/50
#     star_regions = {}
#     import matplotlib
#     list_of_times = list(t_values)
#     cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
#     gradient = np.arange(0, 1, 1/len(list_of_times))
#     list_of_colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(list_of_times)]
    
#     for t, times in enumerate(t_values):
#         # if t%2:
#         # radius = 0.25 + (1/50)*t
#         # # radius = 0.01 + t * 0.25  # Increase radius gradually
#         rotation = 0.32 #t * 0.01   # Vary rotation based on t
#         # rotation = t * 0.01  # Vary rotation based on t
#         radius = 0.5 + t * 0.05  # Increase radius gradually
#         x, y = generate_dense_circles(num_circles=1, 
#                                       base_radius=radius, 
#                                       density=1500)
#         # x, y = generate_dense_star(num_points=5, outer_radius=radius, 
#         #                             inner_radius=radius * 0.5, density=1500, 
#         #                             rotation=rotation)
#         star_path = path.Path(np.column_stack((x, y)))

#         star_regions[t/50] = star_path
#         # if t%1==0:
#         ax.scatter(x, y, s=10, alpha=0.7, 
#                         color=list_of_colours[t])  # Adjust size and transparency
#     # plt.axhline(0, color='gray', linewidth=0.5)
#     # plt.axvline(0, color='gray', linewidth=0.5)
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.yaxis.set_major_locator(MultipleLocator(1))
#     ax.grid(True)
#     ax.set_xlim((-3,3))
#     ax.set_ylim((-3,3))
#     ax.set_xlabel('x0', size=60)
#     ax.set_ylabel('x1', size=60)
#     ax.tick_params(axis='x', rotation=90)
#     ax.tick_params(axis='both', which='major', labelsize=55)
#     ax.set_aspect('equal')
#     fig.tight_layout()
#     # plt.title("Multiple Rotating Stars Scatter Plot")
#     # plt.axis("equal")  # Keep aspect ratio square
#     # for t,times in enumerate(t_values):
#     #     if t%1==0 and t>0:        
#     #         star_regions[t_values[t-1]] = star_regions[times]
#     #         star_regions[t_values[t-2]] = star_regions[times]
#     # Show plot
#     plt.show()
#     import numpy as np
#     from random import sample
#     import matplotlib.pyplot as plt
#     import matplotlib.colors as pltc
#     all_colors = [k for k,v in pltc.cnames.items()]
#     # colors = 
#     n=50
#     N=1000
#     cols = sample(all_colors, n)
#     # bes = BESProcess(dim=2, initial=0.0, T=1.0)
#     bes = BrownianMotion(initial=0.0, T=1.0)
#     # paths = bes.simulate(n=50, N=5)
#     paths = np.zeros((n,N, 2))
#     X0 = np.zeros((n,N))
#     X1 = np.zeros((n,N))
#     T = np.zeros((n,N))
#     STOPPING = np.zeros((n,N))
#     for p in range(N):
#         # print(f'path: {p}')
#         ts=np.arange(n)/(n)
#         x0,x1 = bes.sample(n), bes.sample(n)
#         t=0
#         st = np.full(n,False)
#         while t<n:
#             # if t==0:
#             #     rad = 0.2
#             # else:
#             # rad=0.1+t*0.5
#             rad = 0.25 + (1/50)*t
#             # if np.sqrt(x0[t]**2+x1[t]**2)<rad:
#             if check_if_contains(star_regions[t/(n)], (x0[t], x1[t]) ):
#                 pass
#             else:
#                 st[t] = True
#                 x0[t+1:] = np.inf
#                 x1[t+1:] = np.inf
#                 break      
#             t+=1      
#         X0[:,p] = x0
#         X1[:,p] = x1
#         T[:,p] = ts
#         if not np.isinf(x0[-1]):
#             st[-1] = True


#         STOPPING[:,p] = st

#     for t in range(n):
#         if t%1==0:
#         # if True:
#             plt.scatter(X0[(STOPPING==1)&(T==(t*1/(n)))],
#                         X1[(STOPPING==1)&(T==(t*1/(n)))],
#                         s=5, color=cols[t])   
             
#     plt.xlim(-1.5,1.5)
#     plt.ylim(-1.5,1.5)    
#     plt.show()
    
#     plt.scatter(X0[(STOPPING==0)],
#                         X1[(STOPPING==0)],
#                         s=5, color='green')

#     plt.scatter(X0[(STOPPING==1)],
#                         X1[(STOPPING==1)],
#                         s=5, color='red')
#     plt.xlim(-1.5,1.5)
#     plt.ylim(-1.5,1.5)  
#     plt.show()   
#     # bm_expert_data = RADIAL(total_n=500).sim_expert()
#     # plt.scatter(bm_expert_data['state_mem'][bm_expert_data['action_mem']==1][:,0],bm_expert_data['state_mem'][bm_expert_data['action_mem']==1][:,1],s=5, color='green')
#     # plt.scatter(bm_expert_data['state_mem'][bm_expert_data['action_mem']==0][:,0],bm_expert_data['state_mem'][bm_expert_data['action_mem']==0][:,1], s=5, color='red')

#     # plt.xlim(-1.5,1.5)
#     # plt.ylim(-1.5,1.5)
#     # plt.show()
    
# %%
