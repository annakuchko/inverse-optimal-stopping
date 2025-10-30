import numpy as np
import random
# from inverse_opt_stopping.data_preprocessing import shift_array
import pandas as pd
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
import numpy as np 
import matplotlib.pyplot as plt

def bessel_2d(n,N,gamma, full_paths):
    from stochastic.processes.continuous import BesselProcess
    X = np.zeros((n,2,N))
    A = np.zeros((n,1,N))
    for path in range(N):
        bessel = BesselProcess(dim=1, t=1)
        t = np.arange(n)/n
        x1 = bessel.sample_at(t)
        x2 = bessel.sample_at(t)
        dt=1/n
        
        for i in reversed(range(n)):
            if i==(n-1):
                A[i,:,path] = 0
            elif i==0:
                A[i,:,path]=1
            # elif np.sqrt(x1[i]**2+x2[i]**2)>=20*t[i]:
            elif np.sqrt(x1[i]**2+x2[i]**2)*t[i]>=t[i]:
                print(f'comparing {np.sqrt(x1[i]**2+x2[i]**2)} >= {10*t[i]}')
                print(f'time: {t[i]}')
                print()
                if not full_paths: 
                    x1[i+1:] = np.inf
                    x2[i+1:] = np.inf
                    A[i:,:,path] = 0
                else:
                    A[i,:,path] = 1
            else:
                A[i,:,path] = 1
        X[:,:,path] = np.array([x1,x2]).T
    if not full_paths:
        return X
    else:
        return X, A
    
    
def sim_bm(n,N,gamma,full_paths=False, continuation_g=True, cumulative_g=True):
    
    sigma=1
    X = np.zeros((n,2,N))
    A = np.ones((n,1,N))
    x10=0
    x20=0
    def G(t,x1,x2):
        x1_signs = np.array([np.sign(i) for i in x1])
        x2_signs = np.array([np.sign(i) for i in x2])
        t1 = -(x1_signs*x2_signs)
        return (x1**2+x2**2)
    
    def g(t,x1,x2):
        x1_signs = np.array([np.sign(i) for i in x1])
        x2_signs = np.array([np.sign(i) for i in x2])
        t1 = (x1_signs*x2_signs)
        t2 = (x1**2+x2**2)
        return 1.5*t2*t1    
    # def G(t,x1,x2):
    #     return (x1**2+x2**2)
    
    # def g(t,x1,x2):
    #     s_dist = np.sqrt(x1**2+x2**2)
    #     g_fun = s_dist.copy()
    #     g_fun[s_dist<1] = 1/10
    #     g_fun[s_dist>=1] = -8
    #     return g_fun
    
    for path in range(N):
        dt=1/n
        t = np.arange(n)/n
        x1 = np.hstack([np.array(0), np.cumsum(sigma *np.random.randn(n-1))])*dt
        x2 = np.hstack([np.array(0), np.cumsum(sigma *np.random.randn(n-1))])*dt
        st_gain = G(t,x1,x2)
        cont_gain = g(t,x1,x2)
        cumsum_g = np.cumsum(cont_gain)*(gamma**(np.arange(0,n,1)))
        # if path<=10:
        #     plt.plot(st_gain[:],c='red')
        #     plt.plot(cont_gain[:],c='green')
        #     plt.plot(cumsum_g[:], c='blue')
        #     plt.show()
        value = np.zeros_like(t)
        if continuation_g:
            if cumulative_g:
                for i in reversed(range(n)):
                    if i==(n-1):
                        value[i] = st_gain[i]
                        A[i,:,path] = 0
                    elif st_gain[i]>=cumsum_g[i]+gamma*value[i+1]:
                        value[i] = st_gain[i]
                        if not full_paths: 
                            x1[i+1:] = np.inf
                            x2[i+1:] = np.inf
                        A[i:,:,path] = 0
                    else:
                        value[i] = cumsum_g[i]+gamma*value[i+1]
                        A[i,:,path] = 1
            else:
                for i in reversed(range(n)):
                    if i==(n-1):
                        value[i] = st_gain[i]
                        A[i,:,path] = 0
                    elif st_gain[i]>=cont_gain[i]+gamma*value[i+1]:
                        value[i] = st_gain[i]
                        if not full_paths: 
                            x1[i+1:] = np.inf
                            x2[i+1:] = np.inf
                        A[i:,:,path] = 0
                    else:
                        value[i] = cont_gain[i]+gamma*value[i+1]
                        A[i,:,path] = 1

        else:
            for i in reversed(range(n)):
                if i==(n-1):
                    value[i] = st_gain[i]
                    A[i,:,path] = 0
                elif st_gain[i]>=gamma*value[i+1]:
                    value[i] = st_gain[i]
                    if not full_paths: 
                        x1[i+1:] = np.inf
                        x2[i+1:] = np.inf
                    A[i:,:,path] = 0
                else:
                    value[i] = gamma*value[i+1]
                    A[i,:,path] = 1
        X[:,:,path] = np.array([x1,x2]).T
    if not full_paths:
        return X
    else:
        return X, A
class SBM:
    def __init__(self, dat_type='gG', total_n=100):
        if dat_type=='gG':
            path_to_data='data/stoppedPaths/stoppedPaths_gG.pkl'
        elif dat_type=='G':
            path_to_data='data/stoppedPaths/stoppedPaths_G.pkl'
        self.path_to_data = path_to_data
        self.train_test_paths = 0 #random paths ids out of np.unique(PATH_IDS)
        all_ids = [i for i in range(total_n)]
        self.training_ids = random.sample(all_ids, round(total_n*0.70))
        self.testing_ids = list(set(all_ids) - set(self.training_ids))
        # self.training_ids = random.sample(all_ids, 500)
        # self.testing_ids = list(set(all_ids) - set(self.training_ids))
 
    def sim_expert(self, episodes=None, max_path_length=None):
        # train_memory = self.memory #[self.memory[self.memory['path_ids']==self.train_test_paths[0]][i] for i in range(len(np.unique(self.memory['path_ids'])))]
        return self.load_transform_data(N=500, n=0, gamma=0.75, full_paths=False, ids=self.training_ids)
    
    def sim_test(self, episodes, max_path_length):
        
        return self.load_transform_data(N=1000, n=500, gamma=0.75, full_paths=False, ids=self.testing_ids)
    
    def load_transform_data(self,N=500, n=51, gamma=0.75, full_paths=False, ids=None):
        data = np.load(self.path_to_data, allow_pickle=True).copy()[:,:,ids]
        # data = np.load(self.path_to_data, allow_pickle=True).copy().reshape(50,2,-1)[:,:,n:N]
        
        # if full_paths:
        #     data, action_mem = sim_bm(n=n,N=N,gamma=gamma,full_paths=full_paths)
        # else:
        #     data = sim_bm(n=n,N=N,gamma=gamma,full_paths=full_paths)
        
        data = np.hstack([data.copy(), np.repeat(np.array([i/data.shape[0] for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data.copy(), np.repeat(np.array([i for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data, np.array([np.repeat(i, data.shape[0]) for i in range(data.shape[2])]).T.reshape(data.shape[0],1,-1)])

        inf_ids = np.isinf(data)[:,0,:,][:,None,:]
        data[:,2,:][:,None,:][inf_ids] = np.inf
        data[np.isinf(data)] = np.nan
        shifted_data = shift_array(data)

        # 0 - x1, 1-x2, 2-time, 3-time_ids, 4-path_ids
        nan_mask = np.isnan(data[:,:,:]).T
        nan_mask_next = np.isnan(shifted_data[:,0,:]).T
        nan_mask_last = nan_mask_next.copy()
        nan_mask_last[:,-1] = True
        DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
        action_mem = (nan_mask_next.astype(int)-1)**2
        # DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
        if full_paths:
            st = action_mem
            first_st = np.argmax(st==0,axis=1)
            idxrow,idxcol=np.indices(st.shape)
            first_st2=first_st[:,None]
            mask = idxcol > first_st2
            st[mask]=0
            ACTION_MEM_SIM = st.flatten() 
        else:
            st = action_mem
            # st = (nan_mask_last.astype(int)-1)**2
            first_st = np.argmax(st==0,axis=1)
            idxrow,idxcol=np.indices(st.shape)
            first_st2=first_st[:,None]
            mask = idxcol > first_st2
            st[mask]=0
            ACTION_MEM_SIM = st[~nan_mask[:,0,:]].flatten()
            # ACTION_MEM_SIM = (ACTION_MEM_SIM
        # X1,X2,Time, DONE
        STATE_MEM_SIM = np.array([data[:,0,:].T[~nan_mask[:,0,:]].flatten(), data[:,1,:].T[~nan_mask[:,1,:]].flatten(), data[:,2,:].T[~nan_mask[:,2,:]].flatten()]).T
        STATE2_MEM_SIM = np.array([shifted_data[:,0,:].T[~nan_mask[:,0,:]].flatten(), shifted_data[:,1,:].T[~nan_mask[:,1,:]].flatten(), shifted_data[:,2,:].T[~nan_mask[:,2,:]].flatten()]).T
        STATE2_MEM_SIM = np.nan_to_num(STATE2_MEM_SIM.copy())


        TIME_IDS = np.array(data[:,3,:].T[~nan_mask[:,0,:]].flatten())
        PATH_IDS = np.array(data[:,4,:].T[~nan_mask[:,0,:]].flatten())

        state_mem = np.array(STATE_MEM_SIM)
        INIT_STATES = np.zeros_like(TIME_IDS)
        
            
        return {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}

def shift_array_old(array, place=-1):
    new_arr = np.roll(array, place, axis=1)
    # new_arr = new_arr[:,:place]
    new_arr[:,place] = new_arr[:, place-1]
    return new_arr

class SBM_old:
    def __init__(self, path_to_data='data/symmetricalBM/'):
        # (path, time)
        with open(path_to_data+'Time.txt', 'rb') as f:
            TIME = np.loadtxt(f, dtype='str', )
            TIME = np.char.replace(TIME,'NA', 'NAN').astype(np.float64)
            TIME_next = shift_array_old(TIME,-1)
            # TIME = TIME[:,:-1]
        
        with open(path_to_data+'X1.txt', 'rb') as f:
            X1 = np.loadtxt(f, dtype='str',)
            X1 = np.char.replace(X1,'NA', 'NAN').astype(np.float64)
            X1_next = shift_array_old(X1,-1)
            # X1 = X1[:,:-1]
            
        with open(path_to_data+'X2.txt', 'rb') as f:
            X2 = np.loadtxt(f, dtype='str',)
            X2 = np.char.replace(X2,'NA', 'NAN').astype(np.float64)
            X2_next = shift_array_old(X2,-1)
            # X2 = X2[:,:-1]
        
        
        nan_mask = np.isnan(TIME)#.flatten()
        nan_mask_next=np.isnan(TIME_next)#.flatten()
        nan_mask_last = nan_mask_next.copy()
        nan_mask_last[:,-1] = True
        
        DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask].flatten()
        
        ACTION_MEM_SIM = nan_mask_last.astype(int)[~nan_mask].flatten()
        ACTION_MEM_SIM = (ACTION_MEM_SIM-1)**2
        
        STATE_MEM_SIM = np.array([X1[~nan_mask].flatten(), X2[~nan_mask].flatten(), TIME[~nan_mask].flatten()]).T
        
        STATE2_MEM_SIM = np.array([X1_next[~nan_mask].flatten(), X2_next[~nan_mask].flatten(),TIME_next[~nan_mask].flatten()]).T
        STATE2_MEM_SIM = pd.DataFrame(STATE2_MEM_SIM).fillna(method='ffill').values
        
        # nan_mask = np.isnan(TIME)
        # nan_mask_next=np.isnan(TIME_next)
        # nan_mask_last = nan_mask_next.copy()
        # nan_mask_last[:,-1] = True
        
        # DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask].flatten()
        
        # ACTION_MEM_SIM = nan_mask_last.astype(int)[~nan_mask].flatten()
        # ACTION_MEM_SIM = (ACTION_MEM_SIM-1)**2
        
        # STATE_MEM_SIM = np.array([TIME[~nan_mask].flatten(), X1[~nan_mask].flatten(), X2[~nan_mask].flatten()]).T
        
        # STATE2_MEM_SIM = np.array([TIME_next[~nan_mask].flatten(), X1_next[~nan_mask].flatten(), X2_next[~nan_mask].flatten()]).T
        # STATE2_MEM_SIM = pd.DataFrame(STATE2_MEM_SIM).fillna(method='ffill').values
        
        
        PATH_IDS = np.array([np.repeat(i, TIME[i,:].shape[0]) for i in range(TIME.shape[0])])[~nan_mask].flatten()
        TIME_IDS = np.array([np.repeat(i, TIME.shape[0]) for i in range(TIME.shape[1])]).T[~nan_mask].flatten()
        INIT_STATES = STATE_MEM_SIM.copy()
        
        self.memory = {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}
        self.train_test_paths = 0 #random paths ids out of np.unique(PATH_IDS)
    
    def sim_expert(self, episodes=None, max_path_length=None):
        train_memory = self.memory #[self.memory[self.memory['path_ids']==self.train_test_paths[0]][i] for i in range(len(np.unique(self.memory['path_ids'])))]
        return train_memory
    
    def sim_test(self, episodes, max_path_length):
        return self.memory

# import numpy as np
# import pandas as pd

# #Data pre-processing helper-functions
# def to_bool(s):
#     return 1 if s == 'true' else 0

# def shift_array(array, place=-1):
#     new_arr = np.roll(array, place, axis=0)
#     new_arr[place:,:,:] = 0
#     new_arr[place:,0,np.isnan(new_arr[place-1,0,:])] = np.nan
#     new_arr[place:,1,np.isnan(new_arr[place-1,1,:])] = np.nan
#     new_arr[place:,2,np.isnan(new_arr[place-1,2,:])] = np.nan
#     # new_arr[np.isnan(new_arr):] = np.nan
#     return new_arr

# # Oversampling helper-functions
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE, ADASYN

# from imblearn.combine import SMOTEENN
# from imblearn.under_sampling import EditedNearestNeighbours

# def oversampling(input_states, actions, strategy='SMOTE',sampling_strategy=0.25, k_neighbors=15):


#     X_resampled, y_resampled = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors).fit_resample(
#         input_states, actions.astype(int))
#     return X_resampled, y_resampled
    
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# PLOT_ORIG_INPUTS = False
# PLOT_AUG_INPUTS = False
# orig = False

# data = np.load('C:/Users/Asus/Downloads/stoppedPaths/stoppedPaths_G.pkl', allow_pickle=True) [:,:,:5000]
# data = np.hstack([data.copy(), np.repeat(np.array([i/50 for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(50,1,-1)])
# data = np.hstack([data.copy(), np.repeat(np.array([i for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(50,1,-1)])
# data = np.hstack([data, np.array([np.repeat(i, data.shape[0]) for i in range(data.shape[2])]).T.reshape(50,1,-1)])

# inf_ids = np.isinf(data)[:,0,:,][:,None,:]
# data[:,2,:][:,None,:][inf_ids] = np.inf
# data[np.isinf(data)] = np.nan
# shifted_data = shift_array(data)


# nan_mask = np.isnan(data[:,:,:]).T
# nan_mask_next = np.isnan(shifted_data[:,0,:]).T
# nan_mask_last = nan_mask_next.copy()
# nan_mask_last[:,-1] = True
# DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
# ACTION_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
# ACTION_MEM_SIM = (ACTION_MEM_SIM-1)**2
# # X1,X2,Time, DONE
# STATE_MEM_SIM = np.array([data[:,0,:].T[~nan_mask[:,0,:]].flatten(), data[:,1,:].T[~nan_mask[:,1,:]].flatten(), data[:,2,:].T[~nan_mask[:,2,:]].flatten()]).T
# STATE2_MEM_SIM = np.array([shifted_data[:,0,:].T[~nan_mask[:,0,:]].flatten(), shifted_data[:,1,:].T[~nan_mask[:,1,:]].flatten(), shifted_data[:,2,:].T[~nan_mask[:,2,:]].flatten()]).T
# STATE2_MEM_SIM = np.nan_to_num(STATE2_MEM_SIM.copy())


# TIME_IDS = np.array(data[:,3,:].T[~nan_mask[:,0,:]].flatten())
# PATH_IDS = np.array(data[:,4,:].T[~nan_mask[:,0,:]].flatten())

# state_mem = np.array(STATE_MEM_SIM)
# st_mean = np.nanmean(state_mem,axis=0)
# st_mean[2] = 0
# st_std = np.nanstd(state_mem, axis=0)
# st_std[2] = 1
# print(f'st_std: {st_std}')
# print(f'st_mean: {st_mean}')
# st_min = np.nanmin(state_mem, axis=0)
# st_max = np.nanmax(state_mem, axis=0)
# print(f'st_min: {st_min}')
# print(f'st_max: {st_max}')
#     # Something is wrong when i transform input data into memory
# # memory = [(state_mem.copy()-st_mean)/st_std, 
# #               (np.array(STATE2_MEM_SIM).copy()-st_mean)/st_std,
# #               np.array(ACTION_MEM_SIM).copy(), 
# #               np.array(DONE_MEM_SIM).copy()
# #               ]

# memory = [(state_mem.copy()), 
#               (np.array(STATE2_MEM_SIM).copy()),
#               np.array(ACTION_MEM_SIM).copy(), 
#               np.array(DONE_MEM_SIM).copy(),
#                np.array(TIME_IDS).copy(),
#                np.array(PATH_IDS).copy()
#               ]

# # X_resampled, y_resampled = oversampling(input_states=np.hstack([memory[0],
# #                                                                 np.nan_to_num(memory[1]),
# #                                                                 memory[3].reshape(-1,1)]),
# #                                         actions = memory[2].astype(int),
# #                                         sampling_strategy=0.5, 
# #                                         k_neighbors=12)

# # sampled_memory = [X_resampled[:,:3].copy(), 
# #                     np.array(X_resampled[:,3:6]), 
# #                     np.array(y_resampled), 
# #                     np.array(X_resampled[:,6]).reshape(-1),
# #                     ]
# sampled_memory = memory.copy()


# def plot_inputs(inp_type='orig'):
#     if inp_type=='orig':
#         inp_data = data.copy()
#         data_ext = np.vstack([data.copy(), np.full_like(data[0,:,:], np.nan).reshape(1,5,-1)])
#         st_times = np.array([np.where(np.isnan(data_ext[:,0,i]))[0][0] for i in range(data_ext.shape[2])]).copy()-1
#         st_data = np.zeros_like(data[0,:,:].reshape(5,-1))
#         for i in range(data.shape[2]):
#             st_data[:,i] = data[st_times[i], :, i].copy()

#         plt.figure(figsize=(15,10))
#         plt.scatter(st_data[1,:], st_data[0,:])
#         plt.show()

#         plt.figure(figsize=(15,10))
#         plt.scatter(st_data[0,:], st_data[1,:])
#         plt.show()

#         # for i in [39,40,41,42,43,44,45,46,47,48,49,50]:
#         #     mask = st_data[2,:]==i/50
#         #     plt.figure(figsize=(15,10))
#         #     plt.scatter(st_data[0,mask], st_data[1,mask])
#         #     plt.show()


#         fig = plt.figure(figsize=(50,50))
#         ax = fig.add_subplot(projection='3d')
#         ax.scatter(st_data[0,:], st_data[1,:], st_data[2,:], marker = '.',s=25)
#         plt.show()

#     if inp_type=='sampled':
#         st_mask = sampled_memory[2]==0

#         plt.figure(figsize=(15,10))
#         plt.scatter(sampled_memory[0][:,1][st_mask], sampled_memory[0][:,0][st_mask])
#         plt.show()

#         plt.figure(figsize=(15,10))
#         plt.scatter(sampled_memory[0][:,1][st_mask], sampled_memory[0][:,1][st_mask])
#         plt.show()


        # fig = plt.figure(figsize=(50,50))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(sampled_memory[0][:,0][st_mask], sampled_memory[0][:,1][st_mask], sampled_memory[0][:,1][st_mask], marker = '.',s=25)
        # plt.show()

# plot_inputs(inp_type='orig')
# plot_inputs(inp_type='sampled')
