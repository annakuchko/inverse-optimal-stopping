# %%
# sim changepoint
import random
import numpy as np
 
def shift_array(array, place=-1):
    new_arr = np.roll(array, place, axis=0)
    new_arr[place:,:,:] = 0
    new_arr[place:,0,np.isnan(new_arr[place-1,0,:])] = np.nan
    new_arr[place:,1,np.isnan(new_arr[place-1,1,:])] = np.nan
    new_arr[place:,2,np.isnan(new_arr[place-1,2,:])] = np.nan
    # new_arr[np.isnan(new_arr):] = np.nan
    return new_arr
 
class CP1:
    def __init__(self, cp_type='periodic', total_n=75):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 5
        self.scale0 = 1
        self.scale1 = 1
        self.total_n = total_n
        self.sim_data(path_length=51)
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
        data = self.data[:,:,ids]
        data = np.hstack([data.copy(), np.repeat(np.array([i/data.shape[0] for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data.copy(), np.repeat(np.array([i for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data, np.array([np.repeat(i, data.shape[0]) for i in range(data.shape[2])]).T.reshape(data.shape[0],1,-1)])

        inf_ids = np.isinf(data)[:,0,:,][:,None,:]
        data[:,1,:][:,None,:][inf_ids] = np.inf
        data[np.isinf(data)] = np.nan
        shifted_data = shift_array(data)

        nan_mask = np.isnan(data[:,:,:])
        nan_mask_next = np.isnan(shifted_data[:,0,:])
        nan_mask_last = nan_mask_next.copy()
        DONE_MEM_SIM = nan_mask_last.astype(int).T[~nan_mask[:,0,:].T].flatten()
        action_mem = (nan_mask_next.astype(int)-1)**2
        action_mem[-1,:] = 0
      
        if True:
            st = action_mem.copy()
            first_st = np.argmax(st==0,axis=0)
            # first_st[first_st==0] = 10000000
            print(f'first_st: {first_st}')
            idxrow,idxcol=np.indices(st.shape)
            first_st2=first_st[None,:]
            mask = idxrow > first_st2
            st[mask]=0
            ACTION_MEM_SIM = st.T.flatten()[~nan_mask[:,0,:].T.flatten()].flatten()
            # ACTION_MEM_SIM = (ACTION_MEM_SIM
        # X1,X2,Time, DONE
        STATE_MEM_SIM = np.array([data[:,0,:].T[~nan_mask[:,0,:].T].flatten().T,data[:,1,:].T[~nan_mask[:,0,:].T].flatten().T]).T
        STATE2_MEM_SIM = np.array([shifted_data[:,0,:].T[~nan_mask[:,0,:].T].flatten(), shifted_data[:,1,:].T[~nan_mask[:,0,:].T].flatten()]).T
        STATE2_MEM_SIM = np.nan_to_num(STATE2_MEM_SIM.copy())
        TIME_IDS = np.array(data[:,-2,:].T[~nan_mask[:,0,:].T].flatten())
        PATH_IDS = np.array(data[:,-1,:].T[~nan_mask[:,0,:].T].flatten())

        state_mem = np.array(STATE_MEM_SIM)
        INIT_STATES = np.zeros_like(TIME_IDS)
        
            
        return {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}
 
    def sim_data(self,path_length):
        # time, shape, path_id
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1), round(path_length*0.3)))
            start_time = np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0)
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1)
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem     
 
 
class CP2(CP1):
    def sim_data(self,path_length):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 0.5
        self.scale0 = 1
        self.scale1 = 1
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1), round(path_length*0.3)))
            start_time = np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0) + 0.25*path[-1]+ 0.05*path[-2]
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1) + 0.75*path[-1]+ 0.5*path[-2]
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem   
 
class CP3(CP1):
    def sim_data(self,path_length):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 0.5
        self.scale0 = 1
        self.scale1 = 5
        # time, shape, path_id
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1), round(path_length*0.3)))
            start_time = np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0) #+ 0.5*path[-1]+ 0.25*path[-2]
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1) #+ 0.5*path[-1]+ 0.25*path[-2]
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem   
 
class CP4(CP1):
    def sim_data(self,path_length):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 0.5
        self.scale0 = 1
        self.scale1 = 5
        # time, shape, path_id
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1),
                                                           round(path_length*0.3)))
            start_time = np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0) #+ 0.5*path[-1]+ 0.25*path[-2]
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1) #+ 0.5*path[-1]+ 0.25*path[-2]
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem   
class CP1M:
    def __init__(self, cp_type='periodic', total_n=75):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 5
        self.scale0 = 1
        self.scale1 = 1
        self.total_n = total_n
        self.sim_data(path_length=51)
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
        data = self.data[:,:,ids]
        data = np.hstack([data.copy(), np.repeat(np.array([i/data.shape[0] for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data.copy(), np.repeat(np.array([i for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(data.shape[0],1,-1)])
        data = np.hstack([data, np.array([np.repeat(i, data.shape[0]) for i in range(data.shape[2])]).T.reshape(data.shape[0],1,-1)])

        inf_ids = np.isinf(data)[:,0,:,][:,None,:]
        data[:,1,:][:,None,:][inf_ids] = np.inf
        data[np.isinf(data)] = np.nan
        shifted_data = shift_array(data)

        nan_mask = np.isnan(data[:,:,:])
        nan_mask_next = np.isnan(shifted_data[:,0,:])
        nan_mask_last = nan_mask_next.copy()
        DONE_MEM_SIM = nan_mask_last.astype(int).T[~nan_mask[:,0,:].T].flatten()
        action_mem = (nan_mask_next.astype(int)-1)**2
        action_mem[-1,:] = 0
      
        if True:
            st = action_mem.copy()
            first_st = np.argmax(st==0,axis=0)
            # first_st[first_st==0] = 10000000
            print(f'first_st: {first_st}')
            idxrow,idxcol=np.indices(st.shape)
            first_st2=first_st[None,:]
            mask = idxrow > first_st2
            st[mask]=0
            ACTION_MEM_SIM = st.T.flatten()[~nan_mask[:,0,:].T.flatten()].flatten()
            # ACTION_MEM_SIM = (ACTION_MEM_SIM
        # X1,X2,Time, DONE
        STATE_MEM_SIM = np.array([data[:,0,:].T[~nan_mask[:,0,:].T].flatten().T,data[:,1,:].T[~nan_mask[:,0,:].T].flatten().T]).T
        STATE2_MEM_SIM = np.array([shifted_data[:,0,:].T[~nan_mask[:,0,:].T].flatten(), shifted_data[:,1,:].T[~nan_mask[:,0,:].T].flatten()]).T
        STATE2_MEM_SIM = np.nan_to_num(STATE2_MEM_SIM.copy())
        TIME_IDS = np.array(data[:,-2,:].T[~nan_mask[:,0,:].T].flatten())
        PATH_IDS = np.array(data[:,-1,:].T[~nan_mask[:,0,:].T].flatten())

        state_mem = np.array(STATE_MEM_SIM)
        INIT_STATES = np.zeros_like(TIME_IDS)
        
            
        return {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}
 
    def sim_data(self,path_length):
        # time, shape, path_id
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1), round(path_length*0.3)))
            start_time = 0#np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0)
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1)
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem     
 
 
class CP2M(CP1M):
    def sim_data(self,path_length):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 0.5
        self.scale0 = 1
        self.scale1 = 1
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1), round(path_length*0.3)))
            start_time = 0#np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0) + 0.25*path[-1]+ 0.05*path[-2]
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1) + 0.75*path[-1]+ 0.5*path[-2]
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem   
 
class CP3M(CP1M):
    def sim_data(self,path_length):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 0.5
        self.scale0 = 1
        self.scale1 = 5
        # time, shape, path_id
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1), round(path_length*0.3)))
            start_time = 0#np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0) #+ 0.5*path[-1]+ 0.25*path[-2]
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1) #+ 0.5*path[-1]+ 0.25*path[-2]
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem   
 
class CP4M(CP1M):
    def sim_data(self,path_length):
        self.w0 = 1
        self.w1 = 1
        self.loc0 = 0.5
        self.loc1 = 0.5
        self.scale0 = 1
        self.scale1 = 5
        # time, shape, path_id
        paths_mem = np.full((path_length, 1, self.total_n), np.nan)
        for p in range(self.total_n):
            path = [0,0]
            cp_time = path_length - np.random.choice(range(round(path_length*0.1),
                                                           round(path_length*0.3)))
            start_time = 0#np.random.choice(range(round(path_length*0.5)))
            for t in range(path_length):
                if t<cp_time:
                    xt = np.sin(self.w0*t)+np.random.normal(loc=self.loc0,scale=self.scale0) #+ 0.5*path[-1]+ 0.25*path[-2]
                else:
                    xt = np.sin(self.w1*t)+np.random.normal(loc=self.loc1,scale=self.scale1) #+ 0.5*path[-1]+ 0.25*path[-2]
                path.append(xt)
            final_path = np.array(path[2:])[start_time:cp_time+2]
            paths_mem[:len(final_path),0,p] = final_path
        self.data = paths_mem   
#%%  
if __name__=='__main__':
    cp = CP4(total_n=500)
    train_data = cp.sim_expert()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18,13))
    plt.scatter(train_data['state_mem'][:,1][train_data['action_mem']==1],
                train_data['state_mem'][:,0][train_data['action_mem']==1], 
                color='green', 
                s=30, 
                label='a=1')
    plt.scatter(train_data['state_mem'][:,1][train_data['action_mem']==0],
                train_data['state_mem'][:,0][train_data['action_mem']==0],
                color='red', 
                s=50, 
                label='a=0')
    plt.xlabel("t",fontsize=50)
    plt.ylabel("x",fontsize=50)
        
    plt.tick_params(labelsize=45,axis='x', labelrotation=90)
    plt.tick_params(labelsize=45,axis='y')
    plt.legend(prop={'size': 45})
    plt.title('CP4', fontsize=50)
    plt.tight_layout()
    plt.show()