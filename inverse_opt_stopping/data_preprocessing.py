# from utils import to_bool, shift_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Data pre-processing helper-functions
def to_bool(s):
    return 1 if s == 'true' else 0

def shift_array(array, place=-1):
    # (path, time)
    new_arr = np.roll(array, place, axis=1)
    if len(array.shape)==2:
        new_arr[:,place:] = np.nan
        # new_arr[np.isnan(new_arr[place-1,:]),place:] = np.nan
        # new_arr[place:,np.isnan(new_arr[place-1,:])] = np.nan
        # new_arr[place:,np.isnan(new_arr[place-1,:])] = np.nan
    elif len(array.shape)==3:  
        new_arr[place:,:,:] = 0
        new_arr[place:,0,np.isnan(new_arr[place-1,0,:])] = np.nan
        new_arr[place:,1,np.isnan(new_arr[place-1,1,:])] = np.nan
        new_arr[place:,2,np.isnan(new_arr[place-1,2,:])] = np.nan
        # new_arr[np.isnan(new_arr):] = np.nan
    return new_arr

def preprocess_data(data, scale=False):
    # print(f'data.shape: {data.shape}')
    L = data.shape[0]-1
    data = np.hstack([data.copy(), np.repeat(np.array([i/L for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(L+1,1,-1)])
    data = np.hstack([data.copy(), np.repeat(np.array([i for i in np.arange(0,data.shape[0])]), data.shape[2]).T.reshape(L+1,1,-1)])
    data = np.hstack([data, np.array([np.repeat(i, data.shape[0]) for i in range(data.shape[2])]).T.reshape(L+1,1,-1)])

    inf_ids = np.isinf(data)[:,0,:,][:,None,:]
    data[:,2,:][:,None,:][inf_ids] = np.inf
    data[np.isinf(data)] = np.nan
    shifted_data = shift_array(data)


    nan_mask = np.isnan(data[:,:,:]).T
    nan_mask_next = np.isnan(shifted_data[:,0,:]).T
    nan_mask_last = nan_mask_next.copy()
    nan_mask_last[:,-1] = True

    DONE_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
    ACTION_MEM_SIM = nan_mask_last.astype(int)[~nan_mask[:,0,:]].flatten()
    ACTION_MEM_SIM = (ACTION_MEM_SIM-1)**2
    # X1,X2,Time, DONE
    STATE_MEM_SIM = np.array([data[:,0,:].T[~nan_mask[:,0,:]].flatten(), data[:,1,:].T[~nan_mask[:,1,:]].flatten(), data[:,2,:].T[~nan_mask[:,2,:]].flatten()]).T
    STATE2_MEM_SIM = np.array([shifted_data[:,0,:].T[~nan_mask[:,0,:]].flatten(), shifted_data[:,1,:].T[~nan_mask[:,1,:]].flatten(), shifted_data[:,2,:].T[~nan_mask[:,2,:]].flatten()]).T
    STATE2_MEM_SIM = np.nan_to_num(STATE2_MEM_SIM.copy())


    TIME_IDS = np.array(data[:,3,:].T[~nan_mask[:,0,:]].flatten())
    PATH_IDS = np.array(data[:,4,:].T[~nan_mask[:,0,:]].flatten())

    state_mem = np.array(STATE_MEM_SIM)

    if scale:
        st_mean = np.nanmean(state_mem,axis=0)
        st_mean[2] = 0
        st_std = np.nanstd(state_mem, axis=0)
        st_std[2] = 1
        print(f'st_std: {st_std}')
        print(f'st_mean: {st_mean}')
        st_min = np.nanmin(state_mem, axis=0)
        st_max = np.nanmax(state_mem, axis=0)
        print(f'st_min: {st_min}')
        print(f'st_max: {st_max}')
        memory = [(state_mem.copy()-st_mean)/st_std,
                      ((np.array(STATE2_MEM_SIM).copy()-st_mean)/st_std),
                      np.array(ACTION_MEM_SIM).copy(),
                      np.array(DONE_MEM_SIM).copy(),
                      np.array(TIME_IDS).copy(),
                      np.array(PATH_IDS).copy()
                      ]

    memory = [(state_mem.copy()),
                  ((np.array(STATE2_MEM_SIM).copy())),
                  np.array(ACTION_MEM_SIM).copy(),
                  np.array(DONE_MEM_SIM).copy(),
                  np.array(TIME_IDS).copy(),
                  np.array(PATH_IDS).copy()
                  ]
    return memory,data

def plot_inputs(data, sampled_memory, inp_type='orig'):
    if inp_type=='orig':
        inp_data = data.copy()
        data_ext = np.vstack([data.copy(), np.full_like(data[0,:,:], np.nan).reshape(1,5,-1)])
        st_times = np.array([np.where(np.isnan(data_ext[:,0,i]))[0][0] for i in range(data_ext.shape[2])]).copy()-1
        st_data = np.zeros_like(data[0,:,:].reshape(5,-1))
        for i in range(data.shape[2]):
            st_data[:,i] = data[st_times[i], :, i].copy()

        plt.figure(figsize=(15,10))
        plt.scatter(st_data[2,:], st_data[0,:])
        plt.show()

        plt.figure(figsize=(15,10))
        plt.scatter(st_data[2,:], st_data[1,:])
        plt.show()

        fig = plt.figure(figsize=(50,50))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(st_data[0,:], st_data[1,:], st_data[2,:], marker = '.',s=25)
        plt.show()
        
    if inp_type=='sampled':
        st_mask = sampled_memory[2]==0

        plt.figure(figsize=(15,10))
        plt.scatter(sampled_memory[0][:,2][st_mask], sampled_memory[0][:,0][st_mask])
        plt.show()

        plt.figure(figsize=(15,10))
        plt.scatter(sampled_memory[0][:,2][st_mask], sampled_memory[0][:,1][st_mask])
        plt.show()


        fig = plt.figure(figsize=(50,50))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sampled_memory[0][:,0][st_mask], sampled_memory[0][:,1][st_mask], sampled_memory[0][:,2][st_mask], marker = '.',s=25)
        plt.show()


# #%%
# # HDBSCAN test
# import hdbscan
# sampled_memory,data = preprocess_data(full_data[:,:,:10000])
# st_points = sampled_memory[0][sampled_memory[2]==0]
# # st_points = sampled_memory[0]
# clusterer = hdbscan.HDBSCAN(min_cluster_size=25, gen_min_span_tree=True, 
#                             cluster_selection_epsilon=0.25)
# clusterer.fit(st_points)

# # clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
# #                                       edge_alpha=0.6,
# #                                       node_size=80,
# #                                       edge_linewidth=2)

# # clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
# # clusterer.condensed_tree_.plot()
# clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
# palette = sns.color_palette()
# cluster_colors = [sns.desaturate(palette[col], sat)
#                    if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
#                    zip(clusterer.labels_[clusterer.labels_!=(-1)], clusterer.probabilities_[clusterer.labels_!=(-1)])]
# plt.figure()
# plt.scatter(st_points[:,2][clusterer.labels_!=(-1)], st_points[:,0][clusterer.labels_!=(-1)], c=cluster_colors,s=5)
# plt.show()

# plt.figure()
# plt.scatter(st_points[:,2][clusterer.labels_!=(-1)], st_points[:,1][clusterer.labels_!=(-1)], c=cluster_colors,s=5)
# plt.show()

# fig = plt.figure(figsize=(50,50))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(st_points[:,0][clusterer.labels_!=(-1)], st_points[:,1][clusterer.labels_!=(-1)], st_points[:,2][clusterer.labels_!=(-1)], c=cluster_colors, marker = '.',s=50)
# plt.show()