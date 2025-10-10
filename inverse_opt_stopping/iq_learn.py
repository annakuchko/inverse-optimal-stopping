from inverse_opt_stopping.q_networks import OfflineQNetwork, OfflineQNetwork_orig, gNetwork, DoubleOfflineQNetwork, DoubleOfflineQNetwork_orig
import torch
import torch.nn.functional as F
import numpy as np
import numpy
from torch import logsumexp as softmax
from torch.distributions import Categorical
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import compress
from scipy.ndimage import gaussian_filter
import seaborn
import pandas
import matplotlib.pyplot as pyplot
from scipy import interpolate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import random
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib.patches import Rectangle
from imblearn.over_sampling import SMOTE, ADASYN 
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
# 'js'
sns.set_style("whitegrid")
plt.rcParams['axes.facecolor'] = 'white'

from sklearn.model_selection import train_test_split 
# train, test = train_test_split(X, test_size=0.25, stratify=X['YOUR_COLUMN_LABEL']) 

def random_batch_split(indexes, batch_size, stratify=None):
    batches = []
    ids_left = indexes.copy()
    # print(f'total n of st: {sum(stratify==0)}')
    while len(ids_left)>batch_size:    
        # print(f'test_size: {batch_size/len(ids_left)}')
        _, batch_ids = train_test_split(ids_left, test_size=batch_size, stratify=stratify[ids_left])
        # _, batch_ids = train_test_split(ids_left, test_size=batch_size)
        # batch_ids = random.choices(ids_left, k=batch_size)
        batches.append(batch_ids)
        # print(f'st actions in a batch: {sum(stratify[batch_ids]==0)}')
        ids_left = np.setdiff1d(ids_left, batch_ids)
    batches.append(list(ids_left))
    return batches

class IQ_Agent:
    def __init__(self, obs_dim=None, action_dim=2, divergence='kl_fix', approx_g=True, approx_dynamics=True, gamma=0.99, epsilon=0.1, q_lr=0.001,
                 env_lr=0.001, g_lr=0.001,device='cpu',dt=1,
                 oversampling=None, plot_against_time=False, seed=None,
                 cross_val_splits=3, conservative=False, 
                             discretiser=None, 
                             out_thresh=0.005):
        if seed is None:
            np.random.seed()
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.plot_against_time = plot_against_time
        self.gamma = gamma
        self.approx_dynamics = approx_dynamics
        self.approx_g = approx_g
        self.oversampling = oversampling
        self.device = device
        self.approx_dynamics = approx_dynamics
        self.cross_val_splits = cross_val_splits
        self.conservative = conservative 
        self.discretiser=discretiser 
        self.out_thresh = out_thresh
        self.lr_decay = 0.9999
        self.cs = 0.999
        self.cs_decay = 0.99
        # if cross_val_splits!=1:
            # self.sss = StratifiedKFold(n_splits=cross_val_splits, shuffle=True)
            # self.sss = StratifiedShuffleSplit(n_splits=1, random_state=2024,
            #                          train_size=0.70, test_size=0.30)
        if obs_dim is None:
            raise ValueError(f'Specify state dimensionality')
        if approx_dynamics:
            
            if approx_g:
                g_net = gNetwork(obs_dim=obs_dim, action_dim=action_dim, gamma=self.gamma,
                                    device=device) 
                g_net.apply(g_net.weights_init_uniform)
                self.g_net = g_net.to(self.device)
                print(f'g_net ON CUDA: {next(self.g_net.parameters()).is_cuda}')
                self.g_loss = torch.nn.SmoothL1Loss().to(self.device)
                # self.g_loss = torch.nn.MSELoss().to(self.device)
                self.g_optimizer = torch.optim.RMSprop(self.g_net.parameters(), lr=g_lr)
                # self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.g_optimizer,
                #                                                         gamma=0.9999)
                self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.g_optimizer,
                                                                        mode='max', factor=0.9999, patience=2,)
                
                # q_net = OfflineQNetwork(obs_dim=obs_dim+1, action_dim=action_dim, gamma=self.gamma,
                #                     device=device)
                q_net = DoubleOfflineQNetwork(obs_dim=obs_dim+1, action_dim=action_dim, gamma=self.gamma,
                                    device=device)
                
                if oversampling not in ["LSMOTE","CS-LSMOTE", None]:
                # if oversampling is not None:
                    raise ValueError(f'Cannot do {oversampling} oversampling when approximating g. Use LSMOTE.')
            else:
                # q_net = OfflineQNetwork(obs_dim=obs_dim, action_dim=action_dim, gamma=self.gamma,
                #                     device=device)
                q_net = DoubleOfflineQNetwork(obs_dim=obs_dim, action_dim=action_dim, gamma=self.gamma,
                                    device=device)
                
            q_net.apply(q_net.weights_init_uniform)
            self.q_net = q_net.to(self.device)
            self.env_loss = torch.nn.SmoothL1Loss().to(self.device)
            # self.env_loss = torch.nn.MSELoss().to(self.device)
            self.env_optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=env_lr,)
            # self.env_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.env_optimizer,
                                                                    # gamma=0.9999)
            
            self.env_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.env_optimizer,
                                                                        mode='max', factor=0.9999, patience=2,)
                        
        else:
            q_net = DoubleOfflineQNetwork_orig(obs_dim=obs_dim, action_dim=action_dim, gamma=self.gamma,
                                device=device) 
            q_net.apply(q_net.weights_init_uniform)
            self.q_net = q_net.to(self.device)
            # self.env_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.env_optimizer,
                                                                    # gamma=0.9999)
            
            
        
        print(f'q_net ON CUDA: {next(self.q_net.parameters()).is_cuda}')
        self.critic_optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=q_lr,)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.critic_optimizer,
        #                                                         gamma=0.9999)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.critic_optimizer,
                                                                        mode='max', factor=0.9999, patience=2,)
            
        self.dt = dt
        self.divergence = divergence
        self.epsilon = epsilon
        self.epoch = 0

    def get_rewards(self, state, y):
        self.q_net.eval()
        irl_q0 = self.infer_q(state=state, action=np.zeros_like(state[:,0]))
        irl_q1 = self.infer_q(state=state, action=np.ones_like(state[:,0]))
        irl_next_s = self.infer_next_s(state=state)
        irl_V = self.getV(irl_next_s, evaluate=True)
        # recover rewards
        irl_G = irl_q0
        irl_g = irl_q1 - self.q_net.gamma*irl_V.cpu() - y
        return irl_G, irl_g

    def getQ(self, obs, evaluate=False):
        if evaluate:
            self.q_net.eval()
            with torch.no_grad():
                q = self.q_net.get_Q(obs.clone().to(self.device))
        else:
            self.q_net.train()
        # with torch.no_grad():
            q = self.q_net.get_Q(obs.clone().to(self.device))
        return q


    def getQ_s_a(self, states, actions, evaluate=False):
        Qs = self.getQ(states, evaluate)
        Q_s_a = torch.gather(Qs.cpu(), dim=-1, index=torch.tensor(actions).to(torch.int64).cpu().reshape(-1, 1))
        return Q_s_a


    def getS_dash(self, states, evaluate=False):
        if evaluate:
            self.q_net.eval()
        else:
            self.q_net.train()
        with torch.no_grad():
            s_dash = self.q_net.get_next_s(states.clone().to(self.device)).cpu()+states.clone()
        return s_dash

    def getV(self, obs, evaluate=False):
        q = self.getQ(obs, evaluate)
        v = self.epsilon * softmax(q/self.epsilon, dim=1, keepdim=True)
        return v

    def infer_q(self, state, action):
        return self.getQ_s_a(state, action, evaluate=True)

    def infer_next_s(self, state):
        return self.getS_dash(state, evaluate=True)

    def g_update(self, batch, epoch):
        obs = batch[0]
        y, y_next, y_prev = batch[-3:]
        obs_ext = torch.hstack([obs, y]).clone()
        # obs_ext_prev = torch.hstack([obs, y_prev]).clone()
        self.g_net.train()
        
        # g_phi = self.g_net(obs_ext_prev.to(self.device))
        g_phi = self.g_net(obs.to(self.device))
        g_iq = self.get_rewards(obs_ext,y_prev)[1]
        # print()
        g_loss = self.g_loss(g_phi.to(self.device).float(), g_iq.to(self.device).float()).to(self.device)
        # if epoch %1==0:
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
            
            
        return g_loss.detach().cpu().numpy()

    def iq_update(self, batch, epoch, cs, update=False):
        # expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
        obs, next_obs, action = batch[0:3]
        if self.approx_g:
            y,y_next,y_prev = batch[-3:]
            obs = torch.hstack([obs, y]).clone()
            next_obs = torch.hstack([next_obs, y_next]).clone()

        next_obs_k = next_obs-obs
        if self.approx_g:
            self.g_net.train()

        self.q_net.train()

        current_V = self.getV(obs, evaluate=False)
        if self.approx_dynamics:
            current_Q1,current_Q2, next_state_k1, next_state_k2 = self.q_net(obs.to(self.device), action.to(self.device))
            
            next_obs_tilde1 = next_state_k1.clone().cpu()+obs[:,:].clone()
            next_obs_tilde2 = next_state_k2.clone().cpu()+obs[:,:].clone()
        else:
            current_Q1, current_Q2 = self.q_net(obs.to(self.device), action.to(self.device))
            next_obs_tilde1 = next_obs[:,:].clone().cpu()
            next_obs_tilde2 = next_obs_tilde1.clone()
        next_V1 = self.getV(next_obs_tilde1, evaluate=False)
        next_V2 = self.getV(next_obs_tilde2, evaluate=False)


        critic_loss1, loss_dict1 = self.iq_loss(self.q_net, current_Q1.to(self.device), current_V.to(self.device), next_V1.to(self.device), batch, cs)
        critic_loss2, loss_dict2 = self.iq_loss(self.q_net, current_Q2.to(self.device), current_V.to(self.device), next_V2.to(self.device), batch, cs)
        critic_loss = (critic_loss1+critic_loss2)/2
        loss_names = list(loss_dict1)
        avg_losses = [(loss_dict1[i]+loss_dict2[i])/2 for i in loss_names]
        loss_dict = dict(zip(loss_names, avg_losses))
        if self.approx_dynamics:
            state_loss1 = self.env_loss(next_state_k1.to(self.device).float(), next_obs_k[:,:].to(self.device).float()).to(self.device)
            state_loss2 = self.env_loss(next_state_k2.to(self.device).float(), next_obs_k[:,:].to(self.device).float()).to(self.device)
            state_loss = (state_loss1 + state_loss2)/2
            loss_dict['env_loss'] = state_loss.detach().cpu().numpy()
        # Optimize the critic
        
        self.critic_optimizer.zero_grad()
        if self.approx_dynamics:
            if epoch % 5==0:
                self.env_optimizer.zero_grad()
                state_loss.backward(retain_graph=True, create_graph=True)

        critic_loss.backward()
        # step critic
        if self.approx_dynamics:
            self.env_optimizer.step()
        self.critic_optimizer.step()
        
        return loss_dict


    def choose_action(self, state, sample=False):
        if isinstance(state):
            state = np.array(state)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q, next_state_k = self.q_net(state.to(self.device))
            dist = F.softmax(q/self.epsilon, dim=1)
            dist = Categorical(dist)
            action = torch.argmax(dist, dim=1)

        return action.detach().cpu().numpy()[0]

    def test(self, test_memory, from_grid=False, validation=False):
        if not validation:
            self.q_net.load_state_dict(self.best_q_net)
            if self.approx_g:
                self.g_net.load_state_dict(self.best_g_net)
        self.q_net.eval()
        if self.approx_g:
            self.g_net.eval()
        conservative=self.conservative 
        discretiser=self.discretiser
        out_thresh=self.out_thresh 
        
        if from_grid:
            self.plot_st_reg_bm(test_memory, conservative, discretiser, out_thresh, name=f'Out-degeee threshold={out_thresh}')
        else:
            if conservative: 
                unique_ints, k_outs = discretiser.ints
                state_discretised = discretiser.transform(test_memory['state_mem'])
                out_degree_test = np.zeros_like(state_discretised[:,0])
                for i in range(len(k_outs)):
                    out_degree_test[(state_discretised==unique_ints[i]).all(axis=1)] =  k_outs[i]

            mem_keys = list(test_memory.keys())
            Q0 = np.full((np.unique(test_memory['path_ids']).shape[0],np.max(test_memory['time_ids']).astype(int)+1), np.nan)
            Q1 = np.full((np.unique(test_memory['path_ids']).shape[0],np.max(test_memory['time_ids']).astype(int)+1), np.nan)
            if self.approx_g:
                Y_mem = np.full((np.unique(test_memory['path_ids']).shape[0],np.max(test_memory['time_ids']).astype(int)+1), np.nan)
            
            for p, path_id in enumerate(np.unique(test_memory['path_ids'].astype(int))):
                sample={}
                for i in range(len(mem_keys)):
                    sample[mem_keys[i]] = torch.from_numpy(np.array(list(compress(test_memory[mem_keys[i]],test_memory['path_ids']==path_id)))) 
                sample['state_mem'] = (sample['state_mem']-self.mean)/self.std
                if conservative: 
                    sample['k_out'] = np.array(list(compress(out_degree_test,test_memory['path_ids']==path_id)))
                if self.approx_g:
                    y_hist = [0] 
                    y_hist_prev = [0]
                    # y_next_hist = []
                Q0_results = []
                Q1_results = []
                
                # print(f"path: {path_id}, sample['time_ids']: {sample['time_ids']}")
                # y_temp = [0]
                for time_id in sample['time_ids']:
                    # print(f'time_id: {time_id}')
                    with torch.no_grad():
                        base_state = sample['state_mem'][sample['time_ids']==time_id].reshape(1,-1)
                        
                        if self.approx_g:
                            # ext_state = torch.hstack([base_state.clone(), torch.from_numpy(np.array(y_hist[-1])).reshape(1,-1)])
                            irl_g = self.g_net(base_state)
                            y_hist.append((y_hist_prev[-1]+ self.gamma* irl_g.clone()).detach().cpu().numpy()[0][0])
                            if time_id==0:
                                y_hist_prev.append(0)
                            else:
                                y_hist_prev.append(y_hist[-2])
                            state = torch.hstack([base_state.clone(), torch.from_numpy(np.array(y_hist[-1])).reshape(1,-1)])
                        else:
                            state = base_state.clone()
                            
                        irl_q0 = self.infer_q(state=state, action=np.zeros((1))).cpu()
                        irl_q1 = self.infer_q(state=state, action=np.ones((1))).cpu()
                        
                        # if self.approx_g:
                            
                        
                                
                        if conservative and sample['k_out'][(sample['time_ids']==time_id).cpu().detach().numpy()].reshape(1,-1)<out_thresh: 
                            Q0_results.append(0)
                            Q1_results.append(0)
                        else: 
                            Q0_results.append(irl_q0.detach().cpu().numpy()[0][0])
                            Q1_results.append(irl_q1.detach().cpu().numpy()[0][0])
                        # if self.approx_g:                    
                            
                Q0_results = np.array(Q0_results)
                Q1_results = np.array(Q1_results)
                st_id = np.where(Q0_results>=Q1_results)[0]
                if st_id.size!=0:
                    if isinstance(st_id, np.ndarray):
                        st_id = st_id[0]
                    Q0_results[st_id+1:] = 0
                    Q1_results[st_id+1:] = 0
                
                Q0[p,:len(Q0_results)] = Q0_results
                Q1[p,:len(Q1_results)] = Q1_results
                if self.approx_g:
                    Y_mem[p,:len(y_hist[:-1])] = y_hist[:-1]
            
            # X1 = (X1*self.std[0])+self.mean[0]
            # X2 = (X2*self.std[1])+self.mean[1]
            
            # if self.approx_g:
            #     df_y = pd.DataFrame({"x1":X1.flatten(),
            #         "x2":X2.flatten(),
            #       "Y":Y_mem.flatten()})
    
            #     data_y = pd.pivot_table(df_y.copy(), values='Y', index='x1', columns='x2')
            #     from scipy.ndimage import gaussian_filter
            #     data_norm0 = gaussian_filter(data0.fillna(0), sigma=5, mode='constant')
            #     data_norm0[data_norm0==0]=None
            #     data_norm_y = data_y.copy()
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm_y)
            #     plt.title(f'Y_mem, no gaussian smoothing, x2 vs x1')
            #     plt.show()
            
            #     data_norm_y = gaussian_filter(data_y.fillna(0).copy(), sigma=3, mode='constant')
            #     data_norm_y[data_norm_y==0]=None
            #     data_norm_y = data_y
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm_y.copy())
            #     plt.title(f'Y_mem, gaussian smoothing, x2 vs x1')
            #     plt.show()
            
            
            #     df_y1 = pd.DataFrame({"x2":X2.flatten(),
            #             "t":T.flatten(),
            #           "Y":Y_mem.flatten()})
        
            #     data_y1 = pd.pivot_table(df_y1.copy(), values='Y', index='x2', columns='t')
            #     from scipy.ndimage import gaussian_filter
            #     # data_norm0 = gaussian_filter(data0.fillna(0), sigma=5, mode='constant')
            #     # data_norm0[data_norm0==0]=None
            #     data_norm_y1 = data_y1.copy()
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm_y1)
            #     plt.title(f'Y_mem, no gaussian smoothing, t vs x2')
            #     plt.show()
            
                # data_norm_y1 = gaussian_filter(data_y1.fillna(0).copy(), sigma=3, mode='constant')
                # data_norm_y1[data_norm_y1==0]=None
                # # data_norm_y = data_y
                # plt.figure(figsize=(15,10))
                # sns.heatmap(data_norm_y1)
                # plt.title(f'Y_mem, gaussian smoothing, t vs x2')
                # plt.show()
                
            # Q_diff = Q0.copy()-Q1.copy()
            # st = Q_diff.copy()
            # st[st>=0]=0
            # st[st<0]=1
            
            # firs_st = np.argmax(st==0,axis=1)
            # idxrow,idxcol=np.indices(st.shape)
            # firs_st2=firs_st[:,None]
            # mask = idxcol > firs_st2
            # st[mask]=0
            
            
            Q_diff = Q0.copy()-Q1.copy()
            st = Q_diff.copy()
            st[st>=0]=0
            st[st<0]=1
            
            firs_st = np.array([np.where(st[i,:]==0,1,np.nan)[0] for i in range(st.shape[0])])
            idxrow,idxcol=np.indices(st.shape)
            firs_st2=firs_st[:,None]
            mask = idxcol > firs_st2
            st[mask]=0
            st = st.flatten()
            n_paths = len(np.unique(test_memory['path_ids']))
            n_times = len(np.unique(test_memory['time_ids']))
            # st[k_out<=out_thresh] = 0
            
            # firs_st = np.argmax(st==0,axis=1)
            # idxrow,idxcol=np.indices(st.shape)
            # firs_st2=firs_st[:,None]
            # mask = idxcol > firs_st2
            # st[mask]=0
            # firs_st = np.array([np.where(st[i,:]==0,1,np.nan)[0] for i in range(st.shape[0])])
            # idxrow,idxcol=np.indices(st.shape)
            # firs_st2=firs_st[:,None]
            # mask = idxcol > firs_st2
            # st[mask]=0
            # st = st.flatten()
            # n_paths = len(np.unique(test_memory['path_ids']))
            # n_times = len(np.unique(test_memory['time_ids']))
            from sklearn.metrics import roc_auc_score
            # print(f'np.array([np.arange(n_paths) for i in range(n_paths)]).reshape(-1): {np.array([np.arange(n_paths) for i in range(n_paths)]).reshape(-1).shape}')
            # print(f'np.array([np.repeat(i,n_paths) for i in range(n_paths)]).reshape(-1): {np.array([np.repeat(i,n_paths) for i in range(n_paths)]).reshape(-1).shape}')
            # st_imitation = pd.DataFrame({
            #     'path_ids': np.array([np.repeat(i,n_times) for i in range(
            #         (test_memory['path_ids']).min().astype(int), 
            #         (test_memory['path_ids']).astype(int).max()+1
            #         )]).reshape(-1),
            #     'time_ids': np.array([np.arange(
            #         (test_memory['time_ids']).astype(int).min(),
            #          (test_memory['time_ids']).astype(int).max()+1
            #          ) for i in range(n_paths)]).reshape(-1),
            #     'st_region':st
            #     })
            st_imitation = pd.DataFrame({
                'path_ids': np.array([np.repeat(i,st.shape[0]/len(np.unique(test_memory['path_ids']))) for i in np.unique(test_memory['path_ids'])]).reshape(-1),
                'time_ids': np.array([np.arange(
                    (test_memory['time_ids']).astype(int).min(),
                     (test_memory['time_ids']).astype(int).max()+1
                     ) for i in range(n_paths)]).reshape(-1),
                'st_region':st
                })
            # print(st_imitation)
            test_ids = [(test_memory['path_ids'][i], test_memory['time_ids'][i]) for i in range(len(test_memory['path_ids']))]
            ids_to_keep = []
            # don't need this filtering for proper test paths, since they should not be stopped
            for i in range(st.shape[0]):
                if (st_imitation.path_ids[i],st_imitation.time_ids[i]) in test_ids:
                    ids_to_keep.append(i)
            st_imitation = st_imitation.iloc[ids_to_keep]
            time_to_event = []
            event_miss = []
            for p, path in enumerate(np.unique(test_memory['path_ids'])):
                path_data = st_imitation[st_imitation.path_ids==path]
                path_expert = np.array([test_memory[i] for i in list(['time_ids','action_mem'])])[:,test_memory['path_ids']==path]
                expert_st_time = path_expert[0,:][path_expert[1,:]==0]
                if expert_st_time.size>0:
                    expert_st_time = expert_st_time[0]
                else:
                    expert_st_time = np.nan
                if (path_data.st_region==0).sum()==0:
                    imitation_st_time = np.array([np.nan])
                else:
                    imitation_st_time = path_data[path_data.st_region==0].time_ids.values[0]
                # print(f'imitation_st_time: {imitation_st_time}')
                # print(f'expert st time: {expert_st_time}')
                if np.isnan(imitation_st_time).sum()==1:
                    event_miss.append(1)
                elif np.isnan(expert_st_time):
                    event_miss.append(0)
                    time_to_event.append(path_expert[0,:].max()-imitation_st_time)   
                else:
                    if int(expert_st_time)<int(imitation_st_time):
                        event_miss.append(1)
                    else:
                        event_miss.append(0)
                        time_to_event.append(expert_st_time-imitation_st_time)
                        
            
            self.mtte = np.mean(time_to_event)
            self.memr = np.mean(event_miss)
            
            st_imitation = st[ids_to_keep]
            st_imitation[np.isnan(st_imitation)] = 0
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, balanced_accuracy_score
            # print(f'st_imitation: {st_imitation.shape}')
            # print(f'test_memory["action_mem"]: {test_memory["action_mem"].shape}')
            # print(f'stop=1, cont=0, average=weighted')
            # print(f'accuracy: {accuracy_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')
            # print(f'roc_auc: {roc_auc_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')
            # print(f'precision: {precision_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')
            # print(f'recall: {recall_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')]
            if not validation: 
                print(f'Balance Accuracy Score: {balanced_accuracy_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')
                print(f'F1: {f1_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')
                print(f'pr_auc: {average_precision_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')
                print(f'median time-to-event: {np.mean(time_to_event)}')
                print(f'median event miss-rate: {np.mean(event_miss)}')
                print(f'number of missed events: {np.sum(event_miss)}')
            
                cm = confusion_matrix(test_memory["action_mem"], 
                                      st_imitation, 
                                      normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.grid(False)
                square_coord =  ((-0.48,0.48,0.48,0.48,-0.48,-0.48),
                                 (-0.48,-0.48,0.48,0.48,0.48,-0.48))
                plt.plot(*square_coord, c="g", linewidth=2)
                square_coord =  ((0.52,1.48,1.48,1.48,0.52,0.52),
                                 (-0.48,-0.48,0.48,0.48,0.48,-0.48))
                plt.plot(*square_coord, c="r", linewidth=2)
                plt.show()
                
            # df00 = pd.DataFrame({"x1":X1.flatten(),
            #         "x2":X2.flatten(),
            #       "stopping":st})
    
            # data00 = pd.pivot_table(df00.copy(), values='stopping', index='x1', columns='x2')
            
            # plt.figure(figsize=(10,10),facecolor="w")
            # plt.scatter(X1.flatten()[st==0],X2.flatten()[st==0],s=0.3)
            # plt.rcParams['axes.facecolor'] = 'white'
            # plt.show()
            
            # from scipy.ndimage import gaussian_filter
            # data_norm0 = gaussian_filter(data00.fillna(0), sigma=5, mode='constant')
            # data_norm0[data_norm0==0]=None
            # data_norm00 = data00.copy()
            # plt.figure(figsize=(15,10))
            # sns.heatmap(data_norm00)
            # plt.title(f'Q_diff, no gaussian smoothing, x2 vs x1')
            # plt.show()
            
            # data_norm00 = gaussian_filter(data00.fillna(0).copy(), sigma=3, mode='constant')
            # data_norm00[data_norm00==0]=None
            
            # # data_norm1 = data1
            # plt.figure(figsize=(15,10))
            # sns.heatmap(data_norm00)
            # plt.title(f'Q_diff, gaussian smoothing, x1 vs x2')
            # plt.show()
            
            # self.plot_against_time = True
            # if self.plot_against_time:
            #     df0 = pd.DataFrame({"x1":X1.flatten(),
            #             "t":T.flatten(),
            #           "Q_diff":Q_diff.flatten()})
        
            #     data0 = pd.pivot_table(df0.copy(), values='Q_diff', index='x1', columns='t')
                
            #     # data_norm0 = gaussian_filter(data0.fillna(0), sigma=5, mode='constant')
            #     # data_norm0[data_norm0==0]=None
            #     data_norm0 = data0.copy()
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm0)
            #     plt.title(f'Q_diff, no gaussian smoothing, t vs x1')
            #     plt.show()
                
            #     data_norm01 = gaussian_filter(data0.fillna(0).copy(), sigma=3, mode='constant')
            #     data_norm01[data_norm01==0]=None
            #     # data_norm01 = data0
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm01)
            #     plt.title(f'Q_diff, gaussian smoothing, t vs x1')
            #     plt.show()
                
                
            #     df1 = pd.DataFrame({"x2":X2.flatten(),
            #             "t":T.flatten(),
            #           "Q_diff":Q_diff.flatten()})
        
            #     data1 = pd.pivot_table(df1.copy(), values='Q_diff', index='x2', columns='t')
            #     # from scipy.ndimage import gaussian_filter
            #     # data_norm1 = gaussian_filter(data1.fillna(0), sigma=5, mode='constant')
            #     # data_norm1[data_norm1==0]=None
            #     data_norm1 = data1.copy()
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm1)
            #     plt.title(f'Q_diff, no gaussian smoothing, t vs x2')
            #     plt.show()
                
            #     data_norm11 = gaussian_filter(data1.fillna(0).copy(), sigma=3, mode='constant')
            #     data_norm11[data_norm11==0]=None
            #     # data_norm1 = data1
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm11)
            #     plt.title(f'Q_diff, gaussian smoothing, t vs x2')
            #     plt.show()
            
            # try:
            #     stopping_reg = Q_diff.copy() # Q0-Q1
            #     stopping_reg[stopping_reg>=0]=1 
            #     stopping_reg[stopping_reg<0]=None
            #     df2 = pd.DataFrame({"x1":X1.flatten(),
            #             "x2":X2.flatten(),
            #           "stopping_reg":stopping_reg.flatten()})
        
            #     data2 = pd.pivot_table(df2.copy(), values='stopping_reg', index='x1', columns='x2')
            #     from scipy.ndimage import gaussian_filter
            #     data_norm1 = gaussian_filter(data2.fillna(0), sigma=5, mode='constant')
            #     data_norm1[data_norm1==0]=None
            #     data_norm2 = data2.copy()
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm2)
            #     plt.title(f'stopping_reg, no gaussian smoothing, x2 vs x1')
            #     plt.show()
        
            #     df3 = pd.DataFrame({"x2":X2.flatten(),
            #             "t":T.flatten(),
            #           "stopping_reg":stopping_reg.flatten()})
        
            #     data3 = pd.pivot_table(df3.copy(), values='stopping_reg', index='x2', columns='t')
            #     # from scipy.ndimage import gaussian_filter
            #     # data_norm1 = gaussian_filter(data1.fillna(0), sigma=5, mode='constant')
            #     # data_norm1[data_norm1==0]=None
            #     data_norm3 = data3.copy()
            #     plt.figure(figsize=(15,10))
            #     sns.heatmap(data_norm3)
            #     plt.title(f'stopping_reg, no gaussian smoothing, t vs x2')
            #     plt.show()
            # except:
            #     print(f'Cannot plot empty stopping region')
            # # try:
            
            # print(f'pr_auc: {precision_recall_curve((test_memory["action_mem"]-1)**2, (st_imitation-1)**2, average="weighted")}')
            # print()
            # print(f'stop=0, cont=1, average=binary,pos_label=0')
            # print(f'accuracy: {accuracy_score(test_memory["action_mem"], st_imitation)}')
            # print(f'roc_auc: {roc_auc_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)}')
            # print(f'precision: {precision_score(test_memory["action_mem"], st_imitation, average="binary",pos_label=0)}')
            # print(f'recall: {recall_score(test_memory["action_mem"], st_imitation, average="binary", pos_label=0)}')
            # print(f'f1: {f1_score(test_memory["action_mem"], st_imitation, pos_label=0, average="binary")}')
            # print(f'pr_auc: {average_precision_score(test_memory["action_mem"], st_imitation, pos_label=0)}')
            # self.accuracy = accuracy_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)
            # self.roc_auc = roc_auc_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2, average="weighted")
            self.f1 = f1_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)
            # self.precision = precision_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2, average="weighted")
            # self.recall = recall_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2, average="weighted")
            self.pr_auc = average_precision_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)
            self.balanced_accuracy = balanced_accuracy_score((test_memory["action_mem"]-1)**2, (st_imitation-1)**2)
            # except:
            #     print(f'Something is wrong witht the accuracy')
            return self.mtte, self.memr, self.f1, self.pr_auc, self.balanced_accuracy
            
    def train(self, mem, batch_size=250, n_epoches=5000, verbose=1):
        memory = mem.copy()
        memory['confidence_score'] = np.ones_like(memory['action_mem'])
        EPOCH_BALANCED_ACCURACY = []
        EPOCH_TRAIN_BALANCED_ACCURACY = []
        best_crossval_score = -10000000
        mem_keys = list(memory.keys())
        unique_paths = np.unique(memory['path_ids'])
        train_path_ids = random.sample(list(unique_paths), round(len(unique_paths)*0.70)) 
        train_path_ids = [int(i) for i in train_path_ids]
        val_path_ids = list(set(unique_paths) - set(train_path_ids))
        val_path_ids = [int(i) for i in val_path_ids]
        all_ids = np.arange(len(memory['path_ids']))
        mask_train = np.isin(memory['path_ids'], train_path_ids)
        mask_val = np.isin(memory['path_ids'], val_path_ids)
        train_index = all_ids[mask_train]
        val_index = all_ids[mask_val]
       
        train_batch = [np.array([memory[mem_keys[l]][m] for m in train_index]) for l in range(len(mem_keys))]
        
        orig_train_batch = dict(zip(mem_keys, train_batch))
        if self.oversampling=='SMOTE' or self.oversampling=='CS-SMOTE':
            smote_fit_batch = dict(zip(mem_keys, train_batch))    
            if self.oversampling=='CS-SMOTE':
                cf=True
            else:
                cf=False
            train_batch = self.smote(smote_fit_batch.copy(),confidence_score=cf)
        else:
            train_batch = dict(zip(mem_keys, train_batch))
        self.mean = np.array(orig_train_batch['state_mem']).mean(axis=0).copy()
        self.std = np.array(orig_train_batch['state_mem']).std(axis=0).copy()
        # self.mean = 0
        # self.std = 1
        
        for epoch in np.arange(n_epoches):
            self.epoch = epoch
            
            val_mtte = None
            val_memr = None
            val_f1 = None 
            val_pr_auc = None 
            val_balanced_accuracy = None
            epoch_losses = []
            
            train_ids = random_batch_split(np.arange(len(train_batch['time_ids'])), batch_size, stratify=train_batch['action_mem'])
            mem_keys = list(memory.keys())
            ids_batch = np.random.randint(len(train_batch[mem_keys[0]]), size=batch_size)
            for b_id, ids_batch in enumerate(train_ids):
                path_ids = np.array([train_batch['path_ids'][i] for i in ids_batch])
                time_ids = np.array([train_batch['time_ids'][i] for i in ids_batch])
                cs = torch.from_numpy(np.array([train_batch['confidence_score'][i] for i in ids_batch]))
                batch = [torch.from_numpy(np.array([train_batch[mem_keys[i]][j] for j in ids_batch])) for i in range(len(mem_keys))]
                batch[0] = (batch[0]-self.mean)/self.std
                batch[1] = (batch[1]-self.mean)/self.std
                
                self.q_net.train()
                if self.approx_g:
                    self.g_net.train()
                    y_hist = []
                    y_hist_prev = []
                    y_next_hist = []
                    for k in range(len(path_ids)):
                        mask = (memory['path_ids']==path_ids[k]) & (memory['time_ids']<=time_ids[k])
                        # mask_next = (memory['path_ids']==path_ids[k]) & (memory['time_ids']<=time_ids[k]+1)
                        # st_hist = (torch.from_numpy(np.array(memory['state_mem'])[mask])-self.mean)/self.std
                        # st_next_hist = (torch.from_numpy(np.array(memory['next_state_mem'])[mask_next][-1,:])-self.mean)/self.std
                        time_ids_hist = np.array(memory['time_ids'])[mask]
                        y_temp = [0]
                        y_prev_temp = [0]
                        y_next_temp = []
                        for time_id in time_ids_hist:
                            new_mask = (memory['path_ids']==path_ids[k]) & (memory['time_ids']<=time_id)
                            new_mask_next = (memory['path_ids']==path_ids[k]) & (memory['time_ids']<=time_id+1)
                            st_hist = (torch.from_numpy(np.array(memory['state_mem'])[new_mask])-self.mean)/self.std
                            st_next_hist = (torch.from_numpy(np.array(memory['next_state_mem'])[new_mask_next][-1,:])-self.mean)/self.std
                            with torch.no_grad():
                                self.g_net.eval()
                                # ext_state = torch.hstack([st_hist, torch.from_numpy(np.array(y_temp)).reshape(-1,1)])
                                Y_approx_all = self.g_net(st_hist.to(self.device)).cpu().detach().numpy().reshape(-1,1)
                                if Y_approx_all.shape[0]>0:
                                    Y_approx_all[0,:] = 0
                                Y_approx = (Y_approx_all*(self.gamma**(np.arange(Y_approx_all.shape[0]))).reshape(-1,1)).sum()*self.dt
                                y_temp.append(Y_approx)
            
                                Y_approx_prev = (Y_approx_all[:-1]*(self.gamma**(np.arange(Y_approx_all[:-1].shape[0]))).reshape(-1,1)).sum()*self.dt
                                y_prev_temp.append(Y_approx_prev)
                                # y_hist_prev.append(Y_approx_prev)
                                
                                # ext_st_next = torch.hstack([st_next_hist.reshape(1,-1), torch.from_numpy(np.array(y_temp[-1])).reshape(-1,1)])
                                Y_approx_next_all = self.g_net(st_next_hist.to(self.device)).cpu().detach().numpy().reshape(-1,1)
                                if Y_approx_next_all.shape[0]==1:
                                    Y_approx_all[0,:] = 0
                                Y_approx_next = (Y_approx_next_all*(self.gamma**(Y_approx_all.shape[0]+1))).sum()*self.dt + Y_approx
                                y_next_temp.append(Y_approx_next)
                        
                        y_hist.append(y_temp[-1])
                        y_hist_prev.append(y_prev_temp[-1])
                        y_next_hist.append(y_next_temp[-1])
                        

                    # print(f'y_hist len: {len(y_hist)}')
                    # print(f'y_next_hist len: {len(y_next_hist)}')
                    # print(f'state_mem.shape: {batch[0].shape}')
                    # print(f'next_state_mem.shape: {batch[1].shape}')
                    batch.append(torch.Tensor(np.array(y_hist).reshape(-1,1)))
                    batch.append(torch.Tensor(np.array(y_next_hist).reshape(-1,1)))
                    batch.append(torch.Tensor(np.array(y_hist_prev).reshape(-1,1)))
                if self.approx_g and self.oversampling in ['LSMOTE', 'CS-LSMOTE']:
                    # if self.oversampling=='SMOTE' or self.oversampling=='CS-SMOTE':
                    tmp_batch = batch
                    new_mem_keys = mem_keys + ['y', 'y_next', 'y_prev']
                    smote_fit_batch = dict(zip(new_mem_keys, tmp_batch))    
                    if self.oversampling=='CS-LSMOTE':
                        cf=True
                    else:
                        cf=False
                    # print(f'N stop actions in a batch: {(smote_fit_batch['action_mem']==0).sum()}')
                    tmp_batch = self.smote(smote_fit_batch.copy(),confidence_score=cf)
                    batch = [torch.from_numpy(tmp_batch[new_mem_keys[i]]) for i in range(len(new_mem_keys))]
                
                losses = self.iq_update(batch, epoch, cs)
                epoch_losses.append(losses)
                
                
                    
                if self.approx_g:
                    if b_id % 1==0:
                        self.g_net.train()
                        g_loss = self.g_update(batch, b_id)
                    
                self.epsilon = self.epsilon*0.9999
            if self.epoch % 1 == 0:
                if self.oversampling=='CS-LSMOTE':
                    self.cs = self.cs*self.cs_decay
                    print(f'New confidence score: {self.cs}')
                
                if self.oversampling=='CS-SMOTE':
                    # train_batch['confidence_score'][] = train_batch['confidence_score']*0.9
                    train_batch['confidence_score'] = np.where(train_batch['confidence_score'] < 1, train_batch['confidence_score']*self.cs_decay, train_batch['confidence_score'])
                    print(f'New confidence score: {train_batch['confidence_score'][train_batch['confidence_score']!=1][0]}')
                
                self.q_net.eval()
                with torch.no_grad():
                    path_ids_val = np.array([memory['path_ids'][i] for i in val_index])
                    time_ids_val = np.array([memory['time_ids'][i] for i in val_index])
                    
                    if self.approx_g:
                        self.g_net.eval()
                        batch_val = [np.array([memory[mem_keys[m]][n] for n in val_index]) for m in range(len(mem_keys))]
                    
                        # mask = [(memory['path_ids']==path_ids_val[k]) & (memory['time_ids']<=time_ids_val[k]) for k in range(len(path_ids_val))]
                        
                        # batch_val = []
                        # for m in range(len(mem_keys)):
                        #     val_list = []   
                            
                        #     for p in range(len(path_ids_val)):
                        #         if mem_keys[m]=='path_ids':
                        #             val_list.append(np.repeat(p, len(memory[mem_keys[m]][mask[p]]))) 
                        #         else:
                        #             val_list.append(np.array(memory[mem_keys[m]][mask[p]])) 
                        #     batch_val.append(np.concatenate(val_list))
    
                    else: 
                        batch_val = [np.array([memory[mem_keys[m]][n] for n in val_index]) for m in range(len(mem_keys))]
                    batch_val = dict(zip(mem_keys, batch_val))
                    # print(f'batch_val["path_ids"]: {batch_val["path_ids"]}')
                    # print(f'batch_val["time_ids"]: {batch_val["time_ids"]}')
                    fold_mtte, fold_memr, fold_f1, fold_pr_auc, fold_balanced_accuracy = self.test(test_memory=batch_val, validation=True) 
                    train_mtte, train_memr, train_f1, train_pr_auc, train_balanced_accuracy  = self.test(test_memory=orig_train_batch, validation=True) 
                    
                    val_mtte = fold_mtte.copy()
                    val_memr = fold_memr.copy()
                    val_f1 = fold_f1.copy()
                    val_pr_auc = fold_pr_auc.copy()
                    val_balanced_accuracy = fold_balanced_accuracy.copy()
                    epoch_balanced_acc = val_balanced_accuracy.copy()
                    
                    if self.approx_g:
                        self.g_scheduler.step(fold_balanced_accuracy)
                    self.scheduler.step(fold_balanced_accuracy)
                    if self.approx_dynamics:
                        self.env_scheduler.step(fold_balanced_accuracy)
                    EPOCH_BALANCED_ACCURACY.append(fold_balanced_accuracy.copy())
                    EPOCH_TRAIN_BALANCED_ACCURACY.append(train_balanced_accuracy.copy())
                    # print(f'EPOCH: {epoch}, # cross validation splits: {self.cross_val_splits}')
                    # print(f'AVG MTTE: {round(np.mean(val_mtte),4)}, AVG MEMER {round(np.mean(val_memr),4)}, AVG F1 {round(np.mean(val_f1),4)}, AVG PR_AUC {round(np.mean(val_pr_auc),4)}, AVG BALANCED_ACC {round(np.mean(val_balanced_accuracy),4)}')
                    
                    print("------------------------------------")                
                    print(f'TRAIN AVG MTTE: {round(train_mtte,4)}, AVG MEMER {round(train_memr,4)}, AVG F1 {round(train_f1,4)}, AVG PR_AUC {round(train_pr_auc,4)}, AVG BALANCED_ACC {round(train_balanced_accuracy,4)}')
                    print(f'VAL AVG MTTE: {round(val_mtte,4)}, AVG MEMER {round(val_memr,4)}, AVG F1 {round(val_f1,4)}, AVG PR_AUC {round(val_pr_auc,4)}, AVG BALANCED_ACC {round(val_balanced_accuracy,4)}')
                    
                    if fold_balanced_accuracy>best_crossval_score:
                            print(f'Saving best model checkpoint with balanced acc: {fold_balanced_accuracy}')
                            best_crossval_score = fold_balanced_accuracy.copy()
                            self.best_q_net = deepcopy(self.q_net.state_dict())
                            # torch.save(self.q_net.state_dict(), 'best_q_net.pt')
                            if self.approx_g:
                                self.best_g_net = deepcopy(self.g_net.state_dict())
                                # torch.save(self.g_net .state_dict(), 'best_g_net.pt')
                            
                    val_mtte = None
                    val_memr = None
                    val_f1 = None 
                    val_pr_auc = None 
                    val_balanced_accuracy = None
                    
            if epoch % verbose==0:
                # epoch_avg_loss = es
                if self.approx_g:
                    print(f'{epoch} epoch, loss: {losses}, g_loss: {g_loss}')
                else:
                    print(f'{epoch} epoch, loss: {losses}')


                # for param_group in self.critic_optimizer.param_groups:
                #     print(f'NEW LR: {param_group["lr"]}')
                #     print(f'NEW EPSILON: {self.epsilon}')
                    
           
            
        plt.figure()
        plt.plot(EPOCH_BALANCED_ACCURACY, color='green', label='Validation balanced accuracy')
        plt.plot(EPOCH_TRAIN_BALANCED_ACCURACY, color='red', label='Training balanced accuracy')
        plt.title(f'Balanced accuracy dynamics')
        plt.xlabel('Epoch')
        plt.ylabel('Balaned Acuracy')
        plt.legend()
        plt.show()
        self.best_crossval_score = best_crossval_score
        
    
        
    def smote(self, data_memory, confidence_score=False, ):
        dat = data_memory.copy()
        
        if self.approx_g:
            try:
                tmp_dat = [dat[i].cpu().detach().numpy() for i in list(dat.keys())]
                dat = dict(zip(list(dat.keys()),tmp_dat))
            except:
                print(f'Elements of dict are already of type np.ndarray')
            y_idx = dat['state_mem'].shape[1]*2 + dat['y'].shape[1]*3
            x_size = dat['state_mem'].shape[1]
            y_size = dat['y'].shape[1] 
        else:
            y_idx = dat['state_mem'].shape[1]*2
            x_size = dat['state_mem'].shape[1]
        n_stop_act = max(sum(dat['action_mem'].astype(int)==0)-1,0)
        # print(f'n_stop_act: {n_stop_act}')
        if n_stop_act<=0:
            return dat
        else:
            if self.approx_g:
                df = pd.DataFrame(np.hstack([dat['state_mem'],
                    dat['next_state_mem'],
                    dat['y'].reshape(-1,1), dat['y_next'].reshape(-1,1), dat['y_prev'].reshape(-1,1),
                    dat['action_mem'].astype(int).reshape(-1,1), 
                    
                    ]))
                self.sm = SMOTE(sampling_strategy='minority',k_neighbors=min(5, n_stop_act))
            else:
                df = pd.DataFrame(np.hstack([dat['state_mem'],
                        dat['next_state_mem'],
                        dat['action_mem'].astype(int).reshape(-1,1),
                        ]))
                self.sm = SMOTE(sampling_strategy='minority',k_neighbors=min(12, n_stop_act))
            
            df_res, y_resampled = self.sm.fit_resample(df, df.iloc[:,y_idx])
            df_res.loc[y_resampled==0,x_size:x_size*2-1] = df_res.loc[y_resampled==0,:x_size-1].values
            if self.approx_g:
                df_res.loc[y_resampled==0,x_size*2:x_size*2+y_size*3-1] = df_res.loc[y_resampled==0,x_size:x_size+y_size*3-1].values
                
            dat['state_mem'] = df_res.iloc[:,:x_size].values
            dat['next_state_mem'] = df_res.iloc[:,x_size:x_size*2].values
            if self.approx_g:
                dat['y'] = df_res.iloc[:,x_size*2:x_size*2+y_size].values
                dat['y_next'] = df_res.iloc[:,x_size*2+y_size:x_size*2+y_size*2].values
                dat['y_prev'] = df_res.iloc[:,x_size*2+y_size*2:x_size*2+y_size*3].values
            
            dat['action_mem'] = y_resampled.values
            dat['confidence_score'] = np.ones_like(df_res.iloc[:,0])
            
            dat['path_ids'] = np.ones_like(df_res.iloc[:,0])
            dat['time_ids'] =np.ones_like(df_res.iloc[:,0])
            dat['done_mem'] =np.zeros_like(df_res.iloc[:,0])
            dat['done_mem'][dat['action_mem']==0] = 1
            dat['init_states_mem'] =df_res.iloc[:,:x_size].values
            dat['reward_mem'] = np.zeros_like(dat['done_mem'])

            if confidence_score:
                dat['confidence_score'][~df.index] = self.cs
            st = dat['state_mem']
            act = dat['action_mem']
            # plt.figure(figsize=(25,25), facecolor="w")
            
            
            # plt.scatter(st[~df.index][act[~df.index]==0,0],st[~df.index][act[~df.index]==0,1],
            #             s=5, c='red', 
            #             label='artificial stopping decisions')
            # plt.scatter(data_memory['state_mem'][data_memory['action_mem']==0,0],
            #             data_memory['state_mem'][data_memory['action_mem']==0,1],s=50, 
            #             c='green',label='true stopping decisions')
            # plt.legend(fontsize=30)
            # max_coord = 1.5 #st[~df.index][act[~df.index]==0,:1].max().max()+0.25
            # plt.axis(xmin=-max_coord,
            #           xmax=max_coord,
            #           ymin=-max_coord,
            #           ymax=max_coord)
            
            # plt.xlabel('x1',fontsize=30)
            # plt.ylabel('x2',fontsize=30)
            # plt.rcParams.update({'font.size': 30, 'axes.facecolor': 'white'})
            # plt.grid(color='grey')
            # plt.show()
            # print(f'dat["state_mem"].shape: {dat["state_mem"].shape}')
            # print(f'dat["action_mem"].shape: {dat["action_mem"].shape}')
            # print(f'dat["y"].shape: {dat["y"].shape}')
            # print(f'dat["y_next"].shape: {dat["y_next"].shape}')
            # print(f'dat["y_prev"].shape: {dat["y_prev"].shape}')
            
            return dat

    
    def distance_function(self, x):
        return x - 1/2 * x**2

    def iq_loss(self, agent, current_Q, current_v, next_v,
                batch, cs, loss_method='value_expert', grad_pen=False,
                lambda_gp=0.1, regularize=True):
        div = self.divergence
        obs, next_obs, action, done = batch[:4]
        done = (action-1)**2
        if self.approx_g:
            y,y_next,y_prev = batch[-3:]
            obs = torch.hstack([obs, y]).clone()
            next_obs = torch.hstack([next_obs, y_next]).clone()

        loss_dict = {}
        # keep track of value of initial states
        v0 = self.getV(obs, evaluate=False).mean()
        loss_dict['v0'] = v0.item()

        #  calculate 1st term for IQ loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        # y = (1-done)*gamma*next_v+ 1*done
        rew = ((1 - done) * self.gamma * next_v.T.cpu()).T# + 1*done).T
        # current_Q_s_a = torch.gather(current_Q.cpu(), dim=-1, index=torch.tensor(action).to(torch.int64).cpu().reshape(-1, 1))
        current_Q_s_a = torch.gather(current_Q.cpu(), dim=-1, index=action.clone().detach().to(torch.int64).cpu().reshape(-1, 1))
        # if self.approx_g:
        #     reward = (current_Q_s_a.cpu().clone() - rew.cpu().clone()) - (1-done.reshape(-1,1))*y_prev
        # else:
        reward = (current_Q_s_a.cpu().clone() - rew.cpu().clone())


        with torch.no_grad():
            # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
            if div == "hellinger":
                phi_grad = 1/(1+reward)**2
            elif div == "kl":
                # originyal dual form for kl divergence (sub optimal)
                phi_grad = torch.exp(-reward-1)
            elif div == "kl2":
                # biased dual form for kl divergence
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif div == "kl_fix":
                # our proposed unbiased form for fixing kl divergence
                phi_grad = torch.exp(-reward)
            elif div == "js":
                # jensen–shannon
                phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
            else:
                phi_grad = 1
        loss = -(cs*phi_grad * reward).mean()
        # loss = -(self.distance_function(reward)).mean()
        loss_dict['softq_loss'] = loss.item()

        # calculate 2nd term for IQ loss, we show different sampling strategies
        if loss_method == "value_expert":
            # sample using only expert states (works offline)
            # E_(ρ)[Q(s,a) - γV(s')]

            # if self.approx_g:
            #     value_loss = (current_v.cpu() - rew.cpu()- (1-done.reshape(-1,1))*y_prev).mean()
            # else:
            value_loss = (cs*(current_v.cpu() - rew.cpu())).mean()
            
            # value_loss = (current_v - y).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif loss_method == "value":
            # sample using expert and policy states (works online)
            # E_(ρ)[V(s) - γV(s')]
            # value_loss = (current_v.cpu() - rew.cpu() - (1-done.reshape(-1,1))*y_prev).mean()
            value_loss = (current_v.cpu() - rew.cpu()).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif loss_method == "v0":
            # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.gamma) * v0.cpu()
            loss += v0_loss
            loss_dict['v0_loss'] = v0_loss.item()

        else:
            raise ValueError(f'This sampling method is not implemented: {loss_method}')


        if div == "chi":
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
            # y = (1 - done) * gamma * next_v
            y = ((1 - done) * self.gamma * next_v.T.cpu()).T
            reward = current_Q_s_a.cpu() - rew.cpu()
            chi2_loss = 1/(4 * self.epsilon) * (reward**2).mean()
            loss += chi2_loss
            loss_dict['chi2_loss'] = chi2_loss.item()

        if regularize:
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
            # y = (1 - done) * gamma * next_v
            rew = ((1 - done) * self.gamma * next_v.T.cpu()).T
            # print(f'next_v: {next_v}, {next_v.shape}')
            # print(f'rew: {rew}, {rew.shape}')
            # print(f'current_Q: {current_Q}, {current_Q.shape}')
            # print(f'current_Q_s_a: {current_Q_s_a}, {current_Q_s_a.shape}')
            # if self.approx_g:
            #     reward = current_Q.cpu() - rew.cpu() - (1-done.reshape(-1,1))*y_prev
            # else:
            reward = cs*(current_Q_s_a.cpu() - rew.cpu()) #- (1-done.reshape(-1,1))*y_prev
                
            chi2_loss = 1/(4 * self.epsilon) * (reward**2).mean()
            loss += chi2_loss
            loss_dict['regularize_loss'] = chi2_loss.item()

        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict

    def plot_st_reg_bm(self, test_memory, conservative, discretiser, out_thresh, name):
        if conservative: 
            unique_ints, k_outs = discretiser.ints
        import os
        try:
            os.mkdir('./out') 
        except:
            pass
        state_mem = test_memory['state_mem']
        for j in numpy.arange(state_mem[:,2].min(),state_mem[:,2].max(), 0.01):
            # print(f'TIME: {round(j,3)}')
            X1_test = numpy.arange(state_mem[:,0].min(),state_mem[:,0].max(), 0.05)
            X2_test = numpy.arange(state_mem[:,1].min(),state_mem[:,1].max(), 0.05)
            TIME_test = numpy.repeat(j, X1_test.shape[0])
            mesh = numpy.array(numpy.meshgrid(X1_test, X2_test,TIME_test)).T.reshape(-1,3)
            # print(mesh.shape)
            mesh = torch.Tensor(mesh)
            if conservative: 
                state_discretised = discretiser.transform(mesh.cpu().detach().numpy().copy())
                out_degree_test = np.zeros_like(state_discretised[:,0])
                for i in range(len(k_outs)):
                    out_degree_test[(state_discretised==unique_ints[i]).all(axis=1)] =  k_outs[i]
        
            mesh_norm = (mesh.clone() - self.mean)/self.std
            
            with torch.no_grad():
                irl_q0 = self.infer_q(state=mesh_norm, action=numpy.zeros_like(mesh[:,0]))
                irl_q1 = self.infer_q(state=mesh_norm, action=numpy.ones_like(mesh[:,0]))
                irl_curr_v = self.getV(mesh_norm, evaluate=True)
                if conservative: 
                    irl_q0[out_degree_test<out_thresh] = 0.0
                    irl_q1[out_degree_test<out_thresh] = 0.0
        
                if  self.approx_dynamics:
                    irl_next_s = self.infer_next_s(state=mesh_norm)
                    irl_V = self.getV(irl_next_s, evaluate=True)
                    irl_G = irl_q0
                    irl_g = irl_q1 - self.q_net.gamma*irl_V
                    
                    irl_next_q0 = self.infer_q(state=irl_next_s, action=numpy.zeros_like(mesh[:,0]))
                    irl_next_q1 = self.infer_q(state=irl_next_s, action=numpy.ones_like(mesh[:,0]))
                    
                    
                    g_cases = [irl_q1 > irl_q0]
                    G_cases = [irl_q1 <= irl_q0]
                    next_td = torch.zeros_like(irl_curr_v)
                    next_td[G_cases] = irl_G[G_cases]
                    next_td[g_cases] = (irl_g + torch.maximum(irl_next_q0, irl_next_q1))[g_cases]
                    # irl_q1 =irl_g + iq_agent.q_net.gamma*irl_V
                    # df0 = pandas.DataFrame({"x1":mesh[:,0],
                    #                 "x2":mesh[:,1],
                    #               "Q":irl_G.reshape(-1)})
    
                    # data0 = pandas.pivot_table(df0, values='Q', index='x1', columns='x2')
                    
                    # seaborn.heatmap(data0) 
                    # pyplot.title(f't={round(j,3)}, G')
                    # pyplot.show()
                    
                    # df0 = pandas.DataFrame({"x1":mesh[:,0],
                    #                 "x2":mesh[:,1],
                    #               "Q":irl_g.reshape(-1)})
    
                    # data0 = pandas.pivot_table(df0, values='Q', index='x1', columns='x2')
                    
                    # seaborn.heatmap(data0) 
                    # pyplot.title(f't={round(j,3)}, g')
                    # pyplot.show()
                    
            with open('./out/states'+str(round(j,3))+'.npy', 'wb') as f:
                numpy.save(f, mesh)
            with open('./out/irl_q0'+str(round(j,3))+'.npy', 'wb') as f:
                numpy.save(f, irl_q0)    
            with open('./out/irl_q1'+str(round(j,3))+'.npy', 'wb') as f:
                numpy.save(f, irl_q1)
            with open('./out/irl_curr_v'+str(round(j,3))+'.npy', 'wb') as f:
                numpy.save(f, irl_curr_v)
        
       
    
        # N=10
        def sort_points(xy: numpy.ndarray) -> numpy.ndarray:
            # normalize data  [-1, 1]
            xy_sort = numpy.empty_like(xy)
            xy_sort[:, 0] = 2 * (xy[:, 0] - numpy.min(xy[:, 0]))/(numpy.max(xy[:, 0] - numpy.min(xy[:, 0]))) - 1
            xy_sort[:, 1] = 2* (xy[:, 1] - numpy.min(xy[:, 1])) / (numpy.max(xy[:, 1] - numpy.min(xy[:, 1]))) - 1
    
            # get sort result
            sort_array = numpy.arctan2(xy_sort[:, 0], xy_sort[:, 1])
            sort_result = numpy.argsort(sort_array)
    
            # apply sort result
            return xy[sort_result]
    
        
        fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(30,30))
        # fig.update_layout()
        # fig.layout.update(plot_bgcolor = "white")
        plt.rcParams['axes.facecolor'] = 'white'
        # ax1.scatter(X_resampled[:,1][sel], X_resampled[:,2][sel], s=75)
        ax2.set_xlabel('x2')
        ax2.set_ylabel('x1') 
        # ax1.grid(color='grey')
        # ax1.set_title('All stopping coordinates', size=65)
        cmaps = sns.color_palette("magma", 10)
        from matplotlib.colors import LinearSegmentedColormap
        my_cmap = LinearSegmentedColormap.from_list('my_cmap', cmaps, N=10)
    
        sns.set(font_scale = 4)
        for i in range(0,9):
            try:
                if i==0:
                    with open('out/states0.'+str(i)+'1.npy', 'rb') as f:
                        states = numpy.load(f)
                    with open('out/irl_q00.'+str(i)+'1.npy', 'rb') as f:
                        q_0 = numpy.load(f).reshape(-1)
                    with open('out/irl_q10.'+str(i)+'1.npy', 'rb') as f:
                        q_1 = numpy.load(f).reshape(-1)
                        
                else:
                    with open('out/states0.'+str(i)+'1.npy', 'rb') as f:
                        states = numpy.load(f)
                    with open('out/irl_q00.'+str(i)+'1.npy', 'rb') as f:
                        q_0 = numpy.load(f).reshape(-1)
                    with open('out/irl_q10.'+str(i)+'1.npy', 'rb') as f:
                        q_1 = numpy.load(f).reshape(-1)
                    
                    
                import pandas as pd
                q_0_norm = q_0
                df0 = pd.DataFrame({"x1":states[:,0],
                                "x2":states[:,1],
                              "Q":q_0_norm})
                
                data0 = pd.pivot_table(df0, values='Q', index='x1', columns='x2')
                data_norm0 = data0
                data_nan0 = data_norm0
        
                q_1_norm = q_1
                df1 = pd.DataFrame({"x1":states[:,0],
                                "x2":states[:,1],
                              "Q":q_1_norm})
                data1 = pd.pivot_table(df1, values='Q', index='x1', columns='x2')
                data_norm1 = data1
        
        
        
                data_diff = data_norm0-data_norm1
                
        
                n = data_diff.copy()
                n[n>=0.0]=1
                n[n<0.0]=None
                # seaborn.heatmap(n)
                # pyplot.show()
            #     print(i)
                # if i == 0 or i == 9:
                #     alpha = 0.5
                # else:
                #     alpha = 0.5
                
                nn = n.copy()
                from scipy.ndimage.morphology import binary_dilation
                k = numpy.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1
                nn[~numpy.isnan(nn)] = 1
                nn[nn != 1] = 0
                nn = nn.astype(int)
                boundary = binary_dilation(nn==0, k) & nn
                boundary = boundary.astype(int)
                boundary[boundary==0] = numpy.nan
                # sns.scatterplot(boundary, ax=ax)
                newb = boundary.unstack().reset_index()
                newb.columns = ['x2','x1','vals']
                st_points = newb[~numpy.isnan(newb['vals'])]
                coords = st_points[['x1','x2']].values
                # print(i)
                scoords = sort_points(coords)
                tck,u = interpolate.splprep(scoords.transpose(), s=0)
                unew = numpy.arange(0, 1.01, 0.01)
                out = interpolate.splev(unew, tck)
                ax2.plot(out[0], out[1], color=cmaps[i], linewidth=5)
                ax2.grid(color='grey')
                # ax2.set_title(f'Stopping region', size=65)
            except:
                pass
        cax = fig.add_axes([0.92, 0.12, 0.05, 0.75])
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=my_cmap,
                                        orientation='vertical')
        ax2.set_xlabel('x1', fontsize=30)
        ax2.set_ylabel('x2', fontsize=30)
        for item in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(30)
        ax2.scatter(test_memory['state_mem'][test_memory['action_mem']==0,0],
                    test_memory['state_mem'][test_memory['action_mem']==0,1],
                    s=100, c='green', label='True stopping decisions')
        ax2.set_title(name)
        # ax2.grid(color='grey')
        # ax1.axis(xmin=-1.5,
        #           xmax=1.5,
        #           ymin=-1.5,
        #           ymax=1.5)
        ax2.axis(xmin=-1.5,
                  xmax=1.5,
                  ymin=-1.5,
                  ymax=1.5)
        cbar.set_ticks([i+(0.1/2) for i in np.arange(0,1,0.1)])
        cbar.set_ticklabels(['0.'+str(i)+'1' for i in range(10)])
        cbar.ax.tick_params(labelsize=30) 
        cbar.set_label('Time',  size=30, loc='center', labelpad=40)
        plt.show()
        
# sns.heatmap(data0)
# plt.title('Q0')
# plt.show()

# sns.heatmap(data1)
# plt.title('Q1')
# plt.show()

    # import os
    # import glob
    
    # files = glob.glob('./out/*')
    # for f in files:
    #     os.remove(f)
    

    
 # {'state_mem': STATE_MEM_SIM, 
 #        'next_state_mem': STATE2_MEM_SIM, 
 #        'action_mem': ACTION_MEM_SIM, 
 #        'done_mem': DONE_MEM_SIM,
 #        'init_states_mem': INIT_STATES, 
 #        'reward_mem': REWARD_MEM,
 #        'path_ids':PATH_IDS,
 #        'time_ids':TIME_IDS}
    
    
def plot_st_reg_car(iq_agent, test_memory):
    
    state_mem = test_memory['state_mem']
    X1_test = numpy.arange(state_mem[:,0].min(),state_mem[:,0].max(), 50)
    X2_test = numpy.arange(state_mem[:,1].min(),state_mem[:,1].max(), 50)
    mesh = numpy.array(numpy.meshgrid(X1_test, X2_test)).T.reshape(-1,2)
    mesh = torch.Tensor(mesh)
    mesh_norm = (mesh.clone()- iq_agent.mean)/iq_agent.std
        
    with torch.no_grad():
        irl_q0 = iq_agent.infer_q(state=mesh_norm, action=numpy.zeros_like(mesh[:,0]))
        irl_q1 = iq_agent.infer_q(state=mesh_norm, action=numpy.ones_like(mesh[:,0]))
        irl_curr_v = iq_agent.getV(mesh_norm, evaluate=True)
        if  iq_agent.approx_dynamics:
            irl_next_s = iq_agent.infer_next_s(state=mesh_norm)
            irl_V = iq_agent.getV(irl_next_s, evaluate=True)
            irl_G = irl_q0
            irl_g = irl_q1 - iq_agent.q_net.gamma*irl_V
            
            irl_next_q0 = iq_agent.infer_q(state=irl_next_s, action=numpy.zeros_like(mesh[:,0]))
            irl_next_q1 = iq_agent.infer_q(state=irl_next_s, action=numpy.ones_like(mesh[:,0]))
            g_cases = [irl_q1 > irl_q0]
            G_cases = [irl_q1 <= irl_q0]
            next_td = torch.zeros_like(irl_curr_v)
            next_td[G_cases] = irl_G[G_cases]
            next_td[g_cases] = (irl_g + torch.maximum(irl_next_q0, irl_next_q1))[g_cases]
            irl_q1 = irl_g + iq_agent.getV(irl_next_s, evaluate=True)
        
        q_0_norm = irl_q0
        df0 = pd.DataFrame({"x1":mesh[:,0].reshape(-1),
                        "x2":mesh[:,1].reshape(-1),
                      "Q":q_0_norm.reshape(-1)})
        
        data0 = pd.pivot_table(df0, values='Q', index='x1', columns='x2')
        data_norm0 = data0
        data_nan0 = data_norm0

        q_1_norm = irl_q1
        df1 = pd.DataFrame({"x1":mesh[:,0].reshape(-1),
                        "x2":mesh[:,1].reshape(-1),
                      "Q":q_1_norm.reshape(-1)})
        data1 = pd.pivot_table(df1, values='Q', index='x1', columns='x2')
        data_norm1 = data1



        data_diff = data_norm0-data_norm1
        

        n = data_diff.copy()
        n[n>=0.0]=1
        n[n<0.0]=None
        seaborn.heatmap(n)
        pyplot.show()