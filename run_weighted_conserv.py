# %%
from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# with open('C:/Users/annak/Downloads/DO-IQS/DO-IQS/inverse_opt_stopping/iq_learn_conserv.py', 'r') as f:
    # print(f.read())
from forward_algorithms.simulate_expert_data import Simulation
from inverse_opt_stopping.iq_learn_conserv import IQ_Agent, plot_st_reg_car
from inverse_opt_stopping.iq_learn_base import IQ_Agent as IQ_Agent_base
import os
import shutil
sns.set_style("whitegrid")
# 22a- no weights, 23a - 0.001 weight
for ex in [
        # 'azure', 
            # 'nasa_turbofan', 
            # 'bmG', 
            # 'bmgG',
            # 'radial',
            'star',
            # 'CP1',
            # 'CP2',
            # 'CP3',
            
            # 'CP1M',
            # 'CP2M',
            # 'CP3M',
            # 'car'
 
            ]:
    DRIVER_ROC_AUC = []
    DRIVER_F1 = []
    DRIVER_ACCURACY = []
    DRIVER_PRECISION = []
    DRIVER_RECALL = []
 
    BM_ROC_AUC = []
    BM_F1 = []
    BM_ACCURACY = []
    BM_PRECISION = []
    BM_RECALL = []
    BM_PR_AUC = []
 
    EPOCH_BA = []
    EPOCH_MTTE = []
    EPOCH_MEMR = []
    EPOCH_LOSS = []
       
    BM_MTTE = []
    BM_MEMR = []
    BM_BALANCED_ACC = []
    import shutil
    shutil.rmtree('./out', ignore_errors=True)
 
    N_EPOCHS = 250
    CROSS_VAL_SPLITS = 2
    SEEDS = np.arange(5)
    T_DOWNSAMPLE = 1
    BATCH_SIZE = 128
    Q_LR = 0.01
    ENV_LR = 0.01
    G_LR = 0.001
    SMOTE_K = 12
    out_thresh = 0.000
    out_thresh1 = 0.005
    # out_thresh2 = 0.00005
    # out_thresh3 = 0.0005
    # out_thresh4 = 0.005
    EPS = 0.1
    conservative=True
    IS_CS = False
    dqs_only = []
    for s in SEEDS:
     
        # Symmetrical BM resultse
        if ex=='radial':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='radial', total_n=250)
        elif ex=='star': 
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='star', total_n=250)
        elif ex=='CP1':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='CP1', total_n=250)
        elif ex=='CP2':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='CP2', total_n=250)
        elif ex=='CP3':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='CP3', total_n=250)
        elif ex=='CP1M':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='CP1M', total_n=250)
        elif ex=='CP2M':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='CP2M', total_n=250)
        elif ex=='CP3M':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='CP3M', total_n=250)
        elif ex=='bmG':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='symm_bm_G', total_n=250)
        elif ex=='bmgG':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='symm_bm_gG', total_n=250)
        elif ex=='car':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='car')
        elif ex=='azure':
            bm_sym = Simulation(problem='azure')
            T_DOWNSAMPLE=10
        elif ex=='nasa_turbofan':
            bm_sym = Simulation(problem='nasa_turbofan')
            T_DOWNSAMPLE=10
        
        bm_expert_data = bm_sym.simulate_expert(episodes=250, max_path_length=100)
        bm_test_data = bm_sym.simulate_test(episodes=100, max_path_length=100)
        print(f'RANDOM SEED: {s}')
        import matplotlib
        list_of_times = list(np.unique(bm_expert_data['time_ids']))
        cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
        gradient = np.arange(0, 1, 1/len(list_of_times))
        list_of_colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(list_of_times)]
        from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
       
        # fig, ax = plt.subplots(figsize=(15,15))
        fig, ax = plt.subplots(figsize=(30,15))
     
       
        for k, ts in enumerate(np.unique(bm_expert_data['time_ids'])):
            # plt.scatter(bm_expert_data['state_mem'][bm_expert_data['action_mem']==1][:,0],
            #             bm_expert_data['state_mem'][bm_expert_data['action_mem']==1][:,1],
            #             s=5, color='green')
            # ax.scatter(bm_expert_data['state_mem'][(bm_expert_data['action_mem']==0) & (bm_expert_data['time_ids']==ts)][:,0],
            #             bm_expert_data['state_mem'][(bm_expert_data['action_mem']==0) & (bm_expert_data['time_ids']==ts)][:,1],
            #             s=15,
            #             color = list_of_colours[k]
            #             # color='red'
    
            #             )
            ax.scatter(bm_expert_data['time_ids'][(bm_expert_data['action_mem']==1) & (bm_expert_data['time_ids']==ts)],
            bm_expert_data['state_mem'][(bm_expert_data['action_mem']==1) & (bm_expert_data['time_ids']==ts)][:,0],
                        s=55,
                        # color = list_of_colours[k]
                        color='green'
    
                        )
            ax.scatter(bm_expert_data['time_ids'][(bm_expert_data['action_mem']==0) & (bm_expert_data['time_ids']==ts)],
            bm_expert_data['state_mem'][(bm_expert_data['action_mem']==0) & (bm_expert_data['time_ids']==ts)][:,0],
                        s=55,
                        # color = list_of_colours[k]
                        color='red'
    
                        )
           
           
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True)
        # ax.set_xlim((-3,3))
        # ax.set_ylim((-3,3))
       
        ax.set_ylim((-10,10))
        ax.set_xlim((0,51))
        ax.set_xlabel('t', size=60)
        ax.set_ylabel('x0', size=60)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=45)
        plt.title('CP3', size='60')
        # ax.set_aspect('equal')
       
        fig.tight_layout()
        plt.show()
        if conservative:
            from binning import MDLP_Discretizer, CART_Discretizer
            # discretizer = MDLP_Discretizer(features=np.arange(bm_expert_data['state_mem'].shape[1]))
            discretizer = CART_Discretizer()
            # 0-1-2-3-4-5-150
            discretizer.fit(bm_expert_data['state_mem'], bm_expert_data['action_mem'])
        else:
            discretizer=None
        print(f'Start training')
        
        # print('Classifier')
        # iq_class = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling=False,
        #                     q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=conservative,
        #                     q_entropy=False,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     classify=True)
        # iq_class.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_class.test(test_memory=bm_test_data, from_grid=False)
        # iq_class0_ba = iq_class.balanced_accuracy
        # iq_class0_mtte = iq_class.mtte
        # iq_class0_memr = iq_class.memr
        
        # print('Classifier with SMOTE oversampling')
        # iq_class_SMOTE = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling='SMOTE',
        #                     q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=conservative,
        #                     q_entropy=False,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     classify=True)
        # iq_class_SMOTE.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=False)
        # iq_class_SMOTE0_ba = iq_class_SMOTE.balanced_accuracy
        # iq_class_SMOTE0_mtte = iq_class_SMOTE.mtte
        # iq_class_SMOTE0_memr = iq_class_SMOTE.memr
        
        # # print('IQ-Learn')
        # iq = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling=None,
        #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0.01,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=conservative,
        #                     q_entropy=False,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     classify=False)
        # iq.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq.test(test_memory=bm_test_data, from_grid=False)
        # iq0_ba = iq.balanced_accuracy
        # iq0_mtte = iq.mtte
        # iq0_memr = iq.memr
        # print(f'w0:{iq.degree_model.fc1.bias}, w1:{iq.degree_model.fc1.weight}')
        # import torch
        # rod = np.arange(0,1,0.01)
        # with torch.no_grad():
        #     w = iq.degree_model(torch.from_numpy(rod).to(torch.float32)).cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        # plt.plot(rod, w)
        # plt.show()
       
        # IQ-Learning + oversampling SMOTE
        print('IQ-Learning + oversampling SMOTE')
        # iq_smote = IQ_Agent_base(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling='SMOTE',
        #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=False,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh
        #                     )
        # iq_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_smote.test(test_memory=bm_test_data, from_grid=False)
        
        iq_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_smote.test(test_memory=bm_test_data, from_grid=False)
        iq_smote.test(test_memory=bm_test_data, from_grid=True)
        iq_smote0_ba = iq_smote.balanced_accuracy
        iq_smote0_mtte = iq_smote.mtte
        iq_smote0_memr = iq_smote.memr
       
        # print(f'w0:{iq_smote.degree_model.fc1.weight}, w1:{iq_smote.degree_model.fc1.bias}')
        import torch
        rod = np.arange(0,1,0.01)
        with torch.no_grad():
            w = iq_smote.degree_model(torch.from_numpy(rod).to(torch.float32)).cpu().detach().numpy()
        import matplotlib.pyplot as plt
        plt.plot(rod, w)
        plt.show()
       
        print('IQ-Learning + oversampling CS-SMOTE')
       
        # iq_cs_smote = IQ_Agent_base(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE',
        #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=False,
        #                     q_entropy=False,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh
        #                     )
        # iq_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        
        iq_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=False,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        iq_cs_smote0_ba = iq_cs_smote.balanced_accuracy
        iq_cs_smote0_mtte = iq_cs_smote.mtte
        iq_cs_smote0_memr = iq_cs_smote.memr
        
        # print(f'w0:{iq_cs_smote.degree_model.fc1.weight}, w1:{iq_cs_smote.degree_model.fc1.bias}')
        import torch
        rod = np.arange(0,1,0.01)
        with torch.no_grad():
            w = iq_cs_smote.degree_model(torch.from_numpy(rod).to(torch.float32)).cpu().detach().numpy()
        import matplotlib.pyplot as plt
        plt.plot(rod, w)
        plt.show()
        # Approximating P, no oversampling
        print('Approximating P, no oversampling')
        
        # iq_p = IQ_Agent_base(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=True, oversampling=None,
        #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=False,
        #                     q_entropy=False,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh
        #                     )
        # iq_p.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_p.test(test_memory=bm_test_data, from_grid=False)
        
        iq_p = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=True, oversampling=None,
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=False,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_p.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_p.test(test_memory=bm_test_data, from_grid=False)
        iq_p.test(test_memory=bm_test_data, from_grid=True)
        iq_p0_ba = iq_p.balanced_accuracy
        iq_p0_mtte = iq_p.mtte
        iq_p0_memr = iq_p.memr
        
        # print(f'w0:{iq_p.degree_model.fc1.bias}, w1:{iq_p.degree_model.fc1.weight}')
        import torch
        rod = np.arange(0,1,0.01)
        with torch.no_grad():
            w = iq_p.degree_model(torch.from_numpy(rod).to(torch.float32)).cpu().detach().numpy()
        import matplotlib.pyplot as plt
        plt.plot(rod, w)
        plt.show()
        # iq_p.out_thresh = out_thresh1
        # iq_p.test(test_memory=bm_test_data, from_grid=False)
        # iq_p1_ba = iq_p.balanced_accuracy
        # iq_p1_mtte = iq_p.mtte
        # iq_p1_memr = iq_p.memr
           
        # Approximating P + oversampling SMOTE
        print('Approximating P + oversampling SMOTE')
        
        # iq_p_smote = IQ_Agent_base(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=True, oversampling='SMOTE',
        #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=False,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh
        #                     )
        # iq_p_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
        
        iq_p_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=True, oversampling='SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_p_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=True)
        iq_p_smote0_ba = iq_p_smote.balanced_accuracy
        iq_p_smote0_mtte = iq_p_smote.mtte
        iq_p_smote0_memr = iq_p_smote.memr
        # print(f'w0:{iq_p_smote.degree_model.fc1.bias}, w1:{iq_p_smote.degree_model.fc1.weight}')
        import torch
        rod = np.arange(0,1,0.01)
        with torch.no_grad():
            w = iq_p_smote.degree_model(torch.from_numpy(rod).to(torch.float32)).cpu().detach().numpy()
        import matplotlib.pyplot as plt
        plt.plot(rod, w)
        plt.show()
        # Approximating P + oversampling CS-SMOTE
        print('Approximating P + oversampling CS-SMOTE')
        
        # iq_p_cs_smote = IQ_Agent_base(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE',
        #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=False,
        #                     q_entropy=False,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh
        #                     )
        # iq_p_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        
        iq_p_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=False,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_p_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        iq_p_cs_smote0_ba = iq_p_cs_smote.balanced_accuracy
        iq_p_cs_smote0_mtte = iq_p_cs_smote.mtte
        iq_p_cs_smote0_memr = iq_p_cs_smote.memr
        
        
        # Approximating g and P, no oversampling
        print('Approximating g and P, no oversampling')
        # iq_diqs = IQ_Agent_base(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=True, approx_dynamics=True, oversampling=None,
        #                     q_lr=0.01, env_lr=0.01, g_lr=0.001,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=False,
        #                     q_entropy=False,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh
        #                     )
        # iq_diqs.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_diqs.test(test_memory=bm_test_data, from_grid=False)
        
        iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=True, approx_dynamics=True, oversampling=None,
                            q_lr=0.01, env_lr=0.01, g_lr=0.001,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=False,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_diqs.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_diqs.test(test_memory=bm_test_data, from_grid=False)
        iq_diqs0_ba = iq_diqs.balanced_accuracy
        iq_diqs0_mtte = iq_diqs.mtte
        iq_diqs0_memr = iq_diqs.memr
        
       
        # Approximating g and P, CS-LSMOTE oversampling
        print('Approximating g and P, CS-LSMOTE oversampling')
        # iq_diqs_cs_smote = IQ_Agent_base(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=True, approx_dynamics=True, oversampling='CS-LSMOTE',
        #                     q_lr=0.01, env_lr=0.01, g_lr=0.001,
        #                     epsilon=EPS,seed=s,
        #                     divergence='hellinger', cross_val_splits=2,
        #                     conservative=False,
        #                     q_entropy=False,
        #                     SMOTE_K=SMOTE_K,
        #                     is_cs=IS_CS,
        #                     discretiser=discretizer,
        #                     out_thresh=out_thresh
        #                     )
        # iq_diqs_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        # iq_diqs_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        
        iq_diqs_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=True, approx_dynamics=True, oversampling='CS-LSMOTE',
                            q_lr=0.01, env_lr=0.01, g_lr=0.001,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=False,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_diqs_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_diqs_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        iq_diqs_cs_smote0_ba = iq_diqs_cs_smote.balanced_accuracy
        iq_diqs_cs_smote0_mtte = iq_diqs_cs_smote.mtte
        iq_diqs_cs_smote0_memr = iq_diqs_cs_smote.memr
        
    
        
        BM_MTTE.append([
          # iq_class0_mtte,
          #             iq_class_SMOTE0_mtte,
                      iq0_mtte,
                        iq_smote0_mtte,
                        iq_cs_smote0_mtte,
                        iq_p0_mtte,
                        iq_p_smote0_mtte,
                        iq_p_cs_smote0_mtte,
                        iq_diqs0_mtte,
                        iq_diqs_cs_smote0_mtte,
          # iq_class1_mtte,
          #             iq_class_SMOTE1_mtte,
          #             iq1_mtte,
          #               iq_smote1_mtte,
          #               iq_cs_smote1_mtte,
          #               iq_p1_mtte,
          #               iq_p_smote1_mtte,
          #               iq_p_cs_smote1_mtte,
          #               iq_diqs1_mtte,
          #               iq_diqs_cs_smote1_mtte,
                                        ])
        BM_MEMR.append([
          
          # iq_class0_memr,
          #             iq_class_SMOTE0_memr,
                      iq0_memr,
                        iq_smote0_memr,
                        iq_cs_smote0_memr,
                        iq_p0_memr,
                        iq_p_smote0_memr,
                        iq_p_cs_smote0_memr,
                        iq_diqs0_memr,
                        iq_diqs_cs_smote0_memr,
          # iq_class1_memr,
                      # iq_class_SMOTE1_memr,
                      # iq1_memr,
                      #   iq_smote1_memr,
                      #   iq_cs_smote1_memr,
                      #   iq_p1_memr,
                      #   iq_p_smote1_memr,
                      #   iq_p_cs_smote1_memr,
                      #   iq_diqs1_memr,
                      #   iq_diqs_cs_smote1_memr,
                        ])
        BM_BALANCED_ACC.append([
         
          
          # iq_class0_ba,
          #             iq_class_SMOTE0_ba,
                      iq0_ba,
                        iq_smote0_ba,
                        iq_cs_smote0_ba,
                        iq_p0_ba,
                        iq_p_smote0_ba,
                        iq_p_cs_smote0_ba,
                        iq_diqs0_ba,
                        iq_diqs_cs_smote0_ba,
          # iq_class1_ba,
          #             iq_class_SMOTE1_ba,
          #             iq1_ba,
          #               iq_smote1_ba,
          #               iq_cs_smote1_ba,
          #               iq_p1_ba,
          #               iq_p_smote1_ba,
          #               iq_p_cs_smote1_ba,
          #               iq_diqs1_ba,
          #               iq_diqs_cs_smote1_ba,
                        ])
       

    
    # for metric in ['BA', 'MTTE', 'MEMR']:
    #   if metric=='BA':
    #     met = EPOCH_BA
    #   elif metric=='MTTE':
    #     met=EPOCH_MTTE
    #   elif metric=='MEMR':
    #     met=EPOCH_MEMR
    #   elif metric=='LOSS':
    #     met=EPOCH_LOSS
    #   mods = ['Classifier','Classifier-SMOTE',
    #           'IQS', 'IQS-SMOTE',
    #           'IQS-CS-SMOTE', 'Model-based IQS',
    #           'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE',
    #           'DO-IQS', 'DO-IQS-LB',
    #           'Classifier_conservative','Classifier-SMOTE_conservative',
    #                   'IQS_conservative', 'IQS-SMOTE_conservative',
    #                   'IQS-CS-SMOTE_conservative', 'Model-based IQS_conservative',
    #                   'Model-based IQS-SMOTE_conservative', 'Model-based IQS-CS-SMOTE_conservative',
    #                   'DO-IQS_conservative', 'DO-IQS-LB_conservative']
    #   memrs = [[l[i] for l in met] for i in range(len(met[0]))]
    #   cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
    #   gradient = np.arange(0, 1, 1/len(memrs))
    #   cols = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(memrs)]
    #   fig, ax = plt.subplots(figsize=(30,25))
    #   for i, m_memr in enumerate(memrs):
    #       mmr = np.array(m_memr)
    #       if metric=='MTTE':
    #         mmr = mmr*T_DOWNSAMPLE
    #       mean = np.mean(mmr, 0)
    #       std = np.std(mmr,0)
    #       lower_q = np.quantile(mmr, 0.25, axis=0)
    #       upper_q = np.quantile(mmr, 0.75, axis=0)
    #       ax.plot(range(mmr.shape[1]), mean, color=cols[i], label=mods[i],linewidth=5)
    #       # plt.fill_between(range(mmr.shape[1]), mean-lower_q, mean+upper_q, color=cols[i], alpha=0.5)
    #       ax.fill_between(range(mmr.shape[1]), mean-std, mean+std, color=cols[i], alpha=0.25)
    
    #   # plt.title('MEMR over training')
    #   ax.set_xlabel('epoch',size=60)
    #   ax.set_ylabel(metric,size=60)
    #   ax.tick_params(axis='both', which='major', labelsize=45)
    
    #   # Put a legend below current axis
    #   ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=45)
    #   plt.title(ex)
    #   plt.tight_layout()
    #   plt.show()
    
    np.save(f'outputs/{ex}_f1_weight_conserv.npy', np.array(BM_F1))
    np.save(f'outputs/{ex}_pr_auc_weight_conserv.npy', np.array(BM_PR_AUC))
    
    np.save(f'outputs/{ex}_mtte_weight_conserv.npy', np.array(BM_MTTE)*T_DOWNSAMPLE)
    np.save(f'outputs/{ex}_memr_weight_conserv.npy', np.array(BM_MEMR))
    np.save(f'outputs/{ex}_balanced_acc_weight_conserv.npy', np.array(BM_BALANCED_ACC))
    
    np.save(f'outputs/{ex}_EPOCH_MTTE_weight_conserv.npy', np.array(EPOCH_MTTE)*T_DOWNSAMPLE)
    np.save(f'outputs/{ex}_EPOCH_MEMR_weight_conserv.npy', np.array(EPOCH_MEMR))
    np.save(f'outputs/{ex}_EPOCH_LOSS_weight_conserv.npy', np.array(EPOCH_LOSS))
    np.save(f'outputs/{ex}_EPOCH_BA_weight_conserv.npy', np.array(EPOCH_BA))
 
# EPOCH_BA = []
# EPOCH_MTTE = []
# EPOCH_MEMR = []
# EPOCH_LOSS = []
    # BM_EMR
    # BM_ROC_AUC.append([iq_iql.roc_auc, iq_p.roc_auc, iq_diqs.roc_auc])
    # BM_F1.append([iq_iql.f1, iq_p.f1, iq_diqs.f1])
    # BM_ACCURACY.append([iq_iql.accuracy, iq_p.accuracy, iq_diqs.accuracy])
    # BM_PRECISION.append([iq_iql.precision, iq_p.precision, iq_diqs.precision])
    # BM_RECALL.append([iq_iql.recall, iq_p.recall, iq_diqs.recall])
    # BM_PR_AUC.append([iq_iql.pr_auc, iq_p.pr_auc, iq_diqs.pr_auc])

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.axis('on')
models_orig = [
# 'Classifier','Classifier-SMOTE',
#         'IQS', 'IQS-SMOTE',
#         'IQS-CS-SMOTE', 'Model-based IQS',
#         'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE',
#         'DO-IQS', 'DO-IQS-LB',
        # 'Classifier_conservative','Classifier-SMOTE_conservative',
                'IQS_conservative', 
                'IQS-SMOTE_conservative',
                'IQS-CS-SMOTE_conservative', 'Model-based IQS_conservative',
                'Model-based IQS-SMOTE_conservative', 'Model-based IQS-CS-SMOTE_conservative',
                'DO-IQS_conservative', 'DO-IQS-LB_conservative']
list_of_models = [0,1,2,3,4,5,6,7]
examples = [
    # 'car',
            # 'azure', 
            # 'nasa_turbofan', 
            # 'bmG', 
            # 'bmgG',
            # 'radial',
            'star',
            # 'CP1',
            # 'CP2',
            # 'CP3'
            # 'CP1M',
            # 'CP2M',
            # 'CP3M'
            ]

 
print('BA')
from tabulate import tabulate
table = {}
for example in examples: 
    ba = np.load(f'outputs/{example}_balanced_acc_weight_conserv.npy')
    table[example] = [str(round(ba[:,i].mean(0),4))+u"\u00B1"+str(round(ba[:,i].std(0),2)) for i, model in enumerate(models_orig)]
df = pd.DataFrame.from_dict(table)
df.index=models_orig
# print(tabulate(df.iloc[:,[0]], tablefmt="latex", headers=[examples[i] for i in [0]]))

# print(tabulate(df.iloc[:,[3,4,1,2]], tablefmt="latex", headers=[examples[i] for i in [3,4,1,2]]))
print(tabulate(df.iloc[:,[0]], tablefmt="latex", headers=[examples[i] for i in [0]]))
print(tabulate(df.iloc[:,[2,3]], tablefmt="latex", headers=[examples[i] for i in [2,3]]))
print(tabulate(df.iloc[:,[4,5,6]], tablefmt="latex", headers=[examples[i] for i in [4,5,6]]))
 
print('MTTE')
table = {}
for example in examples: 
    ba = np.load(f'outputs/{example}_mtte_weight_conserv.npy')
    table[example] = [str(round(ba[:,i].mean(0),4))+u"\u00B1"+str(round(ba[:,i].std(0),2)) for i, model in enumerate(models_orig)]
df = pd.DataFrame.from_dict(table)
df.index=models_orig
# print(tabulate(df.iloc[:,[0]], tablefmt="latex", headers=[examples[i] for i in [0]]))

# print(tabulate(df.iloc[:,[3,4,1,2]], tablefmt="latex", headers=[examples[i] for i in [3,4,1,2]]))
print(tabulate(df.iloc[:,[0,1]], tablefmt="latex", headers=[examples[i] for i in [0,1]]))
print(tabulate(df.iloc[:,[2,3]], tablefmt="latex", headers=[examples[i] for i in [2,3]]))
print(tabulate(df.iloc[:,[4,5,6]], tablefmt="latex", headers=[examples[i] for i in [4,5,6]]))
 

print('MEMR')
table = {}
for example in examples: 
    ba = np.load(f'outputs/{example}_memr_weight_conserv.npy')
    table[example] = [str(round(ba[:,i].mean(0),4))+u"\u00B1"+str(round(ba[:,i].std(0),2)) for i, model in enumerate(models_orig)]
df = pd.DataFrame.from_dict(table)
df.index=models_orig
# print(tabulate(df.iloc[:,[0]], tablefmt="latex", headers=[examples[i] for i in [0]]))

# print(tabulate(df.iloc[:,[3,4,1,2]], tablefmt="latex", headers=[examples[i] for i in [3,4,1,2]]))
print(tabulate(df.iloc[:,[0,1]], tablefmt="latex", headers=[examples[i] for i in [0,1]]))
print(tabulate(df.iloc[:,[2,3]], tablefmt="latex", headers=[examples[i] for i in [2,3]]))
print(tabulate(df.iloc[:,[4,5,6]], tablefmt="latex", headers=[examples[i] for i in [4,5,6]]))
 
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
driver_roc_auc = np.load('outputs/bmG_1_balanced_acc.npy')
bm_roc_auc = np.load('outputs/bmG_1_balanced_acc.npy')
# nmdp_roc_auc = np.load('outputs/bm_my_ygG_mtte.npy')
# oti_roc_auc = np.load('outputs/bmG_mtte.npy')
# #oti_roc_auc = np.delete(oti_roc_auc,1,1)
# oti_roc_auc2 = np.load('outputs/bmgG_mtte.npy')
# oti_roc_auc = driver_roc_auc
# df_rocauc_scores = pd.DataFrame({'example': np.array([np.repeat('BM G only', 5*8),
#                                                       np.repeat('BM g&G',5*8),
#                                               # np.repeat('y,g and G',70), np.repeat('G',70), np.repeat('y and G',70)
#                                               ]).flatten(),
#                                   'balanced_accuracy': np.stack([driver_roc_auc[:,:8],
#                                                             bm_roc_auc[:,:8],
#                                                         # nmdp_roc_auc, oti_roc_auc,oti_roc_auc2
#                                                         ]).flatten(),
#                                   'model': np.repeat(np.array(['IQ-Learning','IQ-SMOTE','IQ-CS-SMOTE','Model-based IQ', 'Model-based IQ-SMOTE', 'Model-based IQ-CS-SMOMTE','DIQS' 'DIQS-CS-LSMOTE',
#                                                               #  'IQ-Learning-Conservative','IQ-SMOTE-Conservative','IQ-CS-SMOTE-Conservative','Model-based IQS-Conservative',
#                                                               #  'IQS-SMOTE-Conservative', 'IQS-CS-SMOMTE-Conservative', 'DIQS-Conservative'
#                                                             ]).reshape(-1,1), 5*1,1).T.flatten()})
# sns.catplot(df_rocauc_scores,kind='bar', x="example", y="balanced_accuracy", hue='model',
#             height=4, aspect=1.5, legend=True, palette='viridis_r',errorbar=('sd',2))
# plt.title('Balanced accuracy for 2d BM examples')
# plt.show()
 
mtte = np.load('outputs/bmG_1_memr.npy')
mtte = mtte[:,:]
memr = np.load('outputs/bmG_1_mtte.npy')
memr = memr[:,:]
import matplotlib.pyplot as plt
a = np.mean(mtte,0)
b = np.mean(memr,0)
 
# c = np.quantile(mtte, 0.25,0) #, np.quantile(x, 0.75)
# c1 = np.quantile(mtte, 0.75,0)
c = np.std(mtte, 0)*2
c1=c
# d = np.quantile(memr, 0.25,0)
# d1 = np.quantile(memr, 0.75,0)
d = np.std(memr, 0)*2
d1=d
models = ['Classifier','Classifier-SMOTE',
        'IQS', 'IQS-SMOTE',
        'IQS-CS-SMOTE', 'Model-based IQS',
        'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE',
        'DO-IQS', 'DO-IQS-LB']
import matplotlib
cmap = matplotlib.colormaps.get_cmap('viridis_r')
gradient = np.linspace(0, 1, len(models))
list_of_colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]
 
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(a, b, c=list_of_colours)
for i, col in enumerate(list_of_colours):
  ax.errorbar(a[i], b[i],
              xerr=np.vstack([c[i],c1[i]]),
              yerr=np.vstack([d[i],d1[i]]),
              fmt='o',
              c=col,
              elinewidth=2)
ax.set_xlabel('MEMR', size=20)
ax.set_ylabel('MTTE', size=20)
ax.set_ylim((0,150))
ax.set_xlim((0,0.5))
ax.tick_params(axis='both', which='major', labelsize=15)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
list_of_pathes = [mpatches.Patch(color=list_of_colours[i], label=models[i]) for i in range(len(models))]
plt.legend(handles=list_of_pathes, fontsize=20)
# plt.title('MEMR to MTTE trade-off', fontsize=25)
# plt.legend()
# for i, txt in enumerate(models):
#     ax.annotate(txt, (a[i], b[i]), fontsize=15)
plt.tight_layout()
plt.show()
# %%