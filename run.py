# %%
from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from forward_algorithms.simulate_expert_data import Simulation
from inverse_opt_stopping.iq_learn_base import IQ_Agent, plot_st_reg_car
import os
import shutil
sns.set_style("whitegrid")

for ex in [
        'azure', 
            'nasa_turbofan', 
            # 'bmG', 
            # 'bmgG',
            # 'radial',
            # 'star',
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
    out_thresh = 0.00005
    out_thresh1 = 0.000005
    out_thresh2 = 0.00005
    out_thresh3 = 0.0005
    out_thresh4 = 0.005
    Q_ENT = True
    EPS = 0.1
    conservative=True
    IS_CS = False
    dqs_only = []
    for s in SEEDS:
     
        # Symmetrical BM resultse
        if ex=='radial':
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='radial', total_n=500)
        elif ex=='star': 
            T_DOWNSAMPLE=1
            bm_sym = Simulation(problem='star', total_n=500)
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
        # print(f"{bm_test_data['state_mem'].shape}")
        # plt.figure(figsize=(10,10),facecolor="w")
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.scatter([bm_expert_data['state_mem'][i][0] for i in range(len(bm_expert_data['state_mem']))],
        #             [bm_expert_data['state_mem'][i][1] for i in range(len(bm_expert_data['state_mem']))], s=0.3)
        # plt.show()
        # st_dat = bm_expert_data['state_mem'][bm_expert_data['action_mem']==0]
        # plt.figure(figsize=(10,10),facecolor="w")
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.scatter(st_dat[:,0],st_dat[:,1],s=0.3)
        # plt.show()
        # t1 = st_dat[:,2]==1/50
        # t2 = st_dat[:,2]==2/50
        # t3 = st_dat[:,2]==5/50
        # t4 = st_dat[:,2]==10/50
        # t5 = st_dat[:,2]==15/50
        # plt.figure(figsize=(10,10),facecolor="w")
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.scatter(st_dat[t1][:,0], st_dat[t1][:,1],c='blue',s=1)
        # plt.scatter(st_dat[t2][:,0], st_dat[t2][:,1],c='green',s=1)
        # plt.scatter(st_dat[t3][:,0], st_dat[t3][:,1],c='orange',s=1)
        # plt.scatter(st_dat[t4][:,0], st_dat[t4][:,1],c='red',s=1)
        # plt.scatter(st_dat[t5][:,0], st_dat[t5][:,1],c='black',s=1)
        # plt.show()
        # plt.figure(figsize=(10,10),facecolor="w")
        # plt.rcParams['axes.facecolor'] = 'white'
        # st_dat = bm_test_data['state_mem'][bm_test_data['action_mem']==0]
        # plt.scatter(st_dat[:,0],st_dat[:,1],s=0.3)
        # plt.show()
        print(f'Start training')
        # IQ-Learn
        print('Classifier')
        iq_class = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=False, oversampling=False,
                            q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=False,
                            discretiser=discretizer,
                            out_thresh=out_thresh,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            classify=True)
        iq_class.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_class.test(test_memory=bm_test_data, from_grid=False)
        # iq_class.test(test_memory=bm_test_data, from_grid=True)
        
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
        # iq_class.test(test_memory=bm_test_data, from_grid=True)
       
        print('Classifier with SMOTE oversampling')
        iq_class_SMOTE = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                            q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=False,
                            discretiser=discretizer,
                            out_thresh=out_thresh,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            classify=True)
        iq_class_SMOTE.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=False)
        # iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=True)
       
        # IQ-Learn
        print('IQ-Learn')
        iq = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0.01,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            discretiser=discretizer,
                            out_thresh=out_thresh,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            classify=False)
        iq.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq.test(test_memory=bm_test_data, from_grid=False)
        # iq.test(test_memory=bm_test_data, from_grid=True)
        # iq.out_thresh = out_thresh1
        # iq.test(test_memory=bm_test_data, from_grid=False)
        # iq.test(test_memory=bm_test_data, from_grid=True)
        # iq.out_thresh = out_thresh2
        # iq.test(test_memory=bm_test_data, from_grid=False)
        # iq.test(test_memory=bm_test_data, from_grid=True)
        # iq.out_thresh = out_thresh3
        # iq.test(test_memory=bm_test_data, from_grid=False)
        # iq.test(test_memory=bm_test_data, from_grid=True)
        # iq.out_thresh = out_thresh4
        # iq.test(test_memory=bm_test_data, from_grid=False)
        # iq.test(test_memory=bm_test_data, from_grid=True)
       
       
       
       
        # import os
        # os.mkdir('./out')
        # try:
        #     plot_st_reg_bm(iq_iql, bm_test_data, name='IQS')
        # except:
        #     print(f'Could not plot empty stopping region for iq_iql')
       
       
        shutil.rmtree('./out', ignore_errors=True)
       
        # IQ-Learning + oversampling SMOTE
        print('IQ-Learning + oversampling SMOTE')
        iq_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', 
                            cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_smote.out_thresh = out_thresh1
        # iq_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_smote.out_thresh = out_thresh2
        # iq_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_smote.out_thresh = out_thresh3
        # iq_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_smote.out_thresh = out_thresh4
        # iq_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_smote.test(test_memory=bm_test_data, from_grid=True)
       
       
        # os.mkdir('./out')
        # try:
        #     plot_st_reg_bm(iq_iql_smote, bm_test_data, name='IQS-SMOTE')
        # except:
        #     print(f'Could not plot empty stopping region for iq_iql_smote')
        # shutil.rmtree('./out', ignore_errors=True)
       
        # IQ-Learning + oversampling CS-SMOTE
        print('IQ-Learning + oversampling CS-SMOTE')
        iq_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_cs_smote.out_thresh = out_thresh1
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_cs_smote.out_thresh = out_thresh2
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_cs_smote.out_thresh = out_thresh3
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_cs_smote.out_thresh = out_thresh4
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # os.mkdir('./out')
        # try:
        #     plot_st_reg_bm(iq_iql_cs_smote, bm_test_data, name='IQS-CS-SMOTE')
        # except:
        #     print(f'Could not plot empty stopping region for iq_iql_cs_smote')    
        # shutil.rmtree('./out', ignore_errors=True)
       
        # Approximating P, no oversampling
        print('Approximating P, no oversampling')
        iq_p = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=True, oversampling=None,
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_p.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_p.test(test_memory=bm_test_data, from_grid=False)
        # iq_p.test(test_memory=bm_test_data, from_grid=True)
        # iq_p.out_thresh = out_thresh1
        # iq_p.test(test_memory=bm_test_data, from_grid=False)
        # iq_p.test(test_memory=bm_test_data, from_grid=True)
        # iq_p.out_thresh = out_thresh2
        # iq_p.test(test_memory=bm_test_data, from_grid=False)
        # iq_p.test(test_memory=bm_test_data, from_grid=True)
        # iq_p.out_thresh = out_thresh3
        # iq_p.test(test_memory=bm_test_data, from_grid=False)
        # iq_p.test(test_memory=bm_test_data, from_grid=True)
        # iq_p.out_thresh = out_thresh4
        # iq_p.test(test_memory=bm_test_data, from_grid=False)
        # iq_p.test(test_memory=bm_test_data, from_grid=True)
       
       
        # os.mkdir('./out')
        # try:
        #     plot_st_reg_bm(iq_p, bm_test_data, name='Model-based IQS')
        # except:
        #     print(f'Could not plot empty stopping region for iq_p')
        # shutil.rmtree('./out', ignore_errors=True)        
           
        # Approximating P + oversampling SMOTE
        print('Approximating P + oversampling SMOTE')
        iq_p_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=True, oversampling='SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_p_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_smote.out_thresh = out_thresh1
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_smote.out_thresh = out_thresh2
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_smote.out_thresh = out_thresh3
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_smote.out_thresh = out_thresh4
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_smote.test(test_memory=bm_test_data, from_grid=True)
       
        # os.mkdir('./out')
        # try:
        #     plot_st_reg_bm(iq_smote, bm_test_data, name='Model-based IQS-SMOTE')
        # except:
        #     print(f'Could not plot empty stopping region for iq_smote')
        # shutil.rmtree('./out', ignore_errors=True)  
       
        # Approximating P + oversampling CS-SMOTE
        print('Approximating P + oversampling CS-SMOTE')
        iq_p_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE',
                            q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_p_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_cs_smote.out_thresh = out_thresh1
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_cs_smote.out_thresh = out_thresh2
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_cs_smote.out_thresh = out_thresh3
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
        # iq_p_cs_smote.out_thresh = out_thresh4
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
    
        # try:
        #     plot_st_reg_bm(iq_cs_smote, bm_test_data, name='Model-based IQS-CS-SMOTE')
        # except:
        #     print(f'Could not plot empty stopping region for iq_cs_smote')
        # shutil.rmtree('./out', ignore_errors=True)    
       
        # Approximating g and P, no oversampling
        print('Approximating g and P, no oversampling')
        iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=True, approx_dynamics=True, oversampling=None,
                            q_lr=0.01, env_lr=0.01, g_lr=0.001,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_diqs.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_diqs.test(test_memory=bm_test_data)
        # iq_diqs.out_thresh = out_thresh1
        # iq_diqs.test(test_memory=bm_test_data, from_grid=False)
        # iq_diqs.out_thresh = out_thresh2
        # iq_diqs.test(test_memory=bm_test_data, from_grid=False)
        # iq_diqs.out_thresh = out_thresh3
        # iq_diqs.test(test_memory=bm_test_data, from_grid=False)
        # iq_diqs.out_thresh = out_thresh4
        # iq_diqs.test(test_memory=bm_test_data, from_grid=False)
    
       
       
        # Approximating g and P, CS-LSMOTE oversampling
        print('Approximating g and P, CS-LSMOTE oversampling')
        iq_diqs_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                            approx_g=True, approx_dynamics=True, oversampling='CS-LSMOTE',
                            q_lr=0.01, env_lr=0.01, g_lr=0.001,
                            epsilon=EPS,seed=s,
                            divergence='hellinger', cross_val_splits=2,
                            conservative=conservative,
                            q_entropy=Q_ENT,
                            SMOTE_K=SMOTE_K,
                            is_cs=IS_CS,
                            discretiser=discretizer,
                            out_thresh=out_thresh
                            )
        iq_diqs_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
        iq_diqs_cs_smote.test(test_memory=bm_test_data)
        # iq_diqs_cs_smote.out_thresh = out_thresh1
        # iq_diqs_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_diqs_cs_smote.out_thresh = out_thresh2
        # iq_diqs_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_diqs_cs_smote.out_thresh = out_thresh3
        # iq_diqs_cs_smote.test(test_memory=bm_test_data, from_grid=False)
        # iq_diqs_cs_smote.out_thresh = out_thresh4
        # iq_diqs_cs_smote.test(test_memory=bm_test_data, from_grid=False)
       
        # dqs_only.append([iq_diqs.f1, iq_diqs.pr_auc, iq_diqs.mtte])
       
        # # CONSERVATIVE
        # # IQ-Learn
        # from binning import MDLP_Discretizer
        # discretizer = MDLP_Discretizer(features=np.arange(bm_expert_data['state_mem'].shape[1]))
        # discretizer.fit(bm_expert_data['state_mem'], bm_expert_data['action_mem'])
    
        # print('IQ-Learn')
        # iq_conserv = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling=None,
        #                     q_lr=0.02, env_lr=0.02, g_lr=0.005,
        #                     epsilon=0.1,seed=s,
        #                     divergence='js', cross_val_splits=2,
        #                     conservative=True,
        #                     discretiser=discretizer,
        #                     out_thresh=0.0001
        #                     )
        # iq_conserv.train(mem=bm_expert_data, batch_size=75, n_epoches=N_EPOCHS)
        # iq_conserv.test(test_memory=bm_test_data)
        # # import os
        # # os.mkdir('./out')
        # # try:
        # #     plot_st_reg_bm(iq_iql, bm_test_data, name='IQS')
        # # except:
        # #     print(f'Could not plot empty stopping region for iq_iql')
       
       
        # shutil.rmtree('./out', ignore_errors=True)
       
        # # IQ-Learning + oversampling SMOTE
        # print('IQ-Learning + oversampling SMOTE')
        # iq_smote_conserv = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling='SMOTE',
        #                     q_lr=0.02, env_lr=0.02, g_lr=0.005,
        #                     epsilon=0.1,seed=s,
        #                     divergence='js', cross_val_splits=2,
        #                     conservative=True,
        #                     discretiser=discretizer,
        #                     out_thresh=0.0001
        #                     )
        # iq_smote_conserv.train(mem=bm_expert_data, batch_size=75, n_epoches=N_EPOCHS)
        # iq_smote_conserv.test(test_memory=bm_test_data)
       
        # # os.mkdir('./out')
        # # try:
        # #     plot_st_reg_bm(iq_iql_smote, bm_test_data, name='IQS-SMOTE')
        # # except:
        # #     print(f'Could not plot empty stopping region for iq_iql_smote')
        # # shutil.rmtree('./out', ignore_errors=True)
       
       
        # # IQ-Learning + oversampling CS-SMOTE
        # print('IQ-Learning + oversampling CS-SMOTE')
        # iq_cs_smote_conserv = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE',
        #                     q_lr=0.02, env_lr=0.02, g_lr=0.005,
        #                     epsilon=0.1,seed=s,
        #                     divergence='js', cross_val_splits=2,
        #                     conservative=True,
        #                     discretiser=discretizer,
        #                     out_thresh=0.0001
        #                     )
        # iq_cs_smote_conserv.train(mem=bm_expert_data, batch_size=75, n_epoches=N_EPOCHS)
        # iq_cs_smote_conserv.test(test_memory=bm_test_data)
        # # os.mkdir('./out')
        # # try:
        # #     plot_st_reg_bm(iq_iql_cs_smote, bm_test_data, name='IQS-CS-SMOTE')
        # # except:
        # #     print(f'Could not plot empty stopping region for iq_iql_cs_smote')    
        # # shutil.rmtree('./out', ignore_errors=True)
       
        # # Approximating P, no oversampling
        # print('Approximating P, no oversampling')
        # iq_p_conserv = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=True, oversampling=None,
        #                     q_lr=0.02, env_lr=0.02, g_lr=0.005,
        #                     epsilon=0.1,seed=s,
        #                     divergence='js', cross_val_splits=2,
        #                     conservative=True,
        #                     discretiser=discretizer,
        #                     out_thresh=0.0001
        #                     )
        # iq_p_conserv.train(mem=bm_expert_data, batch_size=75, n_epoches=N_EPOCHS)
        # iq_p_conserv.test(test_memory=bm_test_data)
        # # os.mkdir('./out')
        # # try:
        # #     plot_st_reg_bm(iq_p, bm_test_data, name='Model-based IQS')
        # # except:
        # #     print(f'Could not plot empty stopping region for iq_p')
        # # shutil.rmtree('./out', ignore_errors=True)        
           
        # # Approximating P + oversampling SMOTE
        # print('Approximating P + oversampling SMOTE')
        # iq_p_smote_conserv = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=True, oversampling='SMOTE',
        #                     q_lr=0.02, env_lr=0.02, g_lr=0.005,
        #                     epsilon=0.1,seed=s,
        #                     divergence='js', cross_val_splits=2,
        #                     conservative=True,
        #                     discretiser=discretizer,
        #                     out_thresh=0.0001
        #                     )
        # iq_p_smote_conserv.train(mem=bm_expert_data, batch_size=75, n_epoches=N_EPOCHS)
        # iq_p_smote_conserv.test(test_memory=bm_test_data)
        # # os.mkdir('./out')
        # # try:
        # #     plot_st_reg_bm(iq_smote, bm_test_data, name='Model-based IQS-SMOTE')
        # # except:
        # #     print(f'Could not plot empty stopping region for iq_smote')
        # # shutil.rmtree('./out', ignore_errors=True)  
       
        # # Approximating P + oversampling CS-SMOTE
        # print('Approximating P + oversampling CS-SMOTE')
        # iq_p_cs_smote_conserv = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE',
        #                     q_lr=0.02, env_lr=0.02, g_lr=0.005,
        #                     epsilon=0.1,seed=s,
        #                     divergence='js', cross_val_splits=2,
        #                     conservative=True,
        #                     discretiser=discretizer,
        #                     out_thresh=0.0001
        #                     )
        # iq_p_cs_smote_conserv.train(mem=bm_expert_data, batch_size=75, n_epoches=N_EPOCHS)
        # iq_p_cs_smote_conserv.test(test_memory=bm_test_data)
        # # os.mkdir('./out')
        # # try:
        # #     plot_st_reg_bm(iq_cs_smote, bm_test_data, name='Model-based IQS-CS-SMOTE')
        # # except:
        # #     print(f'Could not plot empty stopping region for iq_cs_smote')
        # # shutil.rmtree('./out', ignore_errors=True)    
       
        # # Approximating g and P, no oversampling
        # print('Approximating g and P, no oversampling')
        # iq_diqs_conserv = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
        #                     approx_g=True, approx_dynamics=True, oversampling=None,
        #                     q_lr=0.02, env_lr=0.02, g_lr=0.005,
        #                     epsilon=0.1,seed=s,
        #                     divergence='js', cross_val_splits=2,
        #                     conservative=True,
        #                     discretiser=discretizer,
        #                     out_thresh=0.0001
        #                     )
        # iq_diqs_conserv.train(mem=bm_expert_data, batch_size=75, n_epoches=N_EPOCHS)
        # iq_diqs_conserv.test(test_memory=bm_test_data)
       
        EPOCH_BA.append([iq_class.epoch_balanced_accuracy,
                      iq_class_SMOTE.epoch_balanced_accuracy,
                      iq.epoch_balanced_accuracy,
                      iq_smote.epoch_balanced_accuracy,
                      iq_cs_smote.epoch_balanced_accuracy,
                      iq_p.epoch_balanced_accuracy,
                      iq_p_smote.epoch_balanced_accuracy,
                      iq_p_cs_smote.epoch_balanced_accuracy,
                      iq_diqs.epoch_balanced_accuracy,
                      iq_diqs_cs_smote.epoch_balanced_accuracy
                     ])
        EPOCH_MTTE.append([
          iq_class.epoch_mtte,
                      iq_class_SMOTE.epoch_mtte,
                      iq.epoch_mtte,
                      iq_smote.epoch_mtte,
                      iq_cs_smote.epoch_mtte,
                      iq_p.epoch_mtte,
                      iq_p_smote.epoch_mtte,
                      iq_p_cs_smote.epoch_mtte,
                      iq_diqs.epoch_mtte,
                      iq_diqs_cs_smote.epoch_mtte
                     ])
        EPOCH_MEMR.append([
          iq_class.epoch_memr,
                      iq_class_SMOTE.epoch_memr,
                      iq.epoch_memr,
                      iq_smote.epoch_memr,
                      iq_cs_smote.epoch_memr,
                      iq_p.epoch_memr,
                      iq_p_smote.epoch_memr,
                      iq_p_cs_smote.epoch_memr,
                      iq_diqs.epoch_memr,
                      iq_diqs_cs_smote.epoch_memr
                     ])
        EPOCH_LOSS.append([
          iq_class.epoch_loss,
                      iq_class_SMOTE.epoch_loss,
                      iq.epoch_loss,
                      iq_smote.epoch_loss,
                      iq_cs_smote.epoch_loss,
                      iq_p.epoch_loss,
                      iq_p_smote.epoch_loss,
                      iq_p_cs_smote.epoch_loss,
                      iq_diqs.epoch_loss,
                      iq_diqs_cs_smote.epoch_loss
                     ])
       
       
        BM_F1.append([
          iq_class.f1,
                      iq_class_SMOTE.f1,
                      iq.f1,
                      iq_smote.f1,
                      iq_cs_smote.f1,
                      iq_p.f1,
                      iq_p_smote.f1,
                      iq_p_cs_smote.f1,
                      iq_diqs.f1,
                      iq_diqs_cs_smote.f1
                      # iq_conserv.f1,
                      # iq_smote_conserv.f1,
                      # iq_cs_smote_conserv.f1,
                      # iq_p_conserv.f1,
                      # iq_p_smote_conserv.f1,
                      # iq_p_cs_smote_conserv.f1,
                      # iq_diqs_conserv.f1
                     ])
        BM_PR_AUC.append([
          iq_class.pr_auc,
                      iq_class_SMOTE.pr_auc,
                      iq.pr_auc,
                          iq_smote.pr_auc,
                          iq_cs_smote.pr_auc,
                          iq_p.pr_auc,
                          iq_p_smote.pr_auc,
                          iq_p_cs_smote.pr_auc,
                          iq_diqs.pr_auc,
                          iq_diqs_cs_smote.pr_auc
                      # iq_conserv.pr_auc,
                      # iq_smote_conserv.pr_auc,
                      # iq_cs_smote_conserv.pr_auc,
                      # iq_p_conserv.pr_auc,
                      # iq_p_smote_conserv.pr_auc,
                      # iq_p_cs_smote_conserv.pr_auc,
                      # iq_diqs_conserv.pr_auc
                      ])    
        BM_MTTE.append([
          iq_class.mtte,
                      iq_class_SMOTE.mtte,
                      iq.mtte,
                        iq_smote.mtte,
                        iq_cs_smote.mtte,
                        iq_p.mtte,
                        iq_p_smote.mtte,
                        iq_p_cs_smote.mtte,
                        iq_diqs.mtte,
                        iq_diqs_cs_smote.mtte
                      # iq_conserv.mtte,
                      # iq_smote_conserv.mtte,
                      # iq_cs_smote_conserv.mtte,
                      # iq_p_conserv.mtte,
                      # iq_p_smote_conserv.mtte,
                      # iq_p_cs_smote_conserv.mtte,
                      # iq_diqs_conserv.mtte
                                        ])
        BM_MEMR.append([
          iq_class.memr,
                      iq_class_SMOTE.memr,
                      iq.memr,
                        iq_smote.memr,
                        iq_cs_smote.memr,
                        iq_p.memr,
                        iq_p_smote.memr,
                        iq_p_cs_smote.memr,
                        iq_diqs.memr,
                        iq_diqs_cs_smote.memr
                      # iq_conserv.memr,
                      # iq_smote_conserv.memr,
                      # iq_cs_smote_conserv.memr,
                      # iq_p_conserv.memr,
                      # iq_p_smote_conserv.memr,
                      # iq_p_cs_smote_conserv.memr,
                      # iq_diqs_conserv.memr
                        ])
        BM_BALANCED_ACC.append([
          iq_class.balanced_accuracy,
                      iq_class_SMOTE.balanced_accuracy,
                      iq.balanced_accuracy,
                      iq_smote.balanced_accuracy,
                      iq_cs_smote.balanced_accuracy,
                      iq_p.balanced_accuracy,
                      iq_p_smote.balanced_accuracy,
                      iq_p_cs_smote.balanced_accuracy,
                      iq_diqs.balanced_accuracy,
                      iq_diqs_cs_smote.balanced_accuracy
                      # iq_conserv.balanced_accuracy,
                      # iq_smote_conserv.balanced_accuracy,
                      # iq_cs_smote_conserv.balanced_accuracy,
                      # iq_p_conserv.balanced_accuracy,
                      # iq_p_smote_conserv.balanced_accuracy,
                      # iq_p_cs_smote_conserv.balanced_accuracy,
                      # iq_diqs_conserv.balanced_accuracy
                        ])
       
        # BM_F1.append([iq_iql.f1, iq_diqs.f1])
        # BM_PR_AUC.append([iq_iql.pr_auc, iq_diqs.pr_auc])    
        # BM_MTTE.append([iq_iql.mtte, iq_diqs.mtte])
        # BM_MEMR.append([iq_iql.memr, iq_diqs.memr])
    
    
    
    # np.array(EPOCH_MTTE)*T_DOWNSAMPLE)
    # np.array(EPOCH_MEMR))
    # np.array(EPOCH_LOSS))
    # np.array(EPOCH_BA))
    
    for metric in ['BA', 'MTTE', 'MEMR', 'LOSS']:
      if metric=='BA':
        met = EPOCH_BA
      elif metric=='MTTE':
        met=EPOCH_MTTE
      elif metric=='MEMR':
        met=EPOCH_MEMR
      elif metric=='LOSS':
        met=EPOCH_LOSS
      mods = ['Classifier_conserv','Classifier-SMOTE_conserv',
              'IQS_conserv', 'IQS-SMOTE_conserv',
              'IQS-CS-SMOTE_conserv', 'Model-based IQS_conserv',
              'Model-based IQS-SMOTE_conserv', 'Model-based IQS-CS-SMOTE_conserv',
              'DO-IQS_conserv', 'DO-IQS-LB_conserv']
      memrs = [[l[i] for l in met] for i in range(len(met[0]))]
      cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
      gradient = np.arange(0, 1, 1/len(memrs))
      cols = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(memrs)]
      fig, ax = plt.subplots(figsize=(30,25))
      for i, m_memr in enumerate(memrs):
          mmr = np.array(m_memr)
          if metric=='MTTE':
            mmr = mmr*T_DOWNSAMPLE
          mean = np.mean(mmr, 0)
          std = np.std(mmr,0)
          lower_q = np.quantile(mmr, 0.25, axis=0)
          upper_q = np.quantile(mmr, 0.75, axis=0)
          ax.plot(range(mmr.shape[1]), mean, color=cols[i], label=mods[i],linewidth=5)
          # plt.fill_between(range(mmr.shape[1]), mean-lower_q, mean+upper_q, color=cols[i], alpha=0.5)
          ax.fill_between(range(mmr.shape[1]), mean-std, mean+std, color=cols[i], alpha=0.25)
    
      # plt.title('MEMR over training')
      ax.set_xlabel('epoch',size=60)
      ax.set_ylabel(metric,size=60)
      ax.tick_params(axis='both', which='major', labelsize=45)
    
      # Put a legend below current axis
      ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=45)
      plt.title(ex)
      plt.tight_layout()
      plt.show()
    
    np.save(f'outputs/{ex}_f1.npy', np.array(BM_F1))
    np.save(f'outputs/{ex}_pr_auc.npy', np.array(BM_PR_AUC))
    
    np.save(f'outputs/{ex}_mtte.npy', np.array(BM_MTTE)*T_DOWNSAMPLE)
    np.save(f'outputs/{ex}_memr.npy', np.array(BM_MEMR))
    np.save(f'outputs/{ex}_balanced_acc.npy', np.array(BM_BALANCED_ACC))
    
    np.save(f'outputs/{ex}_EPOCH_MTTE.npy', np.array(EPOCH_MTTE)*T_DOWNSAMPLE)
    np.save(f'outputs/{ex}_EPOCH_MEMR.npy', np.array(EPOCH_MEMR))
    np.save(f'outputs/{ex}_EPOCH_LOSS.npy', np.array(EPOCH_LOSS))
    np.save(f'outputs/{ex}_EPOCH_BA.npy', np.array(EPOCH_BA))
 
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
models_orig = ['Classifier_conserv','Classifier-SMOTE_conserv',
        'IQS_conserv', 'IQS-SMOTE_conserv',
        'IQS-CS-SMOTE_conserv', 'Model-based IQS_conserv',
        'Model-based IQS-SMOTE_conserv', 'Model-based IQS-CS-SMOTE_conserv',
        'DO-IQS_conserv', 'DO-IQS-LB_conserv']
list_of_models = [0,1,2,3,4,5,6,7,8,9]
examples = [
    # 'car',
    #         'azure', 
    #         'nasa_turbofan', 
    #         'bmG', 
    #         'bmgG',
    #         'radial',
    #         'star',
    #         'CP1',
    #         'CP2',
    #         'CP3'
            # 'CP1M',
            # 'CP2M',
            # 'CP3M'
            ]
for example in examples: 
    if example=='nasa_turbofan_10' or example=='car_1':
        list_of_models = [2,3,4,5,6,7,8,9]
    else:
        list_of_models = [2,3,4,5,6,7,8,9]
    mtte = np.load(f'outputs/{example}_mtte.npy')
    mtte = mtte[:,list_of_models]
    memr = np.load(f'outputs/{example}_memr.npy')
    memr = memr[:,list_of_models]
    import matplotlib.pyplot as plt
    a = np.median(mtte,0)
    b = np.median(memr,0)
    
    c = a-np.quantile(mtte, 0.25,0) #, np.quantile(x, 0.75)
    c1 = -a+np.quantile(mtte, 0.75,0)
    # c = np.std(mtte, 0)*2
    # c1=c 
    d = b-np.quantile(memr, 0.25,0)
    d1 = -b+np.quantile(memr, 0.75,0)
    # d = np.std(memr, 0)*2
    # d1=d
    # models_orig = ['IQS','IQS-SMOTE','IQS-CS-SMOTE','Model-based IQS', 'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE', 'DO-IQS', 'DO-IQS-CS-LSMOTE']
    models = [models_orig[i] for i in list_of_models]
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
 
    # gradient = np.linspace(0, 1, len(models_orig))[list_of_models]
    gradient = np.arange(0, 1, 1/len(models_orig))[list_of_models]
    list_of_colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(models)]
    # list_of_colours = ['#377eb8', '#ff7f00', '#4daf4a',
    #               '#f781bf', '#a65628', '#984ea3',
    #               '#999999', '#e41a1c', '#dede00']
    # list_of_colours = list_of_colours[:len(list_of_models)]
    fig, ax = plt.subplots(figsize=(25,15))
    ax.scatter(b, a, c=list_of_colours, s=60)
    for i, col in enumerate(list_of_colours):
        markers, caps, bars = ax.errorbar(b[i], a[i], 
                xerr=np.vstack([d[i],d1[i]]), 
                yerr=np.vstack([c[i],c1[i]]), 
                fmt='o',
                c=col,
                elinewidth=5,
                capsize=10, 
                capthick=5)
        [bar.set_alpha(0.75) for bar in bars]
        [cap.set_alpha(0.75) for cap in caps]
    
    ax.set_xlabel('MEMR', size=60)
    ax.set_ylabel('MTTE', size=60)
    # ax.set_ylim((0,70))
    ax.set_xlim((0,0.5))
    ax.tick_params(axis='both', which='major', labelsize=55)
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    list_of_pathes = [mpatches.Patch(color=list_of_colours[i], label=models[i]) for i in range(len(models))]
    # plt.legend(handles=list_of_pathes, fontsize=45,)
    # leg = plt.legend(handles=list_of_pathes, fontsize=70,loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #       ncol=len(list_of_pathes), fancybox=True)
    
    plt.legend().set_visible(False)
    # plt.legend()
    plt.title(example, fontsize=70)
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    # ax.xaxis.set_major_formatter('{x:.0f}')
    # For the minor ticks, use no labels; default NullFormatter.
    # ax.xaxis.set_minor_locator(MultipleLocator(5))
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'plots/{example}_tradeoff.png', bbox_inches = 'tight')
    
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.figure(figsize=(60,10))
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.axis('off')
plt.legend(handles=list_of_pathes,fontsize=100,loc='center', ncol=len(list_of_pathes))
plt.savefig(f'plots/legend.png', bbox_inches = 'tight')
plt.axis('on')
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True
 
from tabulate import tabulate
table = {}
for example in examples: 
    ba = np.load(f'outputs/{example}_balanced_acc.npy')
    table[example] = [str(round(ba[:,i].mean(0),4))+u"\u00B1"+str(round(ba[:,i].std(0),2)) for i, model in enumerate(models_orig)]
df = pd.DataFrame.from_dict(table)
df.index=models_orig
print(tabulate(df.iloc[:,[0,1]], tablefmt="latex", headers=[examples[i] for i in [0,1]]))
print(tabulate(df.iloc[:,[3,4,1,2]], tablefmt="latex", headers=[examples[i] for i in [3,4,1,2]]))
print(tabulate(df.iloc[:,[0,5,6]], tablefmt="latex", headers=[examples[i] for i in [0,5,6]]))
print(tabulate(df.iloc[:,[7,8,9]], tablefmt="latex", headers=[examples[i] for i in [7,8,9]]))
 
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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

rows = [
    ("Classifier",                    6.6309, 0.72,  6.2125, 0.43,  8.0883, 1.14,  7.6432, 1.13),
    ("Classifier-SMOTE",              6.6520, 0.71,  6.2077, 0.49,  8.1716, 1.09,  7.7031, 1.09),
    ("IQS",                           9.0747, 0.56,  9.0747, 0.56, 10.3813, 0.40, 10.3813, 0.40),
    ("IQS-SMOTE",                     0.1702, 0.10,  1.2084, 0.12,  1.0184, 1.32,  2.3360, 0.44),
    ("IQS-CS-SMOTE",                  0.2994, 0.13,  1.1763, 0.07,  0.4908, 0.18,  1.8376, 0.28),
    ("Model-based IQS",               0.1593, 0.12,  1.1918, 0.06,  0.4820, 0.24,  1.8007, 0.26),
    ("Model-based IQS-SMOTE",         0.1654, 0.09,  1.1592, 0.11,  0.4841, 0.44,  1.8555, 0.27),
    ("Model-based IQS-CS-SMOTE",      0.3141, 0.14,  1.1608, 0.09,  0.7593, 0.29,  1.7563, 0.26),
    ("DO-IQS",                        0.2675, 0.10,  1.1666, 0.06,  0.5838, 0.23,  1.8239, 0.23),
    ("DO-IQS-LB",                     0.8790, 0.23,  1.3357, 0.03,  1.8993, 0.22,  2.1927, 0.19),
]

df = pd.DataFrame(rows, columns=[
    "Model",
    "bmG_Standard_mean", "bmG_Standard_std",
    "bmG_Conservative_mean", "bmG_Conservative_std",
    "bmgG_Standard_mean", "bmgG_Standard_std",
    "bmgG_Conservative_mean", "bmgG_Conservative_std",
]).set_index("Model")

# Optional: a pretty MultiIndex columns view
df_pretty = df.copy()
df_pretty.columns = pd.MultiIndex.from_tuples([
    ("bmG","Standard","mean"), ("bmG","Standard","std"),
    ("bmG","Conservative","mean"), ("bmG","Conservative","std"),
    ("bmgG","Standard","mean"), ("bmgG","Standard","std"),
    ("bmgG","Conservative","mean"), ("bmgG","Conservative","std"),
])

rows = [
    ("Classifier",               0.1067, 0.05, 0.0320, 0.02, 0.1413, 0.05, 0.0720, 0.02),
    ("Classifier-SMOTE",         0.1173, 0.05, 0.0373, 0.03, 0.1413, 0.06, 0.0720, 0.02),
    ("IQS",                      0.0000, 0.00, 0.0000, 0.00, 0.0000, 0.00, 0.0000, 0.00),
    ("IQS-SMOTE",                0.4000, 0.06, 0.1440, 0.07, 0.5920, 0.11, 0.2747, 0.09),
    ("IQS-CS-SMOTE",             0.2960, 0.08, 0.1093, 0.04, 0.4293, 0.09, 0.2053, 0.06),
    ("Model-based IQS",          0.3307, 0.08, 0.1040, 0.05, 0.4080, 0.08, 0.1680, 0.07),
    ("Model-based IQS-SMOTE",    0.2667, 0.07, 0.0987, 0.04, 0.4107, 0.10, 0.1973, 0.05),
    ("Model-based IQS-CS-SMOTE", 0.1440, 0.04, 0.0720, 0.05, 0.2720, 0.04, 0.1360, 0.05),
    ("DO-IQS",                   0.2880, 0.06, 0.1040, 0.04, 0.4053, 0.09, 0.1920, 0.07),
    ("DO-IQS-LB",                0.0640, 0.03, 0.0400, 0.03, 0.1440, 0.02, 0.0880, 0.03),
]

df = pd.DataFrame(rows, columns=[
    "Model",
    "bmG_Standard_mean", "bmG_Standard_std",
    "bmG_Conservative_mean", "bmG_Conservative_std",
    "bmgG_Standard_mean", "bmgG_Standard_std",
    "bmgG_Conservative_mean", "bmgG_Conservative_std",
]).set_index("Model")

# Optional pretty MultiIndex columns
df_pretty1 = df.copy()
df_pretty1.columns = pd.MultiIndex.from_tuples([
    ("bmG",  "Standard",     "mean"), ("bmG",  "Standard",     "std"),
    ("bmG",  "Conservative", "mean"), ("bmG",  "Conservative", "std"),
    ("bmgG", "Standard",     "mean"), ("bmgG", "Standard",     "std"),
    ("bmgG", "Conservative", "mean"), ("bmgG", "Conservative", "std"),
])


models = [ 'IQS-SMOTE',
        'IQS-CS-SMOTE', 'Model-based IQS',
        'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE',
        'DO-IQS', 'DO-IQS-LB']
a = df_pretty.bmgG.Standard['mean'].values[3:]
b = df_pretty1.bmgG.Standard['mean'].values[3:]
c = df_pretty.bmgG.Standard['std'].values[3:]
c1 = c
d=df_pretty1.bmgG.Standard['std'].values[3:]
d1=d
fig, ax = plt.subplots(figsize=(25,15))
ax.scatter(b, a, c='blue', s=60)
for i, col in enumerate(models):
    markers, caps, bars = ax.errorbar(b[i], a[i], 
            xerr=np.vstack([d[i],d1[i]]), 
            yerr=np.vstack([c[i],c1[i]]), 
            fmt='o',
            c='blue',
            elinewidth=5,
            capsize=10, 
            capthick=5)
    [bar.set_alpha(0.75) for bar in bars]
    [cap.set_alpha(0.75) for cap in caps]


a = df_pretty.bmG.Conservative['mean'].values[3:]
b = df_pretty1.bmG.Conservative['mean'].values[3:]
c = df_pretty.bmG.Conservative['std'].values[3:]
c1 = c
d=df_pretty1.bmG.Conservative['std'].values[3:]
d1=d

ax.scatter(b, a, c='red', s=60)
for i, col in enumerate(models):
    markers, caps, bars = ax.errorbar(b[i], a[i], 
            xerr=np.vstack([d[i],d1[i]]), 
            yerr=np.vstack([c[i],c1[i]]), 
            fmt='o',
            c='red',
            elinewidth=5,
            capsize=10, 
            capthick=5)
    [bar.set_alpha(0.75) for bar in bars]
    [cap.set_alpha(0.75) for cap in caps]
ax.set_xlabel('MEMR', size=60)
ax.set_ylabel('MTTE', size=60)
ax.set_xlim((0,0.5))
ax.tick_params(axis='both', which='major', labelsize=55)

list_of_pathes = [mpatches.Patch(color='blue', label='Standard'),
                  mpatches.Patch(color='red', label= 'Conservative')]
plt.legend().set_visible(False)
plt.title('bmgG', fontsize=70)
plt.legend(handles=list_of_pathes, fontsize=50)
plt.tight_layout()
plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

rows = [
    ("Classifier",                    6.6309, 0.72,  6.2125, 0.43,  8.0883, 1.14,  7.6432, 1.13),
    ("Classifier-SMOTE",              6.6520, 0.71,  6.2077, 0.49,  8.1716, 1.09,  7.7031, 1.09),
    ("IQS",                           9.0747, 0.56,  9.0747, 0.56, 10.3813, 0.40, 10.3813, 0.40),
    ("IQS-SMOTE",                     0.1702, 0.10,  1.2084, 0.12,  1.0184, 1.32,  2.3360, 0.44),
    ("IQS-CS-SMOTE",                  0.2994, 0.13,  1.1763, 0.07,  0.4908, 0.18,  1.8376, 0.28),
    ("Model-based IQS",               0.1593, 0.12,  1.1918, 0.06,  0.4820, 0.24,  1.8007, 0.26),
    ("Model-based IQS-SMOTE",         0.1654, 0.09,  1.1592, 0.11,  0.4841, 0.44,  1.8555, 0.27),
    ("Model-based IQS-CS-SMOTE",      0.3141, 0.14,  1.1608, 0.09,  0.7593, 0.29,  1.7563, 0.26),
    ("DO-IQS",                        0.2675, 0.10,  1.1666, 0.06,  0.5838, 0.23,  1.8239, 0.23),
    ("DO-IQS-LB",                     0.8790, 0.23,  1.3357, 0.03,  1.8993, 0.22,  2.1927, 0.19),
]

df = pd.DataFrame(rows, columns=[
    "Model",
    "bmG_Standard_mean", "bmG_Standard_std",
    "bmG_Conservative_mean", "bmG_Conservative_std",
    "bmgG_Standard_mean", "bmgG_Standard_std",
    "bmgG_Conservative_mean", "bmgG_Conservative_std",
]).set_index("Model")

# Optional: a pretty MultiIndex columns view
df_pretty = df.copy()
df_pretty.columns = pd.MultiIndex.from_tuples([
    ("bmG","Standard","mean"), ("bmG","Standard","std"),
    ("bmG","Conservative","mean"), ("bmG","Conservative","std"),
    ("bmgG","Standard","mean"), ("bmgG","Standard","std"),
    ("bmgG","Conservative","mean"), ("bmgG","Conservative","std"),
])

rows = [
    ("Classifier",               0.1067, 0.05, 0.0320, 0.02, 0.1413, 0.05, 0.0720, 0.02),
    ("Classifier-SMOTE",         0.1173, 0.05, 0.0373, 0.03, 0.1413, 0.06, 0.0720, 0.02),
    ("IQS",                      0.0000, 0.00, 0.0000, 0.00, 0.0000, 0.00, 0.0000, 0.00),
    ("IQS-SMOTE",                0.4000, 0.06, 0.1440, 0.07, 0.5920, 0.11, 0.2747, 0.09),
    ("IQS-CS-SMOTE",             0.2960, 0.08, 0.1093, 0.04, 0.4293, 0.09, 0.2053, 0.06),
    ("Model-based IQS",          0.3307, 0.08, 0.1040, 0.05, 0.4080, 0.08, 0.1680, 0.07),
    ("Model-based IQS-SMOTE",    0.2667, 0.07, 0.0987, 0.04, 0.4107, 0.10, 0.1973, 0.05),
    ("Model-based IQS-CS-SMOTE", 0.1440, 0.04, 0.0720, 0.05, 0.2720, 0.04, 0.1360, 0.05),
    ("DO-IQS",                   0.2880, 0.06, 0.1040, 0.04, 0.4053, 0.09, 0.1920, 0.07),
    ("DO-IQS-LB",                0.0640, 0.03, 0.0400, 0.03, 0.1440, 0.02, 0.0880, 0.03),
]

df = pd.DataFrame(rows, columns=[
    "Model",
    "bmG_Standard_mean", "bmG_Standard_std",
    "bmG_Conservative_mean", "bmG_Conservative_std",
    "bmgG_Standard_mean", "bmgG_Standard_std",
    "bmgG_Conservative_mean", "bmgG_Conservative_std",
]).set_index("Model")

# Optional pretty MultiIndex columns
df_pretty1 = df.copy()
df_pretty1.columns = pd.MultiIndex.from_tuples([
    ("bmG",  "Standard",     "mean"), ("bmG",  "Standard",     "std"),
    ("bmG",  "Conservative", "mean"), ("bmG",  "Conservative", "std"),
    ("bmgG", "Standard",     "mean"), ("bmgG", "Standard",     "std"),
    ("bmgG", "Conservative", "mean"), ("bmgG", "Conservative", "std"),
])


models = [ 'IQS-SMOTE',
        'IQS-CS-SMOTE', 'Model-based IQS',
        'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE',
        'DO-IQS', 'DO-IQS-LB']
a = df_pretty.radial.Standard['mean'].values[3:]
b = df_pretty1.radial.Standard['mean'].values[3:]
c = df_pretty.radial.Standard['std'].values[3:]
c1 = c
d=df_pretty1.radial.Standard['std'].values[3:]
d1=d
fig, ax = plt.subplots(figsize=(25,15))
ax.scatter(b, a, c='blue', s=60)
for i, col in enumerate(models):
    markers, caps, bars = ax.errorbar(b[i], a[i], 
            xerr=np.vstack([d[i],d1[i]]), 
            yerr=np.vstack([c[i],c1[i]]), 
            fmt='o',
            c='blue',
            elinewidth=5,
            capsize=10, 
            capthick=5)
    [bar.set_alpha(0.75) for bar in bars]
    [cap.set_alpha(0.75) for cap in caps]


a = df_pretty.radial.Conservative['mean'].values[3:]
b = df_pretty1.radial.Conservative['mean'].values[3:]
c = df_pretty.radial.Conservative['std'].values[3:]
c1 = c
d=df_pretty1.radial.Conservative['std'].values[3:]
d1=d

ax.scatter(b, a, c='red', s=60)
for i, col in enumerate(models):
    markers, caps, bars = ax.errorbar(b[i], a[i], 
            xerr=np.vstack([d[i],d1[i]]), 
            yerr=np.vstack([c[i],c1[i]]), 
            fmt='o',
            c='red',
            elinewidth=5,
            capsize=10, 
            capthick=5)
    [bar.set_alpha(0.75) for bar in bars]
    [cap.set_alpha(0.75) for cap in caps]
ax.set_xlabel('MEMR', size=60)
ax.set_ylabel('MTTE', size=60)
ax.set_xlim((0,0.5))
ax.tick_params(axis='both', which='major', labelsize=55)

list_of_pathes = [mpatches.Patch(color='blue', label='Standard'),
                  mpatches.Patch(color='red', label= 'Conservative')]
plt.legend().set_visible(False)
plt.title('radial', fontsize=70)
plt.legend(handles=list_of_pathes, fontsize=50)
plt.tight_layout()
plt.show()