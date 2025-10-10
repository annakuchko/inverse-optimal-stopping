
# %%
from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from forward_algorithms.simulate_expert_data import Simulation
# from inverse_opt_stopping.iq_learn_base import IQ_Agent, plot_st_reg_car

from inverse_opt_stopping.iq_learn_baase_car import IQ_Agent, plot_st_reg_car
import os
import shutil
THRESH = [0.0,0.005]
# THREH = [0.005]
sns.set_style("whitegrid")
n_paths = 50
# azure_ep200_BA.png
# cp1_ep200_BA.png
for ex in [
#             'azure', 
#             'nasa_turbofan', 
#             'bmG', 
#             'bmgG',
            'car',
           # 'radial',
           #  'star',
           #  'CP1',
            # 'CP2',
            # 'CP3',

            ]:
    
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
    # out_thresh = 0.0001
    EPS = 0.01
    GAMMA = 0.99
    IS_CS = False
    conservative = True
    dqs_only = []
    npaths_MTTE = []
    npaths_MEMR = []
    npaths_BALANCED_ACC = []
    for thresh in THRESH:
        print(f'thresh: {thresh}')
        
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
        for s in SEEDS:
            
         
            # Symmetrical BM resultse
            if ex=='radial':
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='radial', total_n=n_paths)
            elif ex=='star': 
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='star', total_n=n_paths)
            elif ex=='CP1':
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='CP1', total_n=n_paths)
            elif ex=='CP2':
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='CP2', total_n=n_paths)
            elif ex=='CP3':
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='CP3', total_n=n_paths)
            elif ex=='bmG':
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='symm_bm_G', total_n=250)
            elif ex=='bmgG':
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='symm_bm_gG', total_n=250)
            elif ex=='car':
                T_DOWNSAMPLE=1
                bm_sym = Simulation(problem='car', total_n=n_paths)
            elif ex=='azure':
                bm_sym = Simulation(problem='azure')
                T_DOWNSAMPLE=10
            elif ex=='nasa_turbofan':
                bm_sym = Simulation(problem='nasa_turbofan')
                T_DOWNSAMPLE=10
            
            bm_expert_data = bm_sym.simulate_expert(episodes=n_paths, max_path_length=150)
            bm_test_data = bm_sym.simulate_test(episodes=75, max_path_length=150)
            print(f'RANDOM SEED: {s}')
            import matplotlib
            list_of_times = list(np.unique(bm_expert_data['time_ids']))
            cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
            gradient = np.arange(0, 1, 1/len(list_of_times))
            list_of_colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(list_of_times)]
            from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
            
            if conservative:
                from binning import CART_Discretizer
                discretizer = CART_Discretizer()
                discretizer.fit(bm_expert_data['state_mem'], bm_expert_data['action_mem'])
            else:
                discretizer=None
            
            print(f'Start training')
            # IQ-Learn
            # print('Classifier')
            # iq_class = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
            #                     approx_g=False, approx_dynamics=False, oversampling=False,
            #                     q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
            #                     epsilon=EPS,seed=s,
            #                     divergence='hellinger', cross_val_splits=2,
            #                     conservative=conservative,
            #                     q_entropy=False,
            #                     discretiser=discretizer,
            #                     out_thresh=thresh,
            #                     SMOTE_K=SMOTE_K,
            #                     is_cs=IS_CS,
            #                     classify=True,
            #                     gamma=GAMMA)
            # iq_class.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_class.test(test_memory=bm_test_data, from_grid=False)
            # # iq.test(test_memory=bm_test_data, from_grid=True)
           
            print('Classifier with SMOTE oversampling')
            iq_class_SMOTE = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                                q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                discretiser=discretizer,
                                out_thresh=thresh,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                classify=True,
                                gamma=GAMMA)
            iq_class_SMOTE.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=False)
            iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=True)
          
            iq_class_SMOTE = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                                q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                discretiser=discretizer,
                                out_thresh=0.005,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                classify=True,
                                gamma=GAMMA)
            iq_class_SMOTE.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=False)
            iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=True)
            # IQ-Learn
            print('IQ-Learn')
            iq = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0.01,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                discretiser=discretizer,
                                out_thresh=thresh,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                classify=False,
                                gamma=GAMMA)
            iq.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq.test(test_memory=bm_test_data, from_grid=True)
            
            iq = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0.01,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                discretiser=discretizer,
                                out_thresh=0.005,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                classify=False,
                                gamma=GAMMA)
            iq.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq.test(test_memory=bm_test_data, from_grid=True)
            
           
            # # shutil.rmtree('./out', ignore_errors=True)
           
            # # IQ-Learning + oversampling SMOTE
            # print('IQ-Learning + oversampling SMOTE')
            # iq_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
            #                     approx_g=False, approx_dynamics=False, oversampling='SMOTE',
            #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
            #                     epsilon=EPS,seed=s,
            #                     divergence='hellinger', cross_val_splits=2,
            #                     conservative=conservative,
            #                     SMOTE_K=SMOTE_K,
            #                     is_cs=IS_CS,
            #                     gamma=GAMMA,
            #                     discretiser=discretizer,
            #                     out_thresh=thresh
            #                     )
            # iq_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_smote.test(test_memory=bm_test_data, from_grid=False)
            
            print('IQ-Learning + oversampling CS-SMOTE')
            iq_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                gamma=GAMMA,
                                discretiser=discretizer,
                                out_thresh=thresh
                                )
            iq_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
            iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
            
            iq_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                gamma=GAMMA,
                                discretiser=discretizer,
                                out_thresh=0.005
                                )
            iq_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
            iq_cs_smote.test(test_memory=bm_test_data, from_grid=True)
            # print('Approximating P, no oversampling')
            # iq_p = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
            #                     approx_g=False, approx_dynamics=True, oversampling=None,
            #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
            #                     epsilon=EPS,seed=s,
            #                     divergence='hellinger', cross_val_splits=2,
            #                     conservative=conservative,
            #                     q_entropy=False,
            #                     SMOTE_K=SMOTE_K,
            #                     is_cs=IS_CS,
            #                     gamma=GAMMA,
            #                     discretiser=discretizer,
            #                     out_thresh=thresh
            #                     )
            # iq_p.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_p.test(test_memory=bm_test_data, from_grid=False)
            
            # print('Approximating P + oversampling SMOTE')
            # iq_p_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
            #                     approx_g=False, approx_dynamics=True, oversampling='SMOTE',
            #                     q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
            #                     epsilon=EPS,seed=s,
            #                     divergence='hellinger', cross_val_splits=2,
            #                     conservative=conservative,
            #                     SMOTE_K=SMOTE_K,
            #                     is_cs=IS_CS,
            #                     gamma=GAMMA,
            #                     discretiser=discretizer,
            #                     out_thresh=thresh
            #                     )
            # iq_p_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
            
            
            print('Approximating P + oversampling CS-SMOTE')
            iq_p_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                gamma=GAMMA,
                                discretiser=discretizer,
                                out_thresh=thresh
                                )
            iq_p_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
            
            iq_p_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=conservative,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                gamma=GAMMA,
                                discretiser=discretizer,
                                out_thresh=0.005
                                )
            iq_p_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=True)
            # print('Approximating g and P, no oversampling')
            # iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
            #                     approx_g=True, approx_dynamics=True, oversampling=None,
            #                     q_lr=0.01, env_lr=0.01, g_lr=0.001,
            #                     epsilon=EPS,seed=s,
            #                     divergence='hellinger', cross_val_splits=2,
            #                     conservative=conservative,
            #                     q_entropy=False,
            #                     SMOTE_K=SMOTE_K,
            #                     is_cs=IS_CS,
            #                     gamma=GAMMA,
            #                     discretiser=discretizer,
            #                     out_thresh=thresh
            #                     )
            # iq_diqs.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_diqs.test(test_memory=bm_test_data)
        
            # print('Approximating g and P, CS-LSMOTE oversampling')
            # iq_diqs_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
            #                     approx_g=True, approx_dynamics=True, oversampling='CS-LSMOTE',
            #                     q_lr=0.01, env_lr=0.01, g_lr=0.001,
            #                     epsilon=EPS,seed=s,
            #                     divergence='hellinger', cross_val_splits=2,
            #                     conservative=conservative,
            #                     q_entropy=False,
            #                     SMOTE_K=SMOTE_K,
            #                     is_cs=IS_CS,
            #                     gamma=GAMMA,
            #                     discretiser=discretizer,
            #                     out_thresh=thresh
            #                     )
            # iq_diqs_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            # iq_diqs_cs_smote.test(test_memory=bm_test_data)
           

        #     EPOCH_BA.append([
        #         # iq_class.epoch_balanced_accuracy,
        #                   iq_class_SMOTE.epoch_balanced_accuracy,
        #         #           iq.epoch_balanced_accuracy,
        #         #           iq_smote.epoch_balanced_accuracy,
        #                   iq_cs_smote.epoch_balanced_accuracy,
        #         #           iq_p.epoch_balanced_accuracy,
        #         #           iq_p_smote.epoch_balanced_accuracy,
        #                   iq_p_cs_smote.epoch_balanced_accuracy,
        #         #           iq_diqs.epoch_balanced_accuracy,
        #                   iq_diqs_cs_smote.epoch_balanced_accuracy
        #                  ])
        #     EPOCH_MTTE.append([
        #       # iq_class.epoch_mtte,
        #                   iq_class_SMOTE.epoch_mtte,
        #       #             iq.epoch_mtte,
        #       #             iq_smote.epoch_mtte,
        #                   iq_cs_smote.epoch_mtte,
        #       #             iq_p.epoch_mtte,
        #       #             iq_p_smote.epoch_mtte,
        #                   iq_p_cs_smote.epoch_mtte,
        #       #             iq_diqs.epoch_mtte,
        #                   iq_diqs_cs_smote.epoch_mtte
        #                  ])
        #     EPOCH_MEMR.append([
        #       # iq_class.epoch_memr,
        #                   iq_class_SMOTE.epoch_memr,
        #       #             iq.epoch_memr,
        #       #             iq_smote.epoch_memr,
        #                   iq_cs_smote.epoch_memr,
        #       #             iq_p.epoch_memr,
        #       #             iq_p_smote.epoch_memr,
        #                   iq_p_cs_smote.epoch_memr,
        #       #             iq_diqs.epoch_memr,
        #                   iq_diqs_cs_smote.epoch_memr
        #                  ])
        #     EPOCH_LOSS.append([
        #       # iq_class.epoch_loss,
        #                   iq_class_SMOTE.epoch_loss,
        #       #             iq.epoch_loss,
        #       #             iq_smote.epoch_loss,
        #                   iq_cs_smote.epoch_loss,
        #       #             iq_p.epoch_loss,
        #       #             iq_p_smote.epoch_loss,
        #                   iq_p_cs_smote.epoch_loss,
        #       #             iq_diqs.epoch_loss,
        #                   iq_diqs_cs_smote.epoch_loss
        #                  ])
           
           
        #     BM_F1.append([
        #       # iq_class.f1,
        #                   iq_class_SMOTE.f1,
        #       #             iq.f1,
        #       #             iq_smote.f1,
        #                   iq_cs_smote.f1,
        #       #             iq_p.f1,
        #       #             iq_p_smote.f1,
        #                   iq_p_cs_smote.f1,
        #       #             iq_diqs.f1,
        #                   iq_diqs_cs_smote.f1
        #                   # iq_conserv.f1,
        #                   # iq_smote_conserv.f1,
        #                   # iq_cs_smote_conserv.f1,
        #                   # iq_p_conserv.f1,
        #                   # iq_p_smote_conserv.f1,
        #                   # iq_p_cs_smote_conserv.f1,
        #                   # iq_diqs_conserv.f1
        #                  ])
        #     BM_PR_AUC.append([
        #       # iq_class.pr_auc,
        #                   iq_class_SMOTE.pr_auc,
        #       #             iq.pr_auc,
        #       #                 iq_smote.pr_auc,
        #                       iq_cs_smote.pr_auc,
        #       #                 iq_p.pr_auc,
        #       #                 iq_p_smote.pr_auc,
        #                       iq_p_cs_smote.pr_auc,
        #       #                 iq_diqs.pr_auc,
        #                       iq_diqs_cs_smote.pr_auc
        #                   # iq_conserv.pr_auc,
        #                   # iq_smote_conserv.pr_auc,
        #                   # iq_cs_smote_conserv.pr_auc,
        #                   # iq_p_conserv.pr_auc,
        #                   # iq_p_smote_conserv.pr_auc,
        #                   # iq_p_cs_smote_conserv.pr_auc,
        #                   # iq_diqs_conserv.pr_auc
        #                   ])    
        #     BM_MTTE.append([
        #       # iq_class.mtte,
        #                   iq_class_SMOTE.mtte,
        #       #             iq.mtte,
        #       #               iq_smote.mtte,
        #                     iq_cs_smote.mtte,
        #       #               iq_p.mtte,
        #       #               iq_p_smote.mtte,
        #                     iq_p_cs_smote.mtte,
        #       #               iq_diqs.mtte,
        #                     iq_diqs_cs_smote.mtte
        #                   # iq_conserv.mtte,
        #                   # iq_smote_conserv.mtte,
        #                   # iq_cs_smote_conserv.mtte,
        #                   # iq_p_conserv.mtte,
        #                   # iq_p_smote_conserv.mtte,
        #                   # iq_p_cs_smote_conserv.mtte,
        #                   # iq_diqs_conserv.mtte
        #                                     ])
        #     BM_MEMR.append([
        #       # iq_class.memr,
        #                   iq_class_SMOTE.memr,
        #       #             iq.memr,
        #       #               iq_smote.memr,
        #                     iq_cs_smote.memr,
        #       #               iq_p.memr,
        #       #               iq_p_smote.memr,
        #                     iq_p_cs_smote.memr,
        #       #               iq_diqs.memr,
        #                     iq_diqs_cs_smote.memr
        #                   # iq_conserv.memr,
        #                   # iq_smote_conserv.memr,
        #                   # iq_cs_smote_conserv.memr,
        #                   # iq_p_conserv.memr,
        #                   # iq_p_smote_conserv.memr,
        #                   # iq_p_cs_smote_conserv.memr,
        #                   # iq_diqs_conserv.memr
        #                     ])
        #     BM_BALANCED_ACC.append([
        #       # iq_class.balanced_accuracy,
        #                   iq_class_SMOTE.balanced_accuracy,
        #       #             iq.balanced_accuracy,
        #       #             iq_smote.balanced_accuracy,
        #                   iq_cs_smote.balanced_accuracy,
        #       #             iq_p.balanced_accuracy,
        #       #             iq_p_smote.balanced_accuracy,
        #                   iq_p_cs_smote.balanced_accuracy,
        #       #             iq_diqs.balanced_accuracy,
        #                   iq_diqs_cs_smote.balanced_accuracy
        #                   # iq_conserv.balanced_accuracy,
        #                   # iq_smote_conserv.balanced_accuracy,
        #                   # iq_cs_smote_conserv.balanced_accuracy,
        #                   # iq_p_conserv.balanced_accuracy,
        #                   # iq_p_smote_conserv.balanced_accuracy,
        #                   # iq_p_cs_smote_conserv.balanced_accuracy,
        #                   # iq_diqs_conserv.balanced_accuracy
        #                     ])
           

    
        # npaths_MTTE.append(BM_MTTE)
        # npaths_MEMR.append(BM_MEMR)
        # npaths_BALANCED_ACC.append(BM_BALANCED_ACC)
        
    models_orig = [
        # 'Classifier',
        'Classifier-SMOTE',
        #     'IQS', 
        #     'IQS-SMOTE',
            'IQS-CS-SMOTE', 
        #     'Model-based IQS',
        #     'Model-based IQS-SMOTE', 
            'Model-based IQS-CS-SMOTE',
        #     'DO-IQS',
            'DO-IQS-LB']
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
    list_of_models = [0,1,2,3]
    # gradient = np.linspace(0, 1, len(models_orig))[list_of_models]
    gradient = np.arange(0, 1, 1/len(models_orig))[list_of_models]
    list_of_colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(models_orig)]
    plt.figure(figsize=(40,20))
    for j, model in enumerate(models_orig):
        col = list_of_colours[j]
        a=[]
        b=[]
        c=[]
        d=[]
        for i, thresh in enumerate(THRESH):
            iq_ba = [npaths_BALANCED_ACC[i][k][j] for k in range(len(npaths_BALANCED_ACC[i]))]
            b.append(np.mean(iq_ba))
            a.append(i)
            c.append(np.std(iq_ba)*2)
            # d.append(np.quantile(iq_ba, .75))
        plt.errorbar(a, b, yerr=c, fmt="r--o", color=col, capsize=25, elinewidth=8,markersize=20)
    plt.title(ex, fontsize=80)
    plt.yticks(fontsize=55,)
    plt.xticks(ticks=a, labels=THRESH,fontsize=55, rotation=90)
    plt.xlabel('threshold', fontsize=80)
    plt.ylabel('BA', fontsize=80)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(40,20))
    for j, model in enumerate(models_orig):
        col = list_of_colours[j]
        a=[]
        b=[]
        c=[]
        d=[]
        for i, thresh in enumerate(THRESH):
            iq_ba = [npaths_MTTE[i][k][j] for k in range(len(npaths_MTTE[i]))]
            b.append(np.mean(iq_ba))
            a.append(i)
            c.append(np.std(iq_ba)*2)
            # d.append(np.quantile(iq_ba, .75))
        plt.errorbar(a, b, yerr=c, fmt="r--o", color=col, capsize=25, elinewidth=8,markersize=20)
    plt.title(ex, fontsize=80)
    plt.yticks(fontsize=55,)
    plt.xticks(ticks=a, labels=THRESH,fontsize=55, rotation=90)
    plt.xlabel('threshold', fontsize=80)
    plt.ylabel('m-TTE', fontsize=80)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(40,20))
    for j, model in enumerate(models_orig):
        col = list_of_colours[j]
        a=[]
        b=[]
        c=[]
        d=[]
        for i, thresh in enumerate(THRESH):
            iq_ba = [npaths_MEMR[i][k][j] for k in range(len(npaths_MEMR[i]))]
            b.append(np.mean(iq_ba))
            a.append(i)
            c.append(np.std(iq_ba)*2)
            # d.append(np.quantile(iq_ba, .75))
        plt.errorbar(a, b, yerr=c, fmt="r--o", color=col, capsize=25, elinewidth=8,markersize=20)
    plt.title(ex, fontsize=80)
    plt.yticks(fontsize=55,)
    plt.xticks(ticks=a, labels=THRESH,fontsize=55, rotation=90)
    plt.xlabel('threshold', fontsize=80)
    plt.ylabel('m-EMR', fontsize=80)
    plt.tight_layout()
    plt.show()
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




