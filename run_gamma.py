
# %%
from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from forward_algorithms.simulate_expert_data import Simulation
from inverse_opt_stopping.iq_learn_base import IQ_Agent, plot_st_reg_car
import os
import shutil
N_PATHS = [50,100,150,250,350,550,750,1000]
sns.set_style("whitegrid")
# azure_ep200_BA.png
# cp1_ep200_BA.png
for ex in [
#             'azure', 
#             'nasa_turbofan', 
#             'bmG', 
#             'bmgG',
            # 'car',
            'radial',
            'star',
            'CP1',
            'CP2',
            'CP3',

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
    out_thresh = 0.0001
    EPS = 0.1
    IS_CS = False
    dqs_only = []
    npaths_MTTE = []
    npaths_MEMR = []
    npaths_BALANCED_ACC = []
    for n_paths in N_PATHS:
        print(f'n_paths: {n_paths}')
        
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
           
            
            print(f'Start training')
            # IQ-Learn
            print('Classifier')
            iq_class = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling=False,
                                q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                # discretiser=discretizer,
                                out_thresh=out_thresh,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                classify=True)
            iq_class.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_class.test(test_memory=bm_test_data, from_grid=False)
            # iq.test(test_memory=bm_test_data, from_grid=True)
           
            print('Classifier with SMOTE oversampling')
            iq_class_SMOTE = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                                q_lr=0.00001, env_lr=ENV_LR, g_lr=0.01,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                # discretiser=discretizer,
                                out_thresh=out_thresh,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                classify=True)
            iq_class_SMOTE.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_class_SMOTE.test(test_memory=bm_test_data, from_grid=False)
           
            # IQ-Learn
            print('IQ-Learn')
            iq = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0.01,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                # discretiser=discretizer,
                                out_thresh=out_thresh,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS,
                                classify=True)
            iq.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq.test(test_memory=bm_test_data, from_grid=False)
            
           
            shutil.rmtree('./out', ignore_errors=True)
           
            # IQ-Learning + oversampling SMOTE
            print('IQ-Learning + oversampling SMOTE')
            iq_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS
                                # discretiser=discretizer,
                                # out_thresh=out_thresh
                                )
            iq_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_smote.test(test_memory=bm_test_data, from_grid=False)
            
            print('IQ-Learning + oversampling CS-SMOTE')
            iq_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS
                                # discretiser=discretizer,
                                # out_thresh=out_thresh
                                )
            iq_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_cs_smote.test(test_memory=bm_test_data, from_grid=False)
            
            print('Approximating P, no oversampling')
            iq_p = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=True, oversampling=None,
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS
                                # discretiser=discretizer,
                                # out_thresh=out_thresh
                                )
            iq_p.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_p.test(test_memory=bm_test_data, from_grid=False)
            
            print('Approximating P + oversampling SMOTE')
            iq_p_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=True, oversampling='SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS
                                # discretiser=discretizer,
                                # out_thresh=out_thresh
                                )
            iq_p_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_p_smote.test(test_memory=bm_test_data, from_grid=False)
            
            
            print('Approximating P + oversampling CS-SMOTE')
            iq_p_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE',
                                q_lr=Q_LR, env_lr=ENV_LR, g_lr=0,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS
                                # discretiser=discretizer,
                                # out_thresh=out_thresh
                                )
            iq_p_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_p_cs_smote.test(test_memory=bm_test_data, from_grid=False)
            
            print('Approximating g and P, no oversampling')
            iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=True, approx_dynamics=True, oversampling=None,
                                q_lr=0.01, env_lr=0.01, g_lr=0.001,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS
                                # discretiser=discretizer,
                                # out_thresh=out_thresh
                                )
            iq_diqs.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_diqs.test(test_memory=bm_test_data)
        
            print('Approximating g and P, CS-LSMOTE oversampling')
            iq_diqs_cs_smote = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0],
                                approx_g=True, approx_dynamics=True, oversampling='CS-LSMOTE',
                                q_lr=0.01, env_lr=0.01, g_lr=0.001,
                                epsilon=EPS,seed=s,
                                divergence='hellinger', cross_val_splits=2,
                                conservative=False,
                                q_entropy=False,
                                SMOTE_K=SMOTE_K,
                                is_cs=IS_CS
                                # discretiser=discretizer,
                                # out_thresh=out_thresh
                                )
            iq_diqs_cs_smote.train(mem=bm_expert_data, batch_size=BATCH_SIZE, n_epoches=N_EPOCHS)
            iq_diqs_cs_smote.test(test_memory=bm_test_data)
           

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
           

    
        npaths_MTTE.append(BM_MTTE)
        npaths_MEMR.append(BM_MEMR)
        npaths_BALANCED_ACC.append(BM_BALANCED_ACC)
        
    models_orig = ['Classifier','Classifier-SMOTE',
            'IQS', 'IQS-SMOTE',
            'IQS-CS-SMOTE', 'Model-based IQS',
            'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE',
            'DO-IQS', 'DO-IQS-LB']
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
    list_of_models = [0,1,2,3,4,5,6,7,8,9]
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
        for i, n_paths in enumerate(N_PATHS):
            iq_ba = [npaths_BALANCED_ACC[i][k][j] for k in range(len(npaths_BALANCED_ACC[i]))]
            b.append(np.mean(iq_ba))
            a.append(i)
            c.append(np.quantile(iq_ba, .25))
            d.append(np.quantile(iq_ba, .75))
        plt.errorbar(a, b, yerr=[c,d], fmt="r--o", color=col, capsize=25, elinewidth=8,markersize=20)
    plt.title(ex)
    plt.yticks(fontsize=55,)
    plt.xticks(ticks=a, labels=N_PATHS,fontsize=55, rotation=90)
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

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.axis('on')
models_orig = ['Classifier','Classifier-SMOTE',
        'IQS', 'IQS-SMOTE',
        'IQS-CS-SMOTE', 'Model-based IQS',
        'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE',
        'DO-IQS', 'DO-IQS-LB']
list_of_models = [0,1,2,3,4,5,6,7,8,9]
examples = [
    'car',
            'azure', 
            'nasa_turbofan', 
            'bmG', 
            'bmgG',
            'radial',
            'star',
            'CP1',
            'CP2',
            'CP3'

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



