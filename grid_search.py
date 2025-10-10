# %%
from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from forward_algorithms.simulate_expert_data import Simulation
from inverse_opt_stopping.iq_learn_base import IQ_Agent 
import time
import os 
import shutil


import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
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
import shutil
SEEDS=[0]
from sklearn.model_selection import ParameterGrid


bm_sym = Simulation(problem='oti')
# bm_sym = Simulation(problem='bessel2')
bm_expert_data = bm_sym.simulate_expert(episodes=1500, max_path_length=200)
bm_test_data = bm_sym.simulate_test(episodes=500, max_path_length=200)
# %%
st = bm_expert_data['state_mem']
act = bm_expert_data['action_mem']
plt.figure(figsize=(15,15))
plt.scatter(st[act==1,0],st[act==1,1], c='green', label='a=1', s=10)
plt.scatter(st[act==0,0],st[act==0,1], c='red', label='a=0', s=50)
plt.xlabel('X1',fontsize=25)
plt.ylabel('X2',fontsize=25)
plt.legend(fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Bessel(2) with radial stopping',fontsize=30)
plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.tight_layout()
plt.show()
# %%
# # %%
# # OTI example, downsample to every 5 sec.
# # batch size 75, n_epochs=35, crossval_split=2
# # q
# # 'target': 0.8422 || 'q_lr': 0.0729
# # q, SMOTE
# # 'target': 0.9736 || 'q_lr': 0.0621
# # q, CS-SMOTE
# # 'target': 0.9756 || 'q_lr': 0.0747
# # ----
# # q, env
# # 'target': 0.7757 || 'env_lr': 0.0496, 'q_lr': 0.0205
# # q, env, SMOTE
# # 'target': 0.9828 || 'env_lr': 0.0265, 'q_lr': 0.0266
# # q, env, CS-SMOTE
# # 'target': 0.9656 || 'env_lr': 0.0227, 'q_lr': 0.0306
# # q, env, g


# # CONSERVATIVE VERSION
# # q, conserv
# # 'target': 0.8506 || 'out_thresh': 0.0002, 'q_lr': 0.0589
# # q, SMOTE, conserv
# #  'target': 0.9268 || 'out_thresh': 0.0001, 'q_lr': 0.0879
# # q, CS-SMOTE, conserv
# # 
# # ---
# # q, env, consverv
# # 'target': 0.8277 || 'out_thresh': 0.0001
# # q, env, SMOTE, conserv
# # 'target': 0.9364 || 'env_lr': 0.0549, 'out_thresh': 0.0001, 'q_lr': 0.09163
# # q, env, CS-SMOTE, conserv
# # 'target': 0.9371 || 'env_lr': 0.0394, 'out_thresh': 0.0001, 'q_lr': 0.0852
# # q, env, g, conserv
# # 
# from bayes_opt import BayesianOptimization
# from bayes_opt import SequentialDomainReductionTransformer
# from bayes_opt import BayesianOptimization
# from bayes_opt import acquisition
# import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# import warnings

# class ThompsonSampling(acquisition.AcquisitionFunction):
#     def __init__(self, random_state=None):
#         super().__init__(random_state)

#     def base_acq(self, y_mean, y_cov):
#         assert y_cov.shape[0] == y_cov.shape[1], "y_cov must be a square matrix."
#         return self.random_state.multivariate_normal(y_mean, y_cov)

#     def _get_acq(self, gp, constraint=None):
#         if constraint is not None:
#             msg = (
#                 f"Received constraints, but acquisition function {type(self)} "
#                 + "does not support constrained optimization."
#             )
#             raise acquisition.ConstraintNotSupportedError(msg)

#         # overwrite the base method since we require cov not std
#         dim = gp.X_train_.shape[1]
#         def acq(x):
#             x = x.reshape(-1, dim)
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 mean, cov = gp.predict(x, return_cov=True)
#             return -1 * self.base_acq(mean, cov)
#         return acq

#     def suggest(self, gp, target_space, n_random=1000, n_l_bfgs_b=0, fit_gp: bool = True):
#         # reduce n_random and n_l_bfgs_b to reduce the computational load
#         return super().suggest(gp, target_space, n_random, n_l_bfgs_b, fit_gp)

# class ExpectedImprovementPerUnitCost(acquisition.ExpectedImprovement):
#     def __init__(self, xi, exploration_decay=None, exploration_decay_delay=None, random_state=None) -> None:
#         super().__init__(xi, exploration_decay, exploration_decay_delay, random_state)
#         self.last_x = None

#     def cost(self, x):
#         if self.last_x is None:
#             return 1
#         return np.mean((self.last_x - np.atleast_2d(x))**2, axis=1) + 1.

#     def _get_acq(self, gp, constraint=None):
#         super_acq = super()._get_acq(gp, constraint)
#         acq = lambda x: super_acq(x) / self.cost(x)
#         return acq

#     def suggest(self, gp, target_space, n_random=10000, n_l_bfgs_b=10, fit_gp: bool = True):
#         # let's get the most recently evaluated point from the target_space
#         self.last_x = target_space.params[-1]
#         return super().suggest(gp, target_space, n_random, n_l_bfgs_b, fit_gp)

# acquisition_function = ThompsonSampling(random_state=2024)
# bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.0,
#                                                           )
# import time
# from binning import MDLP_Discretizer
# discretizer = MDLP_Discretizer(features=np.arange(bm_expert_data['state_mem'].shape[1]))
# discretizer.fit(bm_expert_data['state_mem'], bm_expert_data['action_mem'])
# # # {'target': np.float64(0.7570490536886829), 'params': {'out_thresh': np.float64(0.0006958053401972711)}}
# # {'target': np.float64(0.9480851063829787), 'params': {'out_thresh': np.float64(0.00010930622274778434)}}

# # BM 2dtime to train-test
# # iq: 1428.77 - 11.10
# # iq_smote: 1353.58 - 10.85
# # p_iq: 1389.53 - 8.85
# # p_iq smote: 1336.76 - 9.77
# # diqs: 4880.26-11.48

# # OTI time to train-test
# # iq: 
# # iq_smote: 
# # p_iq: 
# # p_iq smote: 
# # diqs: 

# %%
start = time.time()
iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
                        approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE', #'CS-LSMOTE', #'CS-LSMOTE', #'CS-LSMOTE',#'CS-LSMOTE', #oversampling='CS-LSMOTE', #'CS-LSMOTE', 
                        q_lr=0.01, env_lr=0.01, g_lr=0.01,
                        epsilon=0.1,seed=SEEDS[0],
                        divergence='hellinger', cross_val_splits=2,
                        conservative=False, 
                        q_entropy=False,
                        # lookback=128,
                        # discretiser=discretizer, 
                        # out_thresh=0.001
                        )
iq_diqs.train(mem=bm_expert_data, batch_size=128*1, n_epoches=125)
end = time.time()
print('Training takes {:.2f} seconds '.format(end - start))

start = time.time()
iq_diqs.test(test_memory=bm_test_data)
end = time.time()
print('Testing takes {:.2f} seconds '.format(end - start))


start = time.time()
iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
                        approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE', #'CS-LSMOTE', #'CS-LSMOTE', #'CS-LSMOTE',#'CS-LSMOTE', #oversampling='CS-LSMOTE', #'CS-LSMOTE', 
                        q_lr=0.01, env_lr=0.01, g_lr=0.01,
                        epsilon=0.1,seed=SEEDS[0],
                        divergence='hellinger', cross_val_splits=2,
                        conservative=False, 
                        q_entropy=True
                        # lookback=128,
                        # discretiser=discretizer, 
                        # out_thresh=0.001
                        )
iq_diqs.train(mem=bm_expert_data, batch_size=128*1, n_epoches=125)
end = time.time()
print('Training takes {:.2f} seconds '.format(end - start))

start = time.time()
iq_diqs.test(test_memory=bm_test_data)
end = time.time()
print('Testing takes {:.2f} seconds '.format(end - start))
    
# %%

def training(g_lr):
    start = time.time()

    iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
                        approx_g=True, approx_dynamics=False, oversampling='CS-LSMOTE', 
                        q_lr=0.01, env_lr=0.01, g_lr=g_lr,
                        epsilon=0.1,seed=SEEDS[0],
                        divergence='hellinger', cross_val_splits=2,
                        conservative=False, 
                        # lookback=50,
                        discretiser=discretizer, 
                        # out_thresh=out_thresh
                        )
    iq_diqs.train(mem=bm_expert_data, batch_size=128, n_epoches=50)
    end = time.time()
    print('Training takes {:.2f} seconds '.format(end - start))
    
    start = time.time()
    iq_diqs.test(test_memory=bm_test_data)
    end = time.time()
    print('Testing takes {:.2f} seconds '.format(end - start))
    
    
    return iq_diqs.best_crossval_score

# def training(out_thresh):
#     iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
#                         approx_g=False, approx_dynamics=True, oversampling=None, 
#                         q_lr=0.01, env_lr=0.01, g_lr=0,
#                         epsilon=0.1,seed=SEEDS[0],
#                         divergence='js', cross_val_splits=2,
#                         conservative=True, 
#                         discretiser=discretizer, 
#                         out_thresh=out_thresh)
#     iq_diqs.train(mem=bm_expert_data, batch_size=75, n_epoches=50)
#     iq_diqs.test(test_memory=bm_test_data)
#     return iq_diqs.best_crossval_score

pbounds = {
    # 'q_lr': (0.0001, 0.01),
    'g_lr': (0.0001, 0.1),
    # 'env_lr': (0.0001, 0.01),
    # 'out_thresh': (0.0, 0.009)
    # 'epsilon': (0.001,1)
    }

# {'target': np.float64(0.870892773424419), 'params': {'env_lr': np.float64(0.07975683130691577), 'g_lr': np.float64(0.05286645190056542), 'q_lr': np.float64(0.008766509154228472)}}
# {'target': np.float64(0.619200461925141), 'params': {'g_lr': np.float64(0.0592134373706444)}}
optimizer = BayesianOptimization(
    f=training,
    pbounds=pbounds,
    verbose=2,  
    bounds_transformer=bounds_transformer,
    acquisition_function=acquisition_function, 
    random_state=2024
)

start = time.time()
optimizer.maximize(init_points=5, n_iter=20)
end = time.time()
print('Bayes optimization takes {:.2f} seconds to tune'.format(end - start))
print(optimizer.max)
# {'target': np.float64(0.8089551831431847), 'params': {'env_lr': np.float64(0.00642203685107314), 'g_lr': np.float64(0.00385093254370194), 'q_lr': np.float64(0.020266690773337794)}}
# {'target': np.float64(0.6163446054750402), 'params': {'env_lr': np.float64(0.029441924492880356), 'g_lr': np.float64(0.003498552650931097), 'q_lr': np.float64(0.009488782805921449)}}
# {'target': np.float64(0.6798733929387195), 'params': {'g_lr': np.float64(0.000887852230736775)}}
# %%
plt.plot(optimizer.space.target, label='f(x)')
plt.ylabel('f(x)')
plt.xlabel('Iteration')
plt.legend()
# plt.title('Acquisition function')
plt.show()

param_names = list(optimizer.max['params'].keys())
for i in range(len(param_names)):
    x_min_bound = [b[i][0] for b in bounds_transformer.bounds]
    x_max_bound = [b[i][1] for b in bounds_transformer.bounds]
    x = [x for x in optimizer.space.params[:,i]]
    bounds_transformers_iteration = list(range(2, len(x)))
    plt.plot(x_min_bound[-20:], label=f'{param_names[i]} lower bound')
    plt.plot(x_max_bound[-20:], label=f'{param_names[i]} upper bound')
    plt.plot(x[-20:], label=f'{param_names[i]}')
    plt.ylabel('Parameter value')
    plt.xlabel('Iteration')
    plt.legend()
    plt.title(f'{param_names[i]}')
    plt.show()   
    
# %%
def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma
N=2000
param_names = list(pbounds.keys())
acquisition_function_ = optimizer.acquisition_function
x_obs = [np.array([[res["params"][param]] for res in optimizer.res]) for param in param_names]
y_obs = np.array([res["target"] for res in optimizer.res])
acquisition_function_._fit_gp(optimizer._gp, optimizer._space)
x_params = np.zeros((len(param_names), N))
for i, param in enumerate(param_names):
    param_bounds = pbounds[param]
    x = np.linspace(param_bounds[0], param_bounds[1], N).reshape(-1, 1)
    x_params[i,:] = x.reshape(1,-1)
mu, sigma = posterior(optimizer, x_params.reshape(-1,len(param_names)))
for i in range(len(param_names)):
    plt.title(param_names[i])
    plt.plot(x_obs[i].flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')

    plt.plot(x_params[i,:], mu, '--', color='k', label='Prediction')

    plt.fill(np.concatenate([x_params[0,:], x_params[i,:][::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    plt.show()

# %%
# from bayes_opt import acquisition
# import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# # Bounded region of parameter space

# def posterior(optimizer, grid):
#     mu, sigma = optimizer._gp.predict(grid, return_std=True)
#     return mu, sigma

# def plot_gp(optimizer, x, param):
#     acquisition_function_ = optimizer.acquisition_function
#     fig = plt.figure(figsize=(16, 10))
#     steps = len(optimizer.space)
#     fig.suptitle(
#         'Gaussian Process and Utility Function After {} Steps'.format(steps),
#         fontsize=30
#     )

#     gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
#     axis = plt.subplot(gs[0])
#     acq = plt.subplot(gs[1])

#     x_obs = np.array([[res["params"][param]] for res in optimizer.res])
#     y_obs = np.array([res["target"] for res in optimizer.res])

#     acquisition_function_._fit_gp(optimizer._gp, optimizer._space)
#     mu, sigma = posterior(optimizer, x)

#     # axis.plot(x, y, linewidth=3, label='Target')
#     axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
#     axis.plot(x, mu, '--', color='k', label='Prediction')

#     axis.fill(np.concatenate([x, x[::-1]]),
#               np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
#         alpha=.6, fc='c', ec='None', label='95% confidence interval')

#     axis.set_xlim((-2, 10))
#     axis.set_ylim((None, None))
#     axis.set_ylabel(f'f({param})', fontdict={'size':20})
#     axis.set_xlabel(f'{param}', fontdict={'size':20})

#     utility = -1 * acquisition_function_._get_acq(gp=optimizer._gp)(x)
#     x = x.flatten()

#     acq.plot(x, utility, label='Utility Function', color='purple')
#     acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
#              label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
#     acq.set_xlim((-2, 10))
#     #acq.set_ylim((0, np.max(utility) + 0.5))
#     acq.set_ylabel('Utility', fontdict={'size':20})
#     acq.set_xlabel(f'{param}', fontdict={'size':20})

#     axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
#     acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
#     return fig, fig.axes

# param_names = list(pbounds.keys())
# x_params = np.zeros(len(param_names), 100)
# for i, param in enumerate(param_names):
#     param_bounds = pbounds[param]
#     x = np.linspace(param_bounds[0], param_bounds[1], 100).reshape(-1, 1)
#     x_params[i,:] = x
# plot_gp(optimizer, x, param)

# # %%


# #     iq_diqs = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
# #                         approx_g=False, approx_dynamics=True, oversampling='SMOTE', 
# #                         q_lr=0.01, env_lr=0.01, g_lr=0,
# #                         epsilon=0.1,seed=SEEDS[0],
# #                         divergence='js', cross_val_splits=2,
# #                         conservative=False, 
# #                         discretiser=discretizer, 
# #                         out_thresh=out_thresh)
# #     iq_diqs.train(mem=bm_expert_data, batch_size=75, n_epoches=25)
# #     iq_diqs.test(test_memory=bm_test_data)
# #     return iq_diqs.best_crossval_score
# # pbounds = {
    
    
# iq_iql = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
#                     approx_g=False, approx_dynamics=True, oversampling='SMOTE', 
#                     q_lr=0.01, env_lr=0.01, g_lr=0, epsilon=0.1, seed=SEEDS[0],
#                     divergence='js',
#                     cross_val_splits=2,
#                     conservative=True, 
#                     discretiser=discretizer, 
#                     out_thresh=0.02918)
# iq_iql.train(mem=bm_expert_data, batch_size=75, n_epoches=25)
# iq_iql.test(test_memory=bm_test_data, 
#             from_grid=True)
# iq_iql.test(test_memory=bm_test_data,
#             from_grid=False)

# iq_iql = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
#                         approx_g=False, approx_dynamics=True, oversampling='SMOTE', 
#                         q_lr=0.01, env_lr=0.01, g_lr=0.0, epsilon=0.1, seed=SEEDS[0],
#                         divergence='js',
#                         cross_val_splits=2,
#                         conservative=False)
# iq_iql.train(mem=bm_expert_data, batch_size=75, n_epoches=25)
# iq_iql.test(test_memory=bm_test_data, 
#             from_grid=True)
# iq_iql.test(test_memory=bm_test_data, 
#             from_grid=False)

# plt.show()



# # %%

# %%
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
driver_roc_auc = np.load('outputs/oti_turbofan_10_balanced_acc.npy')
bm_roc_auc = np.load('outputs/oti_turbofan_10_balanced_acc.npy')
# driver_roc_auc = np.load('outputs/oti_azure_10_balanced_acc.npy')
# bm_roc_auc = np.load('outputs/oti_azure_10_balanced_acc.npy')
# nmdp_roc_auc = np.load('outputs/bm_my_ygG_mtte.npy')
# oti_roc_auc = np.load('outputs/bmG_mtte.npy')
# #oti_roc_auc = np.delete(oti_roc_auc,1,1)
# oti_roc_auc2 = np.load('outputs/bmgG_mtte.npy')
# oti_roc_auc = driver_roc_auc
df_rocauc_scores = pd.DataFrame({'example': np.array([np.repeat('BM G only', 5*8),
                                                      np.repeat('BM g&G',5*8), 
                                              # np.repeat('y,g and G',70), np.repeat('G',70), np.repeat('y and G',70)
                                              ]).flatten(),
                                  'balanced_accuracy': np.stack([driver_roc_auc[:,:8],
                                                            bm_roc_auc[:,:8], 
                                                        # nmdp_roc_auc, oti_roc_auc,oti_roc_auc2
                                                        ]).flatten(),
                                  'model': np.repeat(np.array(['IQ-Learning','IQ-SMOTE','IQ-CS-SMOTE','Model-based IQ', 'Model-based IQ-SMOTE', 'Model-based IQ-CS-SOMTE', 'DIQS', 'DIQS-CS-LSMOTE',
                                                              #  'IQ-Learning-Conservative','IQ-SMOTE-Conservative','IQ-CS-SMOTE-Conservative','Model-based IQS-Conservative', 
                                                              #  'IQS-SMOTE-Conservative', 'IQS-CS-SMOMTE-Conservative', 'DIQS-Conservative'
                                                            ]).reshape(-1,1), 5*2,1).T.flatten()})
sns.catplot(df_rocauc_scores,kind='bar', x="example", y="balanced_accuracy", hue='model',
            height=4, aspect=1.5, legend=True, palette='viridis_r',errorbar=('sd',2))
plt.title('Balanced accuracy for 2d BM examples')
plt.show()
# %%
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.axis('on')
list_of_models = [1,2,4,5,6]
examples = [
    # 'car_1',
            'azure_10', 
            'nasa_turbofan_10', 
            'bmG_1', 
            'bmgG_1'

            ]
for example in examples: 
    if example=='nasa_turbofan_10' or example=='car_1':
        list_of_models = [1,2,4,5]
    else:
        list_of_models = [1,2,4,5,6]
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
    models_orig = ['IQS','IQS-SMOTE','IQS-CS-SMOTE','Model-based IQS', 'Model-based IQS-SMOTE', 'Model-based IQS-CS-SMOTE', 'DO-IQS', 'DO-IQS-CS-LSMOTE']
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

    
    driver_roc_auc = np.load(f'outputs/{example}_balanced_acc.npy')
    # bm_roc_auc = np.load('outputs/{example}_balanced_acc.npy')

    df_rocauc_scores = pd.DataFrame({'example': np.array([np.repeat('', 5*len(list_of_models)),
                                                        # np.repeat('BM g&G',5*8), 
                                                # np.repeat('y,g and G',70), np.repeat('G',70), np.repeat('y and G',70)
                                                ]).flatten(),
                                    'balanced_accuracy': np.stack([driver_roc_auc[:,list_of_models]
                                                            # nmdp_roc_auc, oti_roc_auc,oti_roc_auc2
                                                            ]).flatten(),
                                    'model': np.repeat(np.array(models).reshape(-1,1), 5*1,1).T.flatten()})
    # sns.catplot(df_rocauc_scores,kind='bar', x="example", y="balanced_accuracy", hue='model',
    #             height=4, aspect=1.5, legend=True, palette='viridis_r',errorbar=('sd',2))
    plt.figure(figsize=(25,22))
    v = sns.violinplot(data=df_rocauc_scores, x="model", y="balanced_accuracy",inner='quart', 
                    palette=list_of_colours, density_norm='count',
                     inner_kws=dict(color="black", linewidth=5, ls='-'), saturation=1,
                    )
    v.set_xlabel("Balanced Accuracy",fontsize=60)
    v.set_ylabel(" ",fontsize=60)
    v.set_ylim(top=1)
    v.tick_params(labelsize=55,axis='x', labelrotation=90)
    v.tick_params(labelsize=55,axis='y')
    plt.title(example, fontsize=60)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'plots/{example}_bacc.png', bbox_inches = 'tight')

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
# %%
import numpy as np
import PIL
from PIL import Image
from PIL import ImageOps
examples = [
            # 'car_1',
            'bmG_1', 
            'bmgG_1',
            'azure_10', 
            'nasa_turbofan_10', 
            ]
list_im = [f'plots/{examples[i]}_tradeoff.png' for i in range(len(examples))]
imgs    = [ Image.open(i) for i in list_im ]
# imgs = [ImageOps.pad(imgs[i], (imgs[i].size[0]+20, imgs[i].size[1]+20)) for i in range(len(imgs))]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs], reverse = True)[0][1]
imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
#
# # save that beautiful picture
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save(f'plots/trifecta_tradeoff.png' )    

list_im = [f'plots/{examples[i]}_bacc.png' for i in range(len(examples))]
imgs    = [ Image.open(i) for i in list_im ]
# imgs = [ImageOps.pad(imgs[i], (imgs[i].size[0]+20, imgs[i].size[1]+20)) for i in range(len(imgs))]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs], reverse = True)[0][1]
imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
#
# # save that beautiful picture
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save(f'plots/trifecta_bacc.png' )    

list_im = ['plots/trifecta_tradeoff.png', 'plots/trifecta_bacc.png']
imgs    = [ Image.open(i) for i in list_im ]
# imgs = [ImageOps.pad(imgs[i], (imgs[i].size[0]+20, imgs[i].size[1]+20)) for i in range(len(imgs))]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs], reverse = True)[0][1]
imgs_comb = np.vstack([i for i in imgs])
#
# # save that beautiful picture
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save(f'plots/experiments.png' )    


# %%
