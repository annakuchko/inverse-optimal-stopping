from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from math import log
from sklearn.base import TransformerMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split

from forward_algorithms.simulate_expert_data import Simulation
from inverse_opt_stopping.iq_learn_unified import IQ_Agent, plot_st_reg_car
import os
import shutil
sns.set_style("whitegrid")
def entropy_numpy(data_classes, base=2):
    '''
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    '''
    if data_classes.size==0:
        return 1 
    else:
        classes = np.unique(data_classes)
        N = len(data_classes)
        ent = 0  # initialize entropy
    
        # iterate over classes
        for c in classes:
            partition = data_classes[data_classes == c]  # data with class = c
            proportion = len(partition) / N
            #update entropy
            ent -= proportion * log(proportion, base)
    
        return ent


# ex = 'radial'
ex = 'star'

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
out_thresh = 0.00
out_thresh1 = 0.000005
out_thresh2 = 0.00005
out_thresh3 = 0.0005
out_thresh4 = 0.005
Q_ENT = True
EPS = 0.1
conservative=True
IS_CS = False
dqs_only = []

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
import matplotlib
list_of_times = list(np.unique(bm_expert_data['time_ids']))
cmap = matplotlib.colormaps.get_cmap('nipy_spectral_r')
gradient = np.arange(0, 1, 1/len(list_of_times))
list_of_colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient][:len(list_of_times)]
X = bm_expert_data['state_mem']
y = bm_expert_data['action_mem']
if conservative:
    from binning import MDLP_Discretizer, CART_Discretizer
    # discretizer = MDLP_Discretizer(features=np.arange(bm_expert_data['state_mem'].shape[1]))
    discretizer = CART_Discretizer()
    # 0-1-2-3-4-5-150
    discretizer.fit(bm_expert_data['state_mem'], bm_expert_data['action_mem'])
else:
    discretizer=None
  
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
   
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from math import log
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy.ma as ma

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy.ma as ma
from sklearn.tree import DecisionTreeClassifier

# def get_feature_bin_edges(tree, add_inf=True):
#     tree_ = tree.tree_
#     feature_thresholds = defaultdict(set)

#     for node_id in range(tree_.node_count):
#         feature = tree_.feature[node_id]
#         threshold = tree_.threshold[node_id]
#         if feature >= 0:
#             feature_thresholds[feature].add(threshold)

#     feature_bins = {}
#     for feature, thresholds in feature_thresholds.items():
#         sorted_edges = np.sort(list(thresholds))
#         if add_inf:
#             sorted_edges = np.concatenate(([-np.inf], sorted_edges, [np.inf]))
#         feature_bins[feature] = sorted_edges

#     return feature_bins

# def expand_inf_edges(edges, data, buffer_ratio=0.05):
#     edges = edges.copy()
#     data_min, data_max = np.min(data), np.max(data)
#     buffer = (data_max - data_min) * buffer_ratio
#     if np.isneginf(edges[0]):
#         edges[0] = data_min - buffer
#     if np.isposinf(edges[-1]):
#         edges[-1] = data_max + buffer
#     return edges

# def plot_all_heatmaps_from_dtc(X, y, clf):
#     # Get and expand bin edges
#     feature_bins = get_feature_bin_edges(clf)
#     x1_edges = expand_inf_edges(feature_bins[0], X[:, 0])
#     x2_edges = expand_inf_edges(feature_bins[1], X[:, 1])
#     t_edges  = expand_inf_edges(feature_bins[2], X[:, 2])

#     x1, x2, t = X[:, 0], X[:, 1], X[:, 2]

#     # Get leaf-level entropy
#     leaf_ids = clf.apply(X)
#     node_entropy = clf.tree_.impurity
#     sample_entropy = node_entropy[leaf_ids]

#     # Digitize all samples once
#     x1_bins = np.digitize(x1, x1_edges) - 1
#     x2_bins = np.digitize(x2, x2_edges) - 1
#     t_bins  = np.digitize(t,  t_edges)  - 1

#     grid_shape = (len(x1_edges) - 1, len(x2_edges) - 1)

#     for t_idx in range(len(t_edges) - 1):
#         mask = t_bins == t_idx
#         x1_t = x1[mask]
#         x2_t = x2[mask]
#         y_t  = y[mask]
#         x1_b = x1_bins[mask]
#         x2_b = x2_bins[mask]
#         ent  = sample_entropy[mask]

#         if len(x1_t) == 0:
#             continue

#         # Initialize object arrays
#         majority_grid = np.empty(grid_shape, dtype=object)
#         entropy_grid = np.empty(grid_shape, dtype=object)
#         count_grid = np.zeros(grid_shape)

#         for i in range(grid_shape[0]):
#             for j in range(grid_shape[1]):
#                 majority_grid[i, j] = []
#                 entropy_grid[i, j] = []

#         # Fill bins
#         for xi, yi, label, e in zip(x1_b, x2_b, y_t, ent):
#             if 0 <= xi < grid_shape[0] and 0 <= yi < grid_shape[1]:
#                 majority_grid[xi, yi].append(label)
#                 entropy_grid[xi, yi].append(e)
#                 count_grid[xi, yi] += 1

#         # Convert object arrays to float arrays
#         majority_final = np.full(grid_shape, np.nan)
#         entropy_final = np.full(grid_shape, np.nan)

#         for i in range(grid_shape[0]):
#             for j in range(grid_shape[1]):
#                 labels = majority_grid[i, j]
#                 entropies = entropy_grid[i, j]
#                 if labels:
#                     counts = np.bincount(labels, minlength=2)
#                     majority_final[i, j] = np.argmax(counts)
#                 if entropies:
#                     entropy_final[i, j] = np.mean(entropies)

#         # Normalize count grid
#         count_grid_norm = count_grid / count_grid.max() if count_grid.max() > 0 else count_grid

#         # === Plotting ===
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#         titles = ['Majority Action (0=blue, 1=white)',
#                   'Entropy from DTC (base 2)',
#                   'Normalized Action Count']
#         grids = [majority_final, entropy_final, count_grid_norm]
#         cmaps = ['Blues_r', 'PuBu_r', 'PuBu_r']
#         vmins = [0, 0, 0]
#         vmaxs = [1, 1, 1]

#         for ax, grid, title, cmap_name, vmin, vmax in zip(axes, grids, titles, cmaps, vmins, vmaxs):
#             cmap = plt.get_cmap(cmap_name).copy()
#             cmap.set_bad(color='lightgray')
#             masked = ma.masked_invalid(grid)
#             c = ax.pcolormesh(x1_edges, x2_edges, masked.T,
#                               cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
#             fig.colorbar(c, ax=ax)
#             ax.set_title(f'{title}\nTime: {t_edges[t_idx]:.2f} to {t_edges[t_idx + 1]:.2f}')
#             ax.set_xlabel('x1')
#             ax.set_ylabel('x2')

#             # Scatter overlay
#             red_mask = y_t == 0
#             green_mask = y_t == 1
#             ax.scatter(x1_t[red_mask], x2_t[red_mask], c='red', s=10, label='y=0')
#             ax.scatter(x1_t[green_mask], x2_t[green_mask], c='green', s=10, label='y=1')

#         plt.tight_layout()
#         plt.show()


# plot_all_heatmaps_from_dtc(X,y, discretizer.dtd)

#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier

def get_feature_bin_edges(tree, add_inf=True):
    tree_ = tree.tree_
    feature_thresholds = defaultdict(set)

    for node_id in range(tree_.node_count):
        feature = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        if feature >= 0:
            feature_thresholds[feature].add(threshold)

    feature_bins = {}
    for feature, thresholds in feature_thresholds.items():
        sorted_edges = np.sort(list(thresholds))
        if add_inf:
            sorted_edges = np.concatenate(([-np.inf], sorted_edges, [np.inf]))
        feature_bins[feature] = sorted_edges

    return feature_bins

def expand_inf_edges(edges, data, buffer_ratio=0.05):
    edges = edges.copy()
    data_min, data_max = np.min(data), np.max(data)
    buffer = (data_max - data_min) * buffer_ratio
    if np.isneginf(edges[0]):
        edges[0] = -2
    if np.isposinf(edges[-1]):
        edges[-1] = 2
    return edges

def expand_inf_edgest(edges, data, buffer_ratio=0.05):
    edges = edges.copy()
    data_min, data_max = np.min(data), np.max(data)
    buffer = (data_max - data_min) * buffer_ratio
    if np.isneginf(edges[0]):
        edges[0] = 0
    if np.isposinf(edges[-1]):
        edges[-1] = 1
    return edges

def plot_normalized_action_count_heatmaps(X, y, clf):
    # Set font sizes globally
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Extract bin edges
    xy_min, xy_max = -2, 2
    feature_bins = get_feature_bin_edges(clf)
    x1_edges_full = expand_inf_edges(feature_bins[0], X[:, 0])
    x2_edges_full = expand_inf_edges(feature_bins[1], X[:, 1])
    t_edges  = expand_inf_edgest(feature_bins[2], X[:, 2])
    
    # Filter edges to within [-2, 2]
    x1_edges = np.clip(x1_edges_full, xy_min, xy_max)
    x2_edges = np.clip(x2_edges_full, xy_min, xy_max)
   
    x1, x2, t = X[:, 0], X[:, 1], X[:, 2]
    x1_bins = np.digitize(x1, x1_edges) - 1
    x2_bins = np.digitize(x2, x2_edges) - 1
    t_bins  = np.digitize(t,  t_edges)  - 1

    grid_shape = (len(x1_edges) - 1, len(x2_edges) - 1)
    total_samples = len(X)

    for t_idx in range(len(t_edges) - 1):
        mask = t_bins == t_idx
        x1_t = x1[mask]
        x2_t = x2[mask]
        y_t  = y[mask]
        x1_b = x1_bins[mask]
        x2_b = x2_bins[mask]

        if len(x1_t) == 0:
            continue

        # Count samples per bin
        count_grid = np.zeros(grid_shape)
        for xi, yi in zip(x1_b, x2_b):
            if 0 <= xi < grid_shape[0] and 0 <= yi < grid_shape[1]:
                count_grid[xi, yi] += 1

        # Normalize by total number of samples
        norm_grid = count_grid / total_samples

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.get_cmap('PuBu_r').copy()
        cmap.set_bad(color='lightgray')
        masked = ma.masked_invalid(norm_grid)

        c = ax.pcolormesh(
            x1_edges, x2_edges, masked.T,
            cmap=cmap, shading='auto', vmin=0, vmax=norm_grid.max()
        )

        cb = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('ROD', fontsize=14)

        ax.set_title(f'ROD\nTime: {t_edges[t_idx]:.2f} to {t_edges[t_idx + 1]:.2f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        # Scatter overlay
        red_mask = y_t == 0
        green_mask = y_t == 1
        ax.scatter(x1_t[red_mask], x2_t[red_mask], c='red', s=20, label='a=0', edgecolors='k')
        ax.scatter(x1_t[green_mask], x2_t[green_mask], c='green', s=20, label='a=1', edgecolors='k')
        ax.legend(loc='upper right')
        # plt.xlim((-2,2))
        plt.tight_layout()
        plt.show()

plot_normalized_action_count_heatmaps(X,y, discretizer.dtd)
