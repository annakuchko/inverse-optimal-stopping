from sklearn import tree
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)

from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from forward_algorithms.simulate_expert_data import Simulation
from inverse_opt_stopping.iq_learn_unified import IQ_Agent, plot_st_reg_car
import os
import shutil
sns.set_style("whitegrid")
bm_sym = Simulation(problem='star', total_n=500)
# bm_sym = Simulation(problem='symm_bm_gG', total_n=250)
# bm_sym = Simulation(problem='azure')
bm_expert_data = bm_sym.simulate_expert(episodes=250, max_path_length=100)
bm_test_data = bm_sym.simulate_test(episodes=100, max_path_length=100)
X = bm_expert_data['state_mem']
y = bm_expert_data['action_mem']

Xt = bm_test_data['state_mem']
yt = bm_test_data['action_mem']
# %%
from binning import CART_Discretizer
cartd = CART_Discretizer()
cartd.fit(X,y)

# %%
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    min_samples_split=2, 
    min_samples_leaf=1,
    class_weight='balanced',
    # max_depth=5
    )
clf = clf.fit(X, y)

fig = plt.figure(figsize=(50,30))
_ = tree.plot_tree(clf,
                   feature_names=['x'+str(i) for i in range(X.shape[1])],
                   class_names=['stop','cont'],
                   filled=True, 
                   max_depth=5
                   )
# %%
import os
# os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'
import dtreeviz 

viz = dtreeviz.model(clf, X, y,
                     target_name='action',
                feature_names=['x'+str(i) for i in range(X.shape[1])],
                class_names=['stop','cont'],
                
                )

# %%
# viz.view()
v = viz.view(orientation="LR",scale=1,
             depth_range_to_display=(0, 4))     # render as SVG into internal object 
v.show()
# %%
from cairosvg import svg2png
svg2png(v.svg(),write_to='tree_star_ex5000.png', dpi=5000)
# %%
viz.ctree_feature_space(show={'splits','title'}, 
                          features=['x2'],
                          gtype='barstacked',
                          # nbins=40,
                          # elev=30, azim=20,
                          )

# %%
import dtreeviz
# from dtreeviz import decision_boundaries
from dtree_boundaries import decision_boundaries
fig,axes = plt.subplots(1,2, figsize=(8,3.8), dpi=300)
decision_boundaries(clf, X, y, ax=axes[0],
       feature_names=['x'+str(i) for i in range(X.shape[1])],
       class_names=['stop','cont'],
       )
decision_boundaries(clf, X, y, ax=axes[1],
       # show classification regions not probabilities
       show=['instances', 'boundaries', 'misclassified'], 
       feature_names=['x'+str(i) for i in range(X.shape[1])],
       class_names=['stop','cont'],
       )
plt.show()


# v.save("/tmp/iris.svg")
# tree.plot_tree(clf)
# import graphviz
# dot_data = tree.export_graphviz(clf, class_names=['0', '1'], out_file=None)
# graph = graphviz.Source(dot_data)
# graph
