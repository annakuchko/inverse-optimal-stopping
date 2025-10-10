# %%
import pandas as pd
import numpy as np
from math import log
# from feature_engine.discretisation import DecisionTreeDiscretiser


from matplotlib.colors import LogNorm
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
 
def random_fourier_features(X, y=None, num_features=500,
                            sigma=1, return_encoded=True, 
                            plot=True, random_seed=42,
                            pca_components=3):
    """
    Generates Random Fourier Features (RFF) to approximate an RBF kernel embedding.
 
    Parameters:
        X (np.ndarray): Input data of shape (N, K)
        y (np.ndarray): Optional labels for plotting
        num_features (int): Number of RFF features to generate
        sigma (float): Bandwidth parameter of the RBF kernel
        return_encoded (bool): If True, returns the RFF-transformed features
        plot (bool): If True and num_features >= 2, plots PCA of the RFFs
        random_seed (int): Seed for reproducibility
 
    Returns:
        np.ndarray: RFF-transformed data of shape (N, num_features)
    """
    N, K = X.shape
    rng = np.random.default_rng(random_seed)
 
    # Sample random projection directions from Gaussian
    W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(K, num_features))
    b = rng.uniform(0, 2 * np.pi, size=(num_features,))
 
    # Compute RFF features
    projection = np.dot(X, W) + b
    Z = np.sqrt(2.0 / num_features) * np.cos(projection)
    pca = PCA(n_components=pca_components)
    Z_2d = pca.fit_transform(Z)
    # Z_2d = Z.copy()
    # Optional plotting
    if plot and num_features >= 2 and y is not None:
        plt.figure(figsize=(6, 5))
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='k')
        plt.title("Random Fourier Features (PCA Projection)")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
 
    if return_encoded:
        # pca = PCA(n_components=pca_components)
        # pca_features = pca.fit_transform(Z)
        return Z_2d
    
 
def hyperspherical_multifrequency_embedding(X, y=None, max_freq=3, 
                                            return_encoded=True, 
                                            plot=True):
    """
    Computes multifrequency sine-cosine hyperspherical embeddings from Cartesian state vectors
    and optionally plots PCA projections.
 
    Parameters:
        X (np.ndarray): Input data of shape (N, K), assumed to be Cartesian state vectors
        y (np.ndarray): Optional labels of shape (N,), for coloring
        max_freq (int): Maximum frequency to use for sin/cos encoding
        return_encoded (bool): If True, returns encoded features
        plot (bool): If True, shows PCA projection of original vs encoded space
 
    Returns:
        encoded (np.ndarray): shape (N, 2*K*max_freq), the encoded feature space
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
 
    N, K = X.shape
 
    # Step 1: Convert Cartesian to hyperspherical angles
    def cartesian_to_hyperspherical0(x):
        out = []
        for vec in x:
            r = np.linalg.norm(vec)
            angles = []
            for i in range(len(vec) - 1):
                norm = np.linalg.norm(vec[i:])
                angle = np.arccos(vec[i] / norm) if norm > 1e-8 else 0.0
                angles.append(angle)
            if vec[-1] < 0:
                angles[-1] = 2 * np.pi - angles[-1]
            out.append(angles)
        return np.array(out)
 
    angles = cartesian_to_hyperspherical0(X)  # shape (N, K-1)
 
    # Step 2: Multi-frequency sine-cosine encoding
    encoded_list = []
    for k in range(1, max_freq + 1):
        encoded_list.append(np.sin(k * angles))
        encoded_list.append(np.cos(k * angles))
 
    encoded = np.concatenate(encoded_list, axis=1)  # shape (N, 2*(K-1)*max_freq)
    
    # enoded = angles
    pca_original = PCA(n_components=2).fit_transform(X)
    pca_encoded = PCA(n_components=2).fit_transform(encoded)
    # pca_original
    # pca_encoded
    # Step 3: Plotting
    if plot and y is not None:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        scatter_args = {'alpha': 0.7, 'edgecolors': 'k'}
 
        axs[0].scatter(pca_original[:, 0], pca_original[:, 1], c=y, cmap='coolwarm', **scatter_args)
        axs[1].scatter(pca_encoded[:, 0], pca_encoded[:, 1], c=y, cmap='coolwarm', **scatter_args)
        
        axs[0].set_title("Original Cartesian Features (PCA Projection)")
        axs[0].set_xlabel("PC 1")
        axs[0].set_ylabel("PC 2")
        axs[0].grid(True)
 
        axs[1].set_title("Hyperspherical Multifrequency Embedding (PCA Projection)")
        axs[1].set_xlabel("PC 1")
        axs[1].set_ylabel("PC 2")
        axs[1].grid(True)
 
        plt.tight_layout()
        plt.show()
 
    if return_encoded:
        # return pca_encoded
        return encoded
    
 
def cartesian_to_hyperspherical(states: np.ndarray, d_cartesian: int, y=None) -> np.ndarray:
    """
    Converts the first `d_cartesian` components of each state vector from Cartesian
    to hyperspherical coordinates, preserving the rest of the state.
 
    Parameters:
    - states: np.ndarray of shape (N, D_total)
    - d_cartesian: int, number of leading components to convert (D_cartesian >= 2)
 
    Returns:
    - np.ndarray of shape (N, D_total), with first `d_cartesian` columns replaced by
      [r, θ1, θ2, ..., θ_{D_cartesian-1}]
    """
    states = np.asarray(states)
    st_orig = states.copy()
    N, D_total = states.shape
 
    if d_cartesian < 2 or d_cartesian > D_total:
        raise ValueError("d_cartesian must be in the range [2, D_total]")
 
    cartesian_part = states[:, :d_cartesian]
    other_part = states[:, d_cartesian:]
    # other_part = states[:, :d_cartesian]
    
    # inner0 = inner_product_embedding(states, y)
    inner0 = random_fourier_features(states, y)
    # inner0 = hyperspherical_multifrequency_embedding(states, y)
    
    spherical = np.hstack([
                           inner0
                           ])
    # result = np.hstack([spherical, other_part])
    result = np.hstack([spherical, other_part])
 
    return result
    # return st_orig
 
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
 
def cut_point_information_gain_numpy(X, y, cut_point):
    '''
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    '''
    entropy_full = entropy_numpy(y)  # compute entropy of full dataset (w/o split)
 
    #split data at cut_point
    data_left_mask = X <= cut_point #dataset[dataset[feature_label] <= cut_point]
    data_right_mask = X > cut_point #dataset[dataset[feature_label] > cut_point]
    (N, N_left, N_right) = (len(X), data_left_mask.sum(), data_right_mask.sum())
 
    gain = entropy_full - (N_left / N) * entropy_numpy(y[data_left_mask]) - \
        (N_right / N) * entropy_numpy(y[data_right_mask])
 
    return gain
 
import numpy as np
from math import log
from sklearn.base import TransformerMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split
 



from sklearn.tree import DecisionTreeClassifier
class CART_Discretizer:
    def __init__(self,):
        self.dtd =  DecisionTreeClassifier(
            criterion='entropy',
            max_depth=35,
            )
        
    def fit(self, X, y):
        # X = cartesian_to_hyperspherical(X.copy(), X.shape[1], y)
        # X_discrete =self.dtd.fit_transform(X, y).values
        self.dtd.fit(X, y)
        X_discrete  = self.dtd.apply(X.copy()).reshape(-1,1)
        unique_ints = np.unique(X_discrete, axis=0)
        k_outs = [sum((X_discrete==unique_ints[i]).all(axis=1))/X_discrete.shape[0] for i in range(len(unique_ints))]
        plt.plot(k_outs)
        plt.show()
        self.ints = unique_ints, k_outs
        
    def transform(self,X):
        # X = cartesian_to_hyperspherical(X.copy(), X.shape[1])
        # return self.dtd.transform(X).values
        return self.dtd.apply(X.copy()).reshape(-1,1)
        
    def fit_transform(self, X,y):
        # X = cartesian_to_hyperspherical(X.copy(), X.shape[1], y)
        self.fit(X,y)
        # return self.transform(X).values
        return self.dtd.apply(X.copy()).reshape(-1,1)
        
def previous_item(a, val):
    idx = np.where(a == val)[0][0] - 1
    return a[idx]
 
class MDLP_Discretizer(TransformerMixin):
    def __init__(self, features=None, raw_data_shape=None):
        '''
        initializes discretizer object:
            saves raw copy of data and creates self._data with only features to discretize and class
            computes initial entropy (before any splitting)
            self._features = features to be discretized
            self._classes = unique classes in raw_data
            self._class_name = label of class in pandas dataframe
            self._data = partition of data with only features of interest and class
            self._cuts = dictionary with cut points for each feature
        :param X: pandas dataframe with data to discretize
        :param class_label: name of the column containing class in input dataframe
        :param features: if !None, features that the user wants to discretize specifically
        :return:
        '''
        #Initialize descriptions of discretizatino bins
        self._bin_descriptions = {}
 
        #Create array with attr indices to discretize
        if features is None:  # Assume all columns are numeric and need to be discretized
            if raw_data_shape is None:
                raise Exception("If feautes=None, raw_data_shape must be a non-empty tuple")
            self._col_idx = range(raw_data_shape[1])
        else:
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if np.issubdtype(features.dtype, np.integer):
                self._col_idx = features
            elif np.issubdtype(features.dtype, np.bool):  # features passed as mask
                if raw_data_shape is None:
                    raise Exception('If features is a boolean array, raw_data_shape must be != None')
                if len(features) != self._data_raw.shape[1]:
                    raise Exception('Column boolean mask must be of dimensions (NColumns,)')
                self._col_idx = np.where(features)
            else:
                raise Exception('features argument must a np.array of column indices or a boolean mask')
 
    def fit(self, X, y):
        X = cartesian_to_hyperspherical(X.copy(), X.shape[1], y)
        
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self._data_raw = X.copy()  # copy of original input data
        self._class_labels = y.reshape(-1, 1)  # make sure class labels is a column vector
        self._classes = np.unique(self._class_labels)
        self._col_idx = range(X.shape[1])
 
        if len(self._col_idx) != self._data_raw.shape[1]:  # some columns will not be discretized
            self._ignore_col_idx = np.array([var for var in range(self._data_raw.shape[1]) if var not in self._col_idx])
 
        # initialize feature bins cut points
        self._cuts = {f: [] for f in self._col_idx}
 
        # pre-compute all boundary points in dataset
        self._boundaries = self.compute_boundary_points_all_features()
 
        # get cuts for all features
        self.all_features_accepted_cutpoints()
 
        #generate bin string descriptions
        self.generate_bin_descriptions()
 
        #Generate one-hot encoding schema
        X_discrete = self.apply_cutpoints(X.copy())
        unique_ints = np.unique(X_discrete, axis=0)
        k_outs = [sum((X_discrete==unique_ints[i]).all(axis=1))/X_discrete.shape[0] for i in range(len(unique_ints))]
        plt.plot(k_outs)
        plt.show()
        self.ints = unique_ints, k_outs
        return self
 
    def transform(self, X, inplace=False):
        # X = cartesian_to_log_hyperspherical(X.copy(), X.shape[1])
        X = cartesian_to_hyperspherical(X.copy(), X.shape[1])
        if inplace:
            discretized = X
        else:
            discretized = X.copy()
        discretized = self.apply_cutpoints(discretized)
        return discretized
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, inplace=True)
 
    def MDLPC_criterion(self, X, y, feature_idx, cut_point):
        '''
        Determines whether a partition is accepted according to the MDLPC criterion
        :param feature: feature of interest
        :param cut_point: proposed cut_point
        :param partition_index: index of the sample (dataframe partition) in the interval of interest
        :return: True/False, whether to accept the partition
        '''
        #get dataframe only with desired attribute and class columns, and split by cut_point
        left_mask = X <= cut_point
        right_mask = X > cut_point
 
        #compute information gain obtained when splitting data at cut_point
        cut_point_gain = cut_point_information_gain_numpy(X, y, cut_point)
        #compute delta term in MDLPC criterion
        N = len(X) # number of examples in current partition
        partition_entropy = entropy_numpy(y)
        k = len(np.unique(y))
        k_left = len(np.unique(y[left_mask]))
        k_right = len(np.unique(y[right_mask]))
        entropy_left = entropy_numpy(y[left_mask])  # entropy of partition
        entropy_right = entropy_numpy(y[right_mask])
        delta = log(3 ** k, 2) - ((k * partition_entropy) - (k_left * entropy_left) - (k_right * entropy_right))
 
        #to split or not to split
        gain_threshold = (log(N - 1, 2) + delta) / N
 
        if cut_point_gain > gain_threshold:
            return True
        else:
            return False
 
    def feature_boundary_points(self, values):
        '''
        Given an attribute, find all potential cut_points (boundary points)
        :param feature: feature of interest
        :param partition_index: indices of rows for which feature value falls whithin interval of interest
        :return: array with potential cut_points
        '''
 
        missing_mask = np.isnan(values)
        data_partition = np.concatenate([values[:, np.newaxis], self._class_labels], axis=1)
        data_partition = data_partition[~missing_mask]
        #sort data by values
        data_partition = data_partition[data_partition[:, 0].argsort()]
 
        #Get unique values in column
        unique_vals = np.unique(data_partition[:, 0])  # each of this could be a bin boundary
        #Find if when feature changes there are different class values
        boundaries = []
        for i in range(1, unique_vals.size):  # By definition first unique value cannot be a boundary
            previous_val_idx = np.where(data_partition[:, 0] == unique_vals[i-1])[0]
            current_val_idx = np.where(data_partition[:, 0] == unique_vals[i])[0]
            merged_classes = np.union1d(data_partition[previous_val_idx, 1], data_partition[current_val_idx, 1])
            if merged_classes.size > 1:
                boundaries += [unique_vals[i]]
        boundaries_offset = np.array([previous_item(unique_vals, var) for var in boundaries])
        return (np.array(boundaries) + boundaries_offset) / 2
 
    def compute_boundary_points_all_features(self):
        '''
        Computes all possible boundary points for each attribute in self._features (features to discretize)
        :return:
        '''
        def padded_cutpoints_array(arr, N):
            cutpoints = self.feature_boundary_points(arr)
            padding = np.array([np.nan] * (N - len(cutpoints)))
            return np.concatenate([cutpoints, padding])
 
        boundaries = np.empty(self._data_raw.shape)
        boundaries[:, self._col_idx] = np.apply_along_axis(padded_cutpoints_array, 0, self._data_raw[:, self._col_idx], self._data_raw.shape[0])
        mask = np.all(np.isnan(boundaries), axis=1)
        return boundaries[~mask]
 
    def boundaries_in_partition(self, X, feature_idx):
        '''
        From the collection of all cut points for all features, find cut points that fall within a feature-partition's
        attribute-values' range
        :param data: data partition (pandas dataframe)
        :param feature: attribute of interest
        :return: points within feature's range
        '''
        range_min, range_max = (X.min(), X.max())
        mask = np.logical_and((self._boundaries[:, feature_idx] > range_min), (self._boundaries[:, feature_idx] < range_max))
        return np.unique(self._boundaries[:, feature_idx][mask])
 
    def best_cut_point(self, X, y, feature_idx):
        '''
        Selects the best cut point for a feature in a data partition based on information gain
        :param data: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
        '''
        candidates = self.boundaries_in_partition(X, feature_idx=feature_idx)
        if candidates.size == 0:
            return None
        gains = [(cut, cut_point_information_gain_numpy(X, y, cut_point=cut)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)
 
        return gains[0][0] #return cut point
 
    def single_feature_accepted_cutpoints(self, X, y, feature_idx):
        '''
        Computes the cuts for binning a feature according to the MDLP criterion
        :param feature: attribute of interest
        :param partition_index: index of examples in data partition for which cuts are required
        :return: list of cuts for binning feature in partition covered by partition_index
        '''
 
        #Delte missing data
        # mask = np.isnan(X)
        # X = X #[~mask]
        # y = y#[~mask]
 
        #stop if constant or null feature values
        if len(np.unique(X)) < 2:
            return
        #determine whether to cut and where
        cut_candidate = self.best_cut_point(X, y, feature_idx)
        if cut_candidate == None:
            return
        decision = self.MDLPC_criterion(X, y, feature_idx, cut_candidate)
 
        # partition masks
        left_mask = X <= cut_candidate
        right_mask = X > cut_candidate
 
        #apply decision
        if not decision:
            return  # if partition wasn't accepted, there's nothing else to do
        if decision:
            #now we have two new partitions that need to be examined
            left_partition = X[left_mask]
            right_partition = X[right_mask]
            if (left_partition.size == 0) or (right_partition.size == 0):
                return #extreme point selected, don't partition
            self._cuts[feature_idx] += [cut_candidate]  # accept partition
            self.single_feature_accepted_cutpoints(left_partition, y[left_mask], feature_idx)
            self.single_feature_accepted_cutpoints(right_partition, y[right_mask], feature_idx)
            #order cutpoints in ascending order
            self._cuts[feature_idx] = sorted(np.unique([self.X_min[feature_idx]]+self._cuts[feature_idx]+[self.X_max[feature_idx]]))
            return
 
    def all_features_accepted_cutpoints(self):
        '''
        Computes cut points for all numeric features (the ones in self._features)
        :return:
        '''
        for attr in self._col_idx:
            self.single_feature_accepted_cutpoints(X=self._data_raw[:, attr], y=self._class_labels, feature_idx=attr)
        return
 
    def generate_bin_descriptions(self):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''
        bin_label_collection = {}
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i+1])) for i in start_bin_indices]
                bin_label_collection[attr] = bin_labels
                self._bin_descriptions[attr] = {i: bin_labels[i] for i in range(len(bin_labels))}
 
    def apply_cutpoints(self, data):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                # data[:, attr] = 'All'
                data[:, attr] = 0
            else:
                # print(f'Lower bound: {data[:, attr].min()}')
                # print(f'Upper bound: {data[:, attr].max()}')
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                discretized_col = np.digitize(x=data[:, attr], bins=cuts, right=False).astype('float') - 1
                discretized_col[np.isnan(data[:, attr])] = np.nan
                data[:, attr] = discretized_col
        return data
if __name__=="__main__":
    from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from forward_algorithms.simulate_expert_data import Simulation
    from inverse_opt_stopping.iq_learn_base import IQ_Agent #, plot_st_reg_bm, plot_st_reg_car
    import os 
    import shutil
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
    from sklearn.preprocessing import StandardScaler
    
    bm_sym = Simulation(problem='radial', total_n=250)
    df = bm_sym.simulate_expert(episodes=250, max_path_length=100) 
    st_cart = df['state_mem'] #[:,[3,4,0]]
    st = st_cart.copy()
    # st = cartesian_to_hyperspherical(st_cart.copy(), st_cart.shape[1])
    
    act = df['action_mem']
    data = np.concatenate((st, act.reshape(-1,1)), axis=1)
    X, y = st, act
    # 'datetime',
    # 'voltmean', 
    # 'rotatemean', 
    # 'pressuremean',        
    # 'vibrationmean', 
    # 'voltsd', 
    # 'rotatesd', 
    # 'pressuresd', 
    # 'vibrationsd',
    # 'error1count', 'error2count', 'error3count', 'error4count',
    #   'error5count'
    feature_names, class_names = ['x1','x2','t'], ['a']
    # feature_names, class_names = ['x1','x2','t'], ['a']
    numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized
 
        #Split between training and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)
    X_train, y_train = X, y
        #Initialize discretizer object and fit to training data
    
    discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer.fit(X_train, y_train)
    X_train_discretized = discretizer.transform(X_train)
 
        #apply same discretization to test set
    # X_test_discretized = discretizer.transform(X_test)
 
        #Print a slice of original and discretized data
    print('Original dataset:\n%s' % str(X_train[0:5]))
    print('Discretized dataset:\n%s' % str(X_train_discretized[0:5]))
 
        #see how feature 0 was discretized
    # print('Feature: %s' % feature_names[0])
    # print('Interval cut-points: %s' % str(discretizer._cuts[0]))
    # print('Bin descriptions: %s' % str(discretizer._bin_descriptions[0]))
 
    data = np.concatenate((st, act.reshape(-1,1)), axis=1)
 
    pd_df = pd.DataFrame(data.astype(float))
    pd_df.columns = ['x1','x2','t','a']
    pd_df['a'] = pd_df['a'].astype('category')
 
    # all possible pairs of states
    y, x = np.meshgrid(np.hstack([min(st_cart[:,0])-0.1,discretizer._cuts[0],max(st_cart[:,0])+0.1]),
                    np.hstack([min(st_cart[:,1])-0.1,discretizer._cuts[1],max(st_cart[:,1])+0.1]))
 
    n_times = len(discretizer._bin_descriptions[2])
    n_feats_cuts = len(discretizer._bin_descriptions[0]), len(discretizer._bin_descriptions[1])
    N = pd_df.shape[0]
    out_degree_dict = {}
    for k in range(n_times):
        interval_acts = {}
        majority_act = []
        k_out = []
        entropy_vals = []
        st_actions = []
        cont_actions = []
        for i in range(n_feats_cuts[0]):
            for j in range(n_feats_cuts[1]):
                bin_filter = (X_train_discretized[:,0]==i) & (X_train_discretized[:,1]==j) & (X_train_discretized[:,2]==k)
                st_filter = (X_train_discretized[:,0]==i) & (X_train_discretized[:,1]==j) & (X_train_discretized[:,2]==k) & (y_train==0.0)
                st_action = sum(st_filter)
                cont_filter = (X_train_discretized[:,0]==i) & (X_train_discretized[:,1]==j) & (X_train_discretized[:,2]==k) & (y_train==1.0)
                cont_action = sum(cont_filter)
                interval_acts[f'({i},{j})'] = (st_action, cont_action)
                # print(f'Bin: ({i},{j}), cont: {cont_action}, st: {st_action}')
                # print(f'{discretizer._bin_descriptions[0][i]}, {discretizer._bin_descriptions[1][j]}')
                if st_action>=cont_action:
                    majority_act.append(0.0)
                    # print(f'append st action')
                else:
                    majority_act.append(1.0)
                    # print(f'append cont action')
                st_actions.append(pd.DataFrame(st_cart[st_filter,:2]))  
                cont_actions.append(pd.DataFrame(st_cart[cont_filter,:2])) 
                entropy_vals.append(entropy_numpy(y_train[bin_filter]))
                cell_k_out = st_action+cont_action
                k_out.append(cell_k_out)
                # out_degree_dict[]
        
        all_stops = pd.concat(st_actions)
        all_stops.columns = ['x1', 'x2']
        all_conts = pd.concat(cont_actions)
        all_conts.columns = ['x1', 'x2']
        # print(f'np.array(majority_act).shape: {np.array(majority_act).shape}')
        Z = np.array(majority_act).reshape(n_feats_cuts[0],n_feats_cuts[1])
        ENT = np.array(entropy_vals).reshape(n_feats_cuts[0],n_feats_cuts[1])
        K_OUT = np.array(k_out).reshape(n_feats_cuts[0],n_feats_cuts[1])/N
        
        # Z = np.pad(Z, pad_width=(0,1), mode='constant', constant_values=np.nan)
 
        # Z = np.random.rand(10, 8)
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(20,25))
        c = ax0.pcolor(x, y, Z.T, edgecolors='b', linewidth=1, cmap='PuBu_r', shading='auto', vmin=0, vmax=1)
        ax0.scatter(all_stops.x2,all_stops.x1, color='red', s=10)
        ax0.scatter(all_conts.x2,all_conts.x1, color='green', s=1.5)
        ax0.set_title(f'Majority action for time interval: {discretizer._bin_descriptions[2][k]}', size=35)
        ax0.xaxis.set_tick_params(labelsize=25)
        ax0.yaxis.set_tick_params(labelsize=25)
        clb = fig.colorbar(c, ax=ax0)
        clb.ax.tick_params(labelsize=25) 
        
        c = ax1.pcolor(x, y, ENT.T, edgecolors='b', linewidth=1, cmap='PuBu_r', shading='auto', vmin=0, vmax=1)
        ax1.scatter(all_stops.x2,all_stops.x1, color='red', s=10)
        ax1.scatter(all_conts.x2,all_conts.x1, color='green', s=1.5)
        ax1.set_title(f'Bins entropy for time interval: {discretizer._bin_descriptions[2][k]}', size=35)
        ax1.xaxis.set_tick_params(labelsize=25)
        ax1.yaxis.set_tick_params(labelsize=25)
        clb = fig.colorbar(c, ax=ax1)
        clb.ax.tick_params(labelsize=25) 
        
        import matplotlib.colors as colors
 
        
        c = ax2.pcolor(x, y, K_OUT.T, edgecolors='b', linewidth=1, cmap='PuBu_r',shading='auto')
        ax2.scatter(all_stops.x2,all_stops.x1, color='red', s=10)
        ax2.scatter(all_conts.x2,all_conts.x1, color='green', s=1.5)
        ax2.set_title(f'Bin (relative) out-degree for time interval: {discretizer._bin_descriptions[2][k]}', size=35)
        ax2.xaxis.set_tick_params(labelsize=25)
        ax2.yaxis.set_tick_params(labelsize=25)
        clb = fig.colorbar(c, ax=ax2)
        clb.ax.tick_params(labelsize=25) 
        # clb.ax.set_label('# of contacts')
        
        plt.show()
 
        
        
        
    Z = np.random.rand(x.shape[0]-1, x.shape[1]-1)
    plt.pcolor(x, y, Z, edgecolors='b', linewidth=1)
    plt.title('All macro regions', size=25)
    plt.xlim((x.min(),x.max()))
    plt.ylim((y.min(),y.max()))
    plt.show()
 
    bm_expert_data = df
 
    unique_ints = np.unique(X_train_discretized, axis=0)
    k_outs = [sum((X_train_discretized==unique_ints[i]).all(axis=1))/X_train_discretized.shape[0] for i in range(len(unique_ints))]
    # k_out_dict = {unique_ints[i,:].tobytes(): k_outs[i] for i in range(len(unique_ints))}
    bm_test_data = bm_sym.simulate_test(episodes=250, max_path_length=200)
    X_test_discretized = discretizer.transform(bm_test_data['state_mem'])
    out_degree_test = np.zeros_like(X_test_discretized[:,0])
    for i in range(len(k_outs)):
        out_degree_test[(X_test_discretized==unique_ints[i]).all(axis=1)] =  k_outs[i]
    s=0
    iq_iql = IQ_Agent(obs_dim=bm_expert_data['state_mem'][0].shape[0], 
                        approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE', 
                        q_lr=0.01, env_lr=0.01, g_lr=0.001,epsilon=0.1,seed=s,
                        divergence='js', conservative=True, 
                                    discretiser=discretizer, )
    iq_iql.train(mem=bm_expert_data, batch_size=128, n_epoches=50)
    iq_iql.out_thresh = 0.0
    iq_iql.test(test_memory=bm_test_data,
                from_grid=True)
    iq_iql.test(test_memory=bm_test_data,
                from_grid=False)
 
    iq_iql.out_thresh=0.0001
    iq_iql.test(test_memory=bm_test_data,
                from_grid=True)
    iq_iql.test(test_memory=bm_test_data, 
                from_grid=False)
 
    iq_iql.out_thresh=0.01
    iq_iql.test(test_memory=bm_test_data,
                from_grid=True)
    iq_iql.test(test_memory=bm_test_data, 
                from_grid=False)
 
    iq_iql.out_thresh=0.05
    iq_iql.test(test_memory=bm_test_data,
                from_grid=True)
    iq_iql.test(test_memory=bm_test_data, 
                from_grid=False)
 
    iq_iql.out_thresh=0.1
    iq_iql.test(test_memory=bm_test_data, 
                from_grid=True)
    iq_iql.test(test_memory=bm_test_data, 
                from_grid=False)
 
    iq_iql.out_thresh=0.5
    iq_iql.test(test_memory=bm_test_data,
                from_grid=True)
    iq_iql.test(test_memory=bm_test_data, 
                from_grid=False)
    # iq_iql.test(test_memory=bm_test_data, conservative=True, 
    #             discretiser=discretizer, 
    #             out_thresh=0.05,
    #             from_grid=True)
    # iq_iql.test(test_memory=bm_test_data, conservative=True, 
    #             discretiser=discretizer, 
    #             out_thresh=0.1,
    #             from_grid=True)
    # iq_iql.test(test_memory=bm_test_data, conservative=True, k_out=out_degree_test, out_thresh=0.01)
    # iq_iql.test(test_memory=bm_test_data, conservative=True, k_out=out_degree_test, out_thresh=0.025)
    # iq_iql.test(test_memory=bm_test_data, conservative=True, k_out=out_degree_test, out_thresh=0.05)
    # iq_iql.test(test_memory=bm_test_data, conservative=True, k_out=out_degree_test, out_thresh=0.1)
 
    # , conservative=False, k_out=None, out_thresh=5
    # np.frombuffer(list(res.keys())[0])
 
    # Finished test paths
    # F1: 0.1912532637075718
    # pr_auc: 0.10858967237434722
    # median time-to-event: 7.517006802721088
    # median event miss-rate: 0.118
    # number of missed events: 59
 
# def inner_product_embedding(X, y=None, num_reference_directions=150, 
#                             pca_components=2,
#                            plot=True, random_seed=42):
#     """
#     Performs inner product embedding followed by PCA projection.
 
#     Parameters:
#         X (np.ndarray): Input data of shape (N, K)
#         y (np.ndarray): Optional labels of shape (N,)
#         num_reference_directions (int): Number of random hyperspherical reference directions
#         pca_components (int): Number of PCA components to reduce to
#         plot (bool): If True, shows 2D scatter plot if pca_components == 2
#         random_seed (int): Seed for reproducibility
 
#     Returns:
#         np.ndarray: Final PCA-embedded features of shape (N, pca_components)
#     """
#     N, K = X.shape
 
#     # Normalize input vectors
#     X_unit = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
 
#     # Generate random directions
#     rng = np.random.default_rng(random_seed)
#     directions = rng.normal(size=(num_reference_directions, K))
#     directions /= np.linalg.norm(directions, axis=1, keepdims=True)
 
#     # Compute inner products
#     inner_product_features = np.dot(X_unit, directions.T)
 
#     # Apply PCA
#     pca = PCA(n_components=pca_components)
#     pca_features = pca.fit_transform(inner_product_features)
 
#     # Optional 2D plot
#     if plot and pca_components >= 2 and y is not None:
#         scatter_args = {'alpha': 0.7, 'edgecolors': 'k'}
#         plt.figure(figsize=(6, 5))
#         plt.scatter(pca_features[:, 0], pca_features[:, 1], c=y, cmap='coolwarm', **scatter_args)
        
#         plt.title("Inner Product + PCA Embedding")
#         plt.xlabel("PC 1")
#         plt.ylabel("PC 2")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
 
#     return pca_features
 
# def geodesic_distance_embedding(X, y=None, num_reference_points=10, return_encoded=True,
#                                  plot=True, random_seed=42):
#     """
#     Computes geodesic (angular) distance embeddings from input Cartesian state vectors
#     to a set of random unit reference directions on the hypersphere.
#     Optionally plots PCA projections and class coloring.
 
#     Parameters:
#         X (np.ndarray): Input data of shape (N, K), assumed to be Cartesian state vectors
#         y (np.ndarray): Optional labels of shape (N,), for coloring
#         num_reference_points (int): Number of reference unit vectors (anchors) on the sphere
#         return_encoded (bool): If True, returns the encoded geodesic features
#         plot (bool): If True, plots PCA of original and embedded features
#         random_seed (int): Random seed for reproducibility
 
#     Returns:
#         encoded (np.ndarray): shape (N, num_reference_points), geodesic angular distances in radians
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
 
#     N, K = X.shape
 
#     # Normalize input to unit vectors
#     X_norm = np.linalg.norm(X, axis=1, keepdims=True)
#     X_unit = X / np.clip(X_norm, 1e-8, None)
 
#     # Generate reference unit vectors
#     rng = np.random.default_rng(random_seed)
#     reference_vectors = rng.normal(size=(num_reference_points, K))
#     reference_vectors /= np.linalg.norm(reference_vectors, axis=1, keepdims=True)
 
#     # Compute cosine similarities (dot products)
#     dot_products = np.dot(X_unit, reference_vectors.T)
#     dot_products = np.clip(dot_products, -1.0, 1.0)  # for numerical stability
 
#     # Compute geodesic distances (angular distances in radians)
#     geodesic_distances = np.arccos(dot_products)
#     pca_original = PCA(n_components=3).fit_transform(X)
#     pca_encoded = PCA(n_components=3).fit_transform(geodesic_distances)
 
#     # Plotting
#     if plot and y is not None:
        
#         fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#         scatter_args = {'alpha': 0.7, 'edgecolors': 'k'}
 
#         axs[0].scatter(pca_original[:, 0], pca_original[:, 1], c=y, cmap='coolwarm', **scatter_args)
#         axs[1].scatter(pca_encoded[:, 0], pca_encoded[:, 1], c=y, cmap='coolwarm', **scatter_args)
        
#         axs[0].set_title("Original Cartesian Features (PCA Projection)")
#         axs[0].set_xlabel("PC 1")
#         axs[0].set_ylabel("PC 2")
#         axs[0].grid(True)
 
#         axs[1].set_title("Geodesic Distance Embedding (PCA Projection)")
#         axs[1].set_xlabel("PC 1")
#         axs[1].set_ylabel("PC 2")
#         axs[1].grid(True)
 
#         plt.tight_layout()
#         plt.show()
 
#     if return_encoded:
#         return pca_encoded
# def geodesic_embedding(X, y=None, num_reference_points=10, reference_points=None, return_normed=True,
#                         random_seed=42, plot=True):
#     """
#     Performs geodesic feature embedding for input data and optionally plots PCA projections with class labels.
 
#     Parameters:
#         X (np.ndarray): Input data of shape (N, K)
#         y (np.ndarray): Optional class labels of shape (N,), used for color-coding in plot
#         num_reference_points (int): Number of reference directions to sample
#         reference_points (np.ndarray): Optional (M, K) array of unit vectors as reference points
#         return_normed (bool): If True, also return the normalized input data
#         random_seed (int): Seed for reproducibility
#         plot (bool): If True, show PCA projection of original vs geodesic features
 
#     Returns:
#         geodesic_features (np.ndarray): Shape (N, M), angular distances to reference points
#         X_normed (np.ndarray, optional): Normalized input of shape (N, K) if return_normed=True
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
 
#     N, K = X.shape
 
#     # Normalize input vectors to lie on the unit sphere
#     norms = np.linalg.norm(X, axis=1, keepdims=True)
#     X_normed = X / np.clip(norms, 1e-8, None)
 
#     # Generate reference points if not provided
#     if reference_points is None:
#         rng = np.random.default_rng(random_seed)
#         reference_points = rng.normal(size=(num_reference_points, K))
#         reference_points /= np.linalg.norm(reference_points, axis=1, keepdims=True)
 
#     # Compute cosine similarity (dot product of unit vectors)
#     dot_products = np.dot(X_normed, reference_points.T)
#     dot_products = np.clip(dot_products, -1.0, 1.0)  # Ensure numerical stability
 
#     # Convert to angular distances (radians)
#     geodesic_features = np.arccos(dot_products)
 
#     # Plotting
#     if plot and y is not None:
#         pca_original = PCA(n_components=2).fit_transform(X)
#         pca_geodesic = PCA(n_components=2).fit_transform(geodesic_features)
 
#         fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#         scatter_args = {'alpha': 0.7, 'edgecolors': 'k'}
 
#         axs[0].scatter(pca_original[:, 0], pca_original[:, 1], c=y, cmap='coolwarm', **scatter_args)
#         axs[1].scatter(pca_geodesic[:, 0], pca_geodesic[:, 1], c=y, cmap='coolwarm', **scatter_args)
        
#         axs[0].set_title("Original Features (PCA Projection)")
#         axs[0].set_xlabel("PC 1")
#         axs[0].set_ylabel("PC 2")
#         axs[0].grid(True)
 
#         axs[1].set_title("Geodesic Embedding (PCA Projection)")
#         axs[1].set_xlabel("PC 1")
#         axs[1].set_ylabel("PC 2")
#         axs[1].grid(True)
 
#         plt.tight_layout()
#         plt.show()
 
#     if return_normed:
#         return geodesic_features, X_normed
#     else:
#         return geodesic_features