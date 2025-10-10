# from environments.secretary_env import Secretary
from environments.oti_env import OTI_azure, OTI_turbofan
from environments.symmetrical_bm_env import SBM, SBM_old
from environments.car_env import CarEnv
from environments.changepoint_env import CP1, CP2, CP3, CP4, CP1M, CP2M, CP3M, CP4M
from environments.star_env import STAR, RADIAL

class Simulation():
    def __init__(self, problem, total_n=1000):
        self.problem = problem
        # if problem=='secretary':
        #     env = Secretary()
        if problem=='car':
            env = CarEnv()
        elif problem=='azure':
            env = OTI_azure()
        elif problem=='nasa_turbofan':
            env = OTI_turbofan()
        elif problem=='symm_bm_G':
            env = SBM(total_n=total_n, dat_type='G')
        elif problem=='symm_bm_gG':
            env = SBM(total_n=total_n, dat_type='gG')
        elif problem=='bessel2':
            env = SBM_old()
        elif problem=='CP1':
            env = CP1(total_n=total_n)
        elif problem=='CP2':
            env = CP2(total_n=total_n)
        elif problem=='CP3':
            env = CP3(total_n=total_n)
        elif problem=='CP4':
            env = CP4(total_n=total_n)
        elif problem=='CP1M':
            env = CP1M(total_n=total_n)
        elif problem=='CP2M':
            env = CP2M(total_n=total_n)
        elif problem=='CP3M':
            env = CP3M(total_n=total_n)
        elif problem=='CP4M':
            env = CP4M(total_n=total_n)
        elif problem=='star':
            env = STAR(total_n=total_n)
        elif problem=='radial':
            env = RADIAL(total_n=total_n)
        else:
            raise(ValueError('Env not provided or does not exist'))
        self.env = env
        
    def simulate_expert(self, episodes, max_path_length):
        return self.env.sim_expert(episodes, max_path_length)
    
    def simulate_test(self, episodes, max_path_length):
        return self.env.sim_test(episodes, max_path_length)


# Discretisation 
if "__name__"=='__main__':
# if True:
    def is_float(n):
        try:
            float_n = float(n)
        except ValueError:
            return False
        else:
            return True
    
    """Pandas DataFrame↔Table conversion helpers"""
    import numpy as np
    import pandas as pd
    from pandas.api.types import (
        is_categorical_dtype, is_object_dtype,
        is_datetime64_any_dtype, is_numeric_dtype,
    )
    
    from Orange.data import (
        Table, Domain, DiscreteVariable, StringVariable, TimeVariable,
        ContinuousVariable,
    )
    
    # __all__ = ['table_from_frame', 'table_to_frame']
    
    
    def table_from_frame(df,class_name, *, force_nominal=False):
        """
        Convert pandas.DataFrame to Orange.data.Table
    
        Parameters
        ----------
        df : pandas.DataFrame
        force_nominal : boolean
            If True, interpret ALL string columns as nominal (DiscreteVariable).
    
        Returns
        -------
        Table
        """
    
        def _is_discrete(s):
            return (is_categorical_dtype(s) or
                    is_object_dtype(s) and (force_nominal or
                                            s.nunique() < s.size**.666))
    
        def _is_datetime(s):
            if is_datetime64_any_dtype(s):
                return True
            try:
                if is_object_dtype(s):
                    pd.to_datetime(s, infer_datetime_format=True)
                    return True
            except Exception:  # pylint: disable=broad-except
                pass
            return False
    
        # If df index is not a simple RangeIndex (or similar), put it into data
        if not (df.index.is_integer() and (df.index.is_monotonic_increasing or
                                           df.index.is_monotonic_decreasing)):
            df = df.reset_index()
    
        attrs, metas,calss_vars = [], [],[]
        X, M = [], []
    
        # Iter over columns
        for name, s in df.items():
            name = str(name)
            if name == class_name:
                discrete = s.astype('category').cat
                calss_vars.append(DiscreteVariable(name, discrete.categories.astype(str).tolist()))
                X.append(discrete.codes.replace(-1, np.nan).values)
            elif _is_discrete(s):
                discrete = s.astype('category').cat
                attrs.append(DiscreteVariable(name, discrete.categories.astype(str).tolist()))
                X.append(discrete.codes.replace(-1, np.nan).values)
            elif _is_datetime(s):
                tvar = TimeVariable(name)
                attrs.append(tvar)
                s = pd.to_datetime(s, infer_datetime_format=True)
                X.append(s.astype('str').replace('NaT', np.nan).map(tvar.parse).values)
            elif is_numeric_dtype(s):
                attrs.append(ContinuousVariable(name))
                X.append(s.values)
            else:
                metas.append(StringVariable(name))
                M.append(s.values.astype(object))
    
        return Table.from_numpy(Domain(attrs, calss_vars, metas),
                                np.column_stack(X) if X else np.empty((df.shape[0], 0)),
                                None,
                                np.column_stack(M) if M else None)
    
    from inverse_opt_stopping.data_preprocessing import shift_array, to_bool, preprocess_data, plot_inputs 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from forward_algorithms.simulate_expert_data import Simulation
    from inverse_opt_stopping.iq_learn import IQ_Agent, plot_st_reg_bm, plot_st_reg_car
    import os 
    import shutil
    
    import Orange
    sns.set_style("whitegrid")
    # DRIVER_ROC_AUC = []
    # DRIVER_F1 = []
    # DRIVER_ACCURACY = []
    # DRIVER_PRECISION = []
    # DRIVER_RECALL = []
    
    # BM_ROC_AUC = []
    # BM_F1 = []
    # BM_ACCURACY = []
    # BM_PRECISION = []
    # BM_RECALL = []
    # BM_PR_AUC = []
    # import shutil
    # SEEDS=[0]
    # from sklearn.model_selection import ParameterGrid
    # bm_sym = Simulation(problem='symm_bm')
    # df = bm_sym.simulate_expert(episodes=None, max_path_length=None)
    
    
    st = df['state_mem']
    act = df['action_mem']
    data = np.concatenate((st, act.reshape(-1,1)), axis=1)
    
    pd_df = pd.DataFrame(data.astype(float))
    pd_df.columns = ['x1','x2','t','a']
    pd_df['a'] = pd_df['a'].astype('category')
    org_data = table_from_frame(pd_df, 'a')
    disc = Orange.preprocess.Discretize()
    disc.method = Orange.preprocess.discretize.EntropyMDL()
    orange_table_discrete = disc(org_data)
    
    discrete_df = Orange.data.pandas_compat.table_to_frame(orange_table_discrete)
    x1_ints = discrete_df.x1.unique()
    x2_ints = discrete_df.x2.unique()
    t_ints = discrete_df.t.unique()
    all_x1 = np.concatenate(np.array([np.array([i.split(' - ')]).reshape(-1,1) for i in x1_ints]), axis=0).flatten()
    x1_intervals = np.array([float(x) for x in all_x1 if is_float(x)])
    all_x2 = np.concatenate(np.array([np.array([i.split(' - ')]).reshape(-1,1) for i in x2_ints]), axis=0).flatten()
    x2_intervals = np.array([float(x) for x in all_x2 if is_float(x)])
    all_t = np.concatenate(np.array([np.array([i.split(' - ')]).reshape(-1,1) for i in t_ints]), axis=0).flatten()
    t_intervals = np.array([float(x) for x in all_t if is_float(x)])
    
    
    y, x = np.meshgrid(np.hstack([-1,np.unique(x1_intervals),1]),
                    np.hstack([-1,np.unique(x2_intervals),1]))


    
    for k in range(3):
        interval_acts = {}
        majority_act = []
        st_actions = []
        cont_actions = []
        for i in range(6):
            for j in range(8):
                st_filter = (discrete_df.x1==discrete_df.x1.unique()[i]) & (discrete_df.x2==discrete_df.x2.unique()[j]) & (discrete_df.t==discrete_df.t.unique()[k]) & (discrete_df.a=='0.0')
                st_action = sum(st_filter)
                cont_filter = (discrete_df.x1==discrete_df.x1.unique()[i]) & (discrete_df.x2==discrete_df.x2.unique()[j]) & (discrete_df.t==discrete_df.t.unique()[k]) & (discrete_df.a=='1.0')
                cont_action = sum(cont_filter)
                interval_acts[f'({i},{j})'] = (st_action, cont_action)
                print(f'Bin: ({i},{j}), cont: {cont_action}, st: {st_action}')
                print(f'{discrete_df.x1.unique()[i]}, {discrete_df.x2.unique()[i]}')
                if st_action>cont_action:
                    # if i<=0:
                    #     majority_act.append(0.0)
                    # if j<=0:
                    #     majority_act.append(0.0)
                    # if i>=5:
                    #     majority_act.append(0.0)
                    # if j>=7:
                    #     majority_act.append(0.0)
                    majority_act.append(0.0)
                else:
                    # if i<=0:
                    #     majority_act.append(1.0)
                    # if j<=0:
                    #     majority_act.append(1.0)
                    # if i>=5:
                    #     majority_act.append(1.0)
                    # if j>=7:
                    #     majority_act.append(1.0)
                    majority_act.append(1.0)
                st_actions.append(pd_df[st_filter])  
                cont_actions.append(pd_df[cont_filter])  
         
        all_stops = pd.concat(st_actions)
        all_conts = pd.concat(cont_actions)
        Z =np.array(majority_act).reshape(6,8)
        Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
        import random
        import numpy as np

        def add_noise(x):
            if x==0.0:
                noise = float(np.random.randint(1,100))/100
                return noise
            else:
                return x

        n = np.vectorize(add_noise)
        
        # Z = np.random.rand(10, 8)
        plt.pcolor(np.flip(x.T,1), np.flip(y.T,1), Z.T)
        plt.scatter(all_stops.x2,all_stops.x1, color='red', s=0.5).invert_yaxis()
        plt.scatter(all_conts.x2,all_conts.x1, color='green', s=0.5).invert_yaxis()
        plt.title(f'Time interval: {discrete_df.t.unique()[k]}')
        plt.show()   
        
        
#  The order of .x1.unique() and the actual intervals 
# in x1 and x2 are different (see plots as an example 
# of the error)


