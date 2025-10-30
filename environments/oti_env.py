import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA

def shift_array(array, place=-1):
    new_arr = np.roll(array, place, axis=0)
    new_arr[place:,:,:] = 0
    new_arr[place:,0,np.isnan(new_arr[place-1,0,:])] = np.nan
    new_arr[place:,1,np.isnan(new_arr[place-1,1,:])] = np.nan
    new_arr[place:,2,np.isnan(new_arr[place-1,2,:])] = np.nan
    # new_arr[np.isnan(new_arr):] = np.nan
    return new_arr


class OTI_turbofan:
    # https://data.nasa.gov/Raw-Data/PHM-2008-Challenge/nk8v-ckry/about_data
    def __init__(self, path_to_data='data/turbofan_data/train.txt', total_n=1000):
        data = pd.read_csv(path_to_data, sep=" ", header=None).iloc[:,:-2]
        cols = ['unit_id', 'time','opp_set_1', 'opp_set_2']+['sens_'+str(i) for i in range(1,23)]
        data.columns = cols
        data.time -= 1
        data.unit_id -= 1
        # to_drop = ['sens_1','sens_2', 'sens_6', 'sens_11', 'sens_17','sens_19','sens_20']
        # data.drop(to_drop,axis=1,inplace=True)
        data['failure'] = False
        cycles = data.groupby('unit_id')['time'].max()
        for i, ind in enumerate(cycles.index.values) :
            data.loc[(data.unit_id==ind) & (data.time==cycles.values[i]-1), 'failure'] = True
        self.data = data
        # self.pca = PCA(n_components=5)

        # data = self.data
        # self.data = data[data.unit_id.isin(np.arange(n,N+1))]
        
        self.train_test_paths = 0 #random paths ids out of np.unique(PATH_IDS)
        all_ids = [i for i in range(len(self.data.unit_id.unique()))]
        print(f'all_ids: {len(all_ids)}')
        total_n = len(all_ids)
        self.training_ids = random.sample(all_ids, round(total_n*0.70))
        self.testing_ids = list(set(all_ids) - set(self.training_ids))
        # self.training_ids = random.sample(all_ids, 500)
        # self.testing_ids = list(set(all_ids) - set(self.training_ids))
 
    def sim_expert(self, episodes=None, max_path_length=None):
        # train_memory = self.memory #[self.memory[self.memory['path_ids']==self.train_test_paths[0]][i] for i in range(len(np.unique(self.memory['path_ids'])))]
        return self.load_transform_data(N=500, n=0, full_paths=False, ids=self.training_ids)
    
    def sim_test(self, episodes, max_path_length):
        return self.load_transform_data(N=1000, n=500, full_paths=False, ids=self.testing_ids)
    
    def load_transform_data(self,n=100,N=100,full_paths=False,mode='train', ids=None):
        # data = self.data[self.data.unit_id in ids]
        data = self.data[np.in1d(self.data.unit_id, ids)]
        from sklearn.preprocessing import LabelEncoder
        # le = LabelEncoder()
        # data['time'] = le.fit_transform(data.iloc[:,1])
        # once failure, machine stops
        for i in np.unique(data.unit_id):
            first_failure_id = np.where(data[data.unit_id==i].failure)[0]
            failures_copy = data.failure[data.unit_id==i].values.copy()
            if len(first_failure_id)!=0:
                first_failure_id = first_failure_id[0]
                # print(f'first_failure_id: {first_failure_id}')
                failures_copy[first_failure_id-2:] = True
                data.iloc[:,-1][data.unit_id==i] = failures_copy
        # data.dropna(inplace=True)
        # down-sample every 5 hours
        data = data[data['time'].isin(np.arange(data['time'].min(),data['time'].max(), 10))]
        for i in np.unique(data.unit_id):
            # first_failure_id = np.where(data[data.machineID==i].failure)[0]
            failures_copy = data.failure[data.unit_id==i].values.copy()
            first_failure_id = np.where(failures_copy)[0]
            failures_copy = failures_copy.astype(float)
            if len(first_failure_id)!=0:
                first_failure_id = first_failure_id[0]
                failures_copy[first_failure_id+1:] =  np.repeat(np.nan, len(failures_copy[first_failure_id+1:]))
                data.iloc[:,-1][data.unit_id==i] = failures_copy
        # print(f'before drop nan data.failure[:100]: {data.failure[:100]}')
        data.dropna(inplace=True)
        data.failure = data.failure.astype(bool)
        shifted_data = data.groupby('unit_id').shift(-1).reset_index()
        shifted_data['unit_id'] = data.unit_id.values
        nan_ids = np.where(shifted_data.failure.isna())[0]
        for col in data.columns:
            shifted_data[col].iloc[nan_ids] = data[col].iloc[nan_ids].copy()
        # nan_ids = np.where(shifted_data.failure.isna())[0]
        # shifted_data.iloc[nan_ids] = shifted_data.loc[nan_ids-1].copy()
        
        states_names = [
                        'time',
                        'opp_set_1', 
                        'opp_set_2', 
                        'sens_3', 
                        'sens_4',
                        'sens_5', 
                        'sens_7', 
                        'sens_8', 
                        'sens_9', 
                        'sens_10', 
                        'sens_12', 
                        'sens_13',
                        'sens_14', 
                        'sens_15', 
                        'sens_16', 
                        'sens_18', 
                        'sens_21',
                        'sens_22',
                        ]        
        # shifted_data = shift_array(data)
        if mode=='train':
            self.le = LabelEncoder()
            self.le.fit(data.time.copy())
        data['time'] = self.le.transform(data.time)
        shifted_data['time'] = self.le.transform(shifted_data.time)
        # if mode=='train':
        #     self.pca.fit(data[states_names].values)
            
        # return {'state_mem': self.pca.transform(data[states_names].values), 
        #         'next_state_mem': self.pca.transform(shifted_data[states_names].values), 
        #         'action_mem': (data.failure.astype(int).values-1)**2, 
        #         'done_mem': data.failure.astype(int).values,
        #         'init_states_mem': self.pca.transform(data[states_names].values),
        #         'path_ids': data.unit_id.values,
        #         'time_ids': data.time.values}
        return {'state_mem': data[states_names].values, 
                'next_state_mem': shifted_data[states_names].values, 
                'action_mem': (data.failure.astype(int).values-1)**2, 
                'done_mem': data.failure.astype(int).values,
                'init_states_mem': data[states_names].values,
                'path_ids': data.unit_id.values,
                'time_ids': data.time.values}
# if __name__=='__main__':
#     oti = OTI()
# #     oti.sim_expert()

class OTI_azure:
    def __init__(self):
        self.data = pd.read_csv('data/AMLWorkshop-master/Data/features.csv', sep=',')
        all_ids = [i for i in range(len(self.data.machineID.unique()))]
        self.training_ids = random.sample(all_ids, 70)
        self.testing_ids = list(set(all_ids) - set(self.training_ids))
 
    def sim_expert(self, episodes=None, max_path_length=None):
        # train_memory = self.memory #[self.memory[self.memory['path_ids']==self.train_test_paths[0]][i] for i in range(len(np.unique(self.memory['path_ids'])))]
        return self.load_transform_data(n=0,N=49,full_paths=False, ids=self.training_ids, mode='train')
    
    def sim_test(self, episodes, max_path_length): 
        return self.load_transform_data(n=50,N=100,full_paths=False, ids=self.testing_ids, mode='test')
    
    def load_transform_data(self,n,N,full_paths=False, ids=None, mode='train'):
        data = self.data
        # data = data[data.machineID.isin(np.arange(n,N+1))]
        data = data[data.machineID.isin(ids)]
        # data.drop(['error1count', 'error2count', 'error3count', 'error4count',
        #  'error5count', 'model', 'age'], inplace=True, axis=1)
        data.drop(['model', 'age'], inplace=True, axis=1)
        # data.rename(columns={"B": "c"}, inplace=True)
        from sklearn.preprocessing import LabelEncoder
        if mode=='train':
            self.le = LabelEncoder()
            self.le.fit(data.datetime.copy())
        data['datetime'] = self.le.transform(data.iloc[:,0])
        # once failure, machine stops
        # print(f'orig data.failure[:100]: {data.failure[:100]}')
        # for i in np.unique(data.machineID):
        #     failures_copy = data.failure[data.machineID==i].values.copy()
        #     first_failure_id = np.where(failures_copy)[0]
        #     # failures_copy = failures_copy.astype()
        #     if len(first_failure_id)!=0:
        #         first_failure_id = first_failure_id[0]
        #         failures_copy[first_failure_id-2:] = np.repeat(True, len(failures_copy[first_failure_id-2:]))
        #         data.iloc[:,-1][data.machineID==i] = failures_copy
        for i in np.unique(data.machineID):
            first_failure_id = np.where(data[data.machineID==i].failure)[0]
            failures_copy = data.failure[data.machineID==i].values.copy()
            if len(first_failure_id)!=0:
                first_failure_id = first_failure_id[0]
                # print(f'first_failure_id: {first_failure_id}')
                failures_copy[first_failure_id-2:] = True
                data.iloc[:,-1][data.machineID==i] = failures_copy
        
        # down-sample every 15 hours
        data = data[data['datetime'].isin(np.arange(data['datetime'].min(),data['datetime'].max(), 10))]
        # data.reset_index(inplace=True)
        for i in np.unique(data.machineID):
            # first_failure_id = np.where(data[data.machineID==i].failure)[0]
            failures_copy = data.failure[data.machineID==i].values.copy()
            first_failure_id = np.where(failures_copy)[0]
            failures_copy = failures_copy.astype(float)
            if len(first_failure_id)!=0:
                first_failure_id = first_failure_id[0]
                failures_copy[first_failure_id+1:] =  np.repeat(np.nan, len(failures_copy[first_failure_id+1:]))
                data.iloc[:,-1][data.machineID==i] = failures_copy
    
        # print(f'before drop nan data.failure[:100]: {data.failure[:100]}')
        data.dropna(inplace=True)
        data.failure = data.failure.astype(bool)
        # print(f'data.failure[:100]: {data.failure[:100]}')
        shifted_data = data.groupby('machineID').shift(-1).reset_index()
        shifted_data['machineID'] = data.machineID.values
        nan_ids = np.where(shifted_data.failure.isna())[0]
        for col in data.columns:
            shifted_data[col].iloc[nan_ids] = data[col].iloc[nan_ids].copy()
        # nan_ids = np.where(shifted_data.failure.isna())[0]
        # shifted_data.iloc[nan_ids] = shifted_data.loc[nan_ids-1].copy()
        
        states_names = [
                        'datetime',
                        'voltmean', 
                        'rotatemean', 
                        'pressuremean',        
                        'vibrationmean', 
                        'voltsd', 
                        'rotatesd', 
                        'pressuresd', 
                        'vibrationsd',
                        'error1count', 
                        'error2count', 
                        'error3count', 
                        'error4count',
                          'error5count'
                        ]        
        # shifted_data = shift_array(data)
        
        # if mode=='train':
        self.lec = LabelEncoder()
        self.lec.fit(data.datetime.copy())
        data['datetime'] = self.lec.transform(data.datetime.copy())
        shifted_data['datetime'] = self.lec.transform(shifted_data.datetime.copy())
        
        # if mode=='train':
        #     self.pca.fit(data[states_names].values)

        return {'state_mem': data[states_names].values, 
                'next_state_mem': shifted_data[states_names].values, 
                'action_mem': (data.failure.astype(int).values-1)**2, 
                'done_mem': data.failure.astype(int).values,
                'init_states_mem': data[states_names].values,
                'path_ids': data.machineID.values,
                'time_ids': data.datetime.values}