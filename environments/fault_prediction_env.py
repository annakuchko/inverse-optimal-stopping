import pandas as pd
import matplotlib.pyplot as plt


class FaultPredictionEnv:
    def __init__(self):
        features_to_use = ['datetime', 'machineID', 'voltmean', 'rotatemean', 'pressuremean', 'vibrationmean', 'voltsd', 'rotatesd', 'pressuresd', 'vibrationsd', 'failure']
        df = pd.read_csv('data/AMLWorkshop-master/Data/features.csv')[features_to_use]
        df_gb = df[['datetime', 'machineID', 'failure']][df.failure].groupby('machineID').first()
        plt.scatter(df_gb.index, df_gb.datetime, s=0.4) #plot legend for datetime in a shorter format + pigger figure, smaller text
        plt.title('First failure time for each of the machines')
        plt.show()
        self.df = df
        self.df_gb = df_gb
        #  split data into train test, for train do the following transformation 
        # to drop the end of the path when failure happens, for test leave 
        # the whole path untouched, only modify actions (a=0 for all states 
        # after failure happens) to estimate the event miss rate
        
    def sim_expert(self, episodes=500, max_path_length=None):

        data = None
        for i, ids in enumerate(df_gb.index):

            dat = df[df.machineID==ids][pd.to_datetime(df.datetime)<=pd.Timestamp(df_gb.datetime.values[i])]
            if data is None:
                data = dat.copy()
            else:
                data = data._append(dat)
        data.failure = (~data.failure).astype('int')

        return {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES, 
                'reward_mem': REWARD_MEM,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}
    
    def sim_test(self, episodes=500, max_path_length=None):
        return {'state_mem': STATE_MEM_SIM, 
                'next_state_mem': STATE2_MEM_SIM, 
                'action_mem': ACTION_MEM_SIM, 
                'done_mem': DONE_MEM_SIM,
                'init_states_mem': INIT_STATES, 
                'reward_mem': REWARD_MEM,
                'path_ids':PATH_IDS,
                'time_ids':TIME_IDS}

# memory = {'state_mem': STATE_MEM_SIM, 
#         'next_state_mem': STATE2_MEM_SIM, 
#         'action_mem': ACTION_MEM_SIM, 
#         'done_mem': DONE_MEM_SIM,
#         'init_states_mem': INIT_STATES, 
#         # 'reward_mem': REWARD_MEM,
#         'path_ids':PATH_IDS,
#         'time_ids':TIME_IDS}