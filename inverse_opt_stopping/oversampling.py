# Oversampling helper-functions
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

def oversampling(input_states, actions, strategy='SMOTE',sampling_strategy=0.25, k_neighbors=15):

    X_resampled, y_resampled = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors).fit_resample(
        input_states, actions.astype(int))
    # X_resampled[y_resampled==0,:] = X_resampled[y_resampled==0,:].copy()
    return X_resampled, y_resampled

