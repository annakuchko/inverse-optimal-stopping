import numpy as np
import torch


def random_batch_split(indexes, batch_size, stratify=None):
    """Split indexes into batches of size batch_size. Optionally stratify using an array-like """
    batches = []
    ids_left = np.array(indexes).copy()
    if stratify is not None:
        stratify = np.array(stratify)
    while len(ids_left) > batch_size:
        if stratify is not None:
            _, batch_ids = __import__('sklearn.model_selection').model_selection.train_test_split(
                ids_left, test_size=batch_size, stratify=stratify[ids_left]
            )
        else:
            _, batch_ids = __import__('sklearn.model_selection').model_selection.train_test_split(
                ids_left, test_size=batch_size
            )
        batches.append(batch_ids)
        ids_left = np.setdiff1d(ids_left, batch_ids)
    if len(ids_left) > 0:
        batches.append(list(ids_left))
    return batches


def batch_to_tensors(batch_dict, keys, ids, device='cpu', mean=None, std=None):
    """Convert selected keys from batch_dict at positions ids into a list of tensors in the same order as keys.

    Normalizes `state_mem` and `next_state_mem` when mean/std are provided. Moves tensors to device once.
    Returns list of tensors.
    """
    ids = np.array(ids)
    tensors = []
    # Pre-create mean/std tensors if provided
    mean_t = torch.from_numpy(np.array(mean)).float().to(device) if mean is not None else None
    std_t = torch.from_numpy(np.array(std)).float().to(device) if std is not None else None
    for k in keys:
        arr = np.take(np.array(batch_dict[k]), ids, axis=0)
        # ensure numpy array
        arr = np.array(arr)
        # convert to float tensor
        try:
            t = torch.from_numpy(arr).float()
        except Exception:
            # fallback for ragged/object arrays
            t = torch.tensor(arr.tolist(), dtype=torch.float32)
        # normalize if requested and key corresponds to states
        if mean_t is not None and std_t is not None and k in ('state_mem', 'next_state_mem'):
            # broadcast mean/std
            t = (t - mean_t) / std_t
        t = t.to(device)
        tensors.append(t)
    return tensors
