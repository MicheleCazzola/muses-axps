import torch
import random
import numpy as np

from functools import partial

# Helper functions to set random seeds for reproducibility in training and evaluation, including for DataLoader workers
def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_reproducibility(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    worker_init_fn = partial(seed_worker)
    
    return g, worker_init_fn