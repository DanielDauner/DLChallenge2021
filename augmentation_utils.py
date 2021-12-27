import numpy as np


def delete_random_patches(arr):
    
    num_patches = np.random.randint(0,12)
    patch_size = np.random.randint(17,20)
    for _ in range(num_patches):
        i,j = np.random.randint(0,96-patch_size),np.random.randint(0,96-patch_size)
        arr[i:i+patch_size, j:j+patch_size,:] = 0
    
    return arr



