import numpy as np


def delete_random_patches(arr):
    
    num_patches = np.random.randint(5,10)
    patch_size = np.random.randint(17,20)
    div = 96 // patch_size

    for _ in range(num_patches):
        i,j = np.random.randint(0,div+1)*patch_size,np.random.randint(0,div+1)*patch_size
        arr[i:i+patch_size, j:j+patch_size,:] = 0
    
    return arr



