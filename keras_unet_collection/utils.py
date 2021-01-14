import numpy as np
from PIL import Image

def image_to_array(filenames, size, channel):
    '''
    
    Input
    ----------
        filenames: an iterable of the path of image files
        size: the output size (height == width) of image. 
              Processed through PIL.Image.NEAREST
        channel: number of image channels, e.g. channel=3 for RGB.
        
    Output
    ----------
        An array with shape = (filenum, size, size, channel)
        
    '''
    
    # number of files
    L = len(filenames)
    
    # allocation
    out = np.empty((L, size, size, channel))
    
    # loop over filenames
    if channel == 1:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ..., 0] = np.array(pix)
    else:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ...] = np.array(pix)[..., :channel]
    return out[:, ::-1, ...]

def shuffle_ind(L):
    '''
    Input
    ----------
        L: an int that defines the largest index
        
    Output
    ----------
        a numpy array of shuffled indices with shape = (L,)
    '''
    
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind
