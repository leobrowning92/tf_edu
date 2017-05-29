from PIL import Image
import numpy as np
import tensorflow as tf
import glob




def imageto_array(image_path,v=False):
    im = Image.open(image_path)
    arrayim =  np.asarray(im)
    if v:
        print(im.size,im.format,im.mode)
        print(arrayim.shape)
    im.close()
    return arrayim
def imageto_tensor(array):
    return tf.convert_to_tensor(array)
im=imageto_array("000000.png")

class Dataset(object):
    def __init__(self):
        self.fnames=[]

    def load_images(self):
