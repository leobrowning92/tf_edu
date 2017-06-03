from PIL import Image
import numpy as np
import tensorflow as tf
import glob,os,sys





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

class Dataset(object):
    def __init__(self,savedir):
        self.fnames=glob.glob(savedir+"*.png")
        self.number=len(self.fnames)
        self.fnames.sort()
        self.savedir=savedir
        self.info=np.genfromtxt(os.path.join(self.savedir,"info.txt"), delimiter=',')
        self.truths=np.not_equal(self.info,0)
        self.images=self.load_images()
        self.images=np.reshape(self.images,(self.number,1024,4))
        self._epochs_completed=0
        self._shuffled_images=self.images
        np.random.shuffle(self._shuffled_images)

    def load_images(self):
        images=[]
        for fname in self.fnames:
            images.append(imageto_array(fname))
        images=np.stack(images,axis=0)
        return images
    def get_data(self):
        return self.images[:,:,0],self.images[:,:,1]
    def next_batch(self,batch_size):
        start=self._epochs_completed*batch_size
        stop=(self._epochs_completed+1)*batch_size
        if stop>=self.number:
            self._shuffled_images=self.images
            np.random.shuffle(self._shuffled_images)
            self._epochs_completed=1
            return self.images[start:stop,:,0],self.images[start:stop,:,1]
        if stop<self.number:
            self._epochs_completed+=1
            return self.images[start:stop,:,0],self.images[start:stop,:,1]




if __name__=="__main__":
    assert len(sys.argv)==2, "image-gen takes exactly 1 arguments, {} given.".format(len(sys.argv)-1)
    d=Dataset(sys.argv[1])
    print(d.images.shape)
    print(d.truths[:10,:])
