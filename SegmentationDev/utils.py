import numpy as np
import PIL.Image
import os
import scipy
import scipy.misc

def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB')
   #if not (len(img.shape) == 3 and img.shape[2] == 3):
   #    img = np.dstack((img,img,img))
   if img_size != False:
#       img = scipy.misc.imresize(img, img_size,interp="lanczos")
       img = scipy.misc.imresize(img, img_size)
   return img

def get_files(img_dir):
    files = list_files(img_dir)
    return list(map(lambda x: os.path.join(img_dir,x), files))

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files

def binarize(image):
    print('len(image)', len(image[0]))
    for i in range(len(image[0])):
        print('process')
        if image[0][i] >200:
            image[0][i] =255
        else:
            image[0][i] = 0


    return image
