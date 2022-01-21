#Brian Hosler
#January 2022


import numpy as np
import tensorflow as tf


def int64_feature(value):
    '''
    Coerce value to a tensorflow Int64List feature
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    '''
    Coerce value to a tensorflow BytesList feature
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    '''
    Coerce value to a tensorflow FloatList feature
    '''
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

def num_patches(dataset):
    if type(dataset)==str:
        dataset = tf.data.TFRecordDataset(dataset)
    return dataset.reduce(np.int64(0),lambda x,_:x+1)

def listAll(searchPath,extension='.jpg'):
    '''
    Search for all files with a given extension under a given
    search directory.
    '''
    fileList=[]
    for pth, dname, fname in os.walk(searchPath,followlinks=True):
        fileList+= [os.path.join(pth,vid) for vid in fname if vid.endswith(extension)]
    return fileList

def extract_patches(img, w=128):
    '''
    '''
    p=[]
    l=[]
    for y in range(0, img.shape[0]-w+1, w):
        for x in range(0, img.shape[1]-w+1, w):
            l.append((y,x))
            p.append(img[y:y+w,x:x+w])
    return np.stack(p), np.stack(l)
