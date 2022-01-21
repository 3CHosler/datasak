#Brian Hosler
#January 2022

import os
from tempfile import TemporaryDirectory
from random import randrange as randint

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import util
from . import record


def _saveDatabase(data,filepath='dataSplit.pkl'):
    '''
    Given a dictionary of available image paths, store into a pickle.
    Defaults to a "save" directory under the current directory
    '''
    with open(filepath, 'wb') as saveFile:
        pickle.dump(data, saveFile, -1)

def _loadDatabase(filepath='dataSplit.pkl'):
    '''
    Given a filepath to a pickle, open it and return the data
    '''
    if os.path.exists(filepath):
        with open(filepath,'rb') as saveFile:
            data=pickle.load(saveFile)
        return data
    else:
        print(f"I can't find {filepath}")

def _makeDatabase(path, classes=None, EXTS=None):
    '''
    '''
    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a valid directory")
    #Search for all appropriate media files(images/videos)
    all_files = util.listAll(path, EXTS)
    #If classes aren't supplied, assume they're immediate subdirectories
    if classes is None: classes=next(os.walk(path))[1]
    #Divide all files by their class/label
    org_data={cls:[] for cls in classes}
    #TODO: switch these around and put in a break statement
    for fpath in all_files:
        for cls in classes:
            if cls in fpath:
                org_data[cls].append(fpath)
                break
    #Divide each class into training and testing
    for cls, files in org_data.items():
        n = len(files)//10
        org_data[cls]={
                'train':files[n:],
                'val':files[:n]}
    return org_data

def getDatabase(path, classes=None, new=None, EXTS=None, save=True):
    '''
    '''
    #First, check if there is an existing split and if the user wants it
    if new==False:
        return _loadDatabase(path)
    elif new is None and os.path.isfile(path):
        return _loadDatabase(path)
    elif new is None and os.path.isdir(path):
        return _loadDatabase(os.path.join(path,'dataSplit.pkl'))
    #Ok, we're going to make a new file then
    database = _makeDatabase(path, classes, EXTS)
    if save:
        _saveDatabase(database)
    return database

#TODO investigate hasattr(obj, attr-string) for user-defined generators
def writeTFdataset(dataGen, outfile, limit=0):
    '''
    Given a generator, store into a tfrecords
    '''
    #Make a writer
    writer = tf.io.TFRecordWriter(outfile)
    #decide if progress will have target number. no need to be correct
    progress = tqdm(desc=f"Writing to {outfile}", total=(limit if limit>0 else None))
    for i,dtm in enumerate(dataGen):
        example = record.pack(dtm)
        writer.write(example.SerializeToString())
        progress.update()
        if i==limit-1: break
    progress.close()
    writer.close()

#TODO: smart dims
#TODO: warn when cycle_length*block_length is greater than shuffbuff
def TFdataset(fname, batch_size=16, dims=128, rpt=True, shuffbuff=2000):
    '''
    Open a tfrecords file, shuffles, batches, and returns an iterator
    '''
    unpack = lambda x:record.unpack(x, dims)
    #Determine if we're reading from multiple datasets
    if isinstance(fname,(list,tuple)) and len(fname)>1:
        dataset = (tf.data.Dataset.from_tensor_slices(fname).interleave(
            lambda x:tf.data.TFRecordDataset(x).map(unpack, num_parallel_calls=4),
            cycle_length=len(fname),
            block_length=batch_size))
    else:
        if isinstance(fname,(list,tuple)):
            fname=fname[0]
        dataset = tf.data.TFRecordDataset([fname],num_parallel_reads=8)
        dataset = dataset.map(unpack)
    #Apply shuffle, batch, repeat
    dataset = dataset.shuffle(shuffbuff).prefetch(200)
    if batch_size>1:
        dataset = dataset.batch(batch_size)
    return dataset

#TODO: backend.tqdm?
def shuffle_record(fpath, num_shards=10, new=None):
    '''
    Divide the file fpath randomly into 10 shards, then recombine them
    '''
    with TemporaryDirectory() as tmpdir:
        shards = [f"{os.path.join(tmpdir,str(i))}.tfr" for i in range(num_shards)]
        #Distribute
        tfile = tf.data.TFRecordDataset(fpath)
        writer = [tf.io.TFRecordWriter(s) for s in shards]
        for example in tqdm(tfile):
            k = randint(num_shards)
            writer[k].write(example.numpy())
        for w in writer: w.close()
        #Recombine
        if new is None: new=fpath
        tfile = [tf.data.TFRecordDataset(s) for s in shards]
        writer = tf.io.TFRecordWriter(new)
        for f in tfile:
            for xmple in f:
                writer.write(xmple.numpy())
        writer.close()
    return new



