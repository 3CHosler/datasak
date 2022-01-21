#Brian Hosler
#January 2022

import cv2
import numpy as np
import hos264

import util




def imagePatches(fpath, w=128):
    '''
    '''
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    patches, locs = util.extract_patches(img,w)
    return patches

def videoPatches(fpath, frames=[0], w=128):
    '''
    '''
    patches=[]
    if type(frames)==int: frames=list(frames)
    vid = cv2.VideoCapture(fpath)
    f0 = min(frames)
    vid.set(cv2.CAP_PROP_POS_FRAMES,f0)
    while vid.get(cv2.CAP_PROP_POS_FRAMES)>f0:
        f0-=3
        vid.set(cv2.CAP_PROP_POS_FRAMES,f0)
    for fi in range(f0,max(frames)+1):
        if fi in frames:
            check, img = vid.retrieve()
            if not check:
                vid.grab()
                continue
            img = cv2.cvtColor(cv2.COLOR_BGR2RGB)
            pats, locs = util.extract_patches(img,w)
            patches.extend(pats)
        vid.grab()
    vid.release()
    return patches

#TODO: Test
def frameTypePatches(fpath, ftype='I', num_frames=3, w=128):
    '''
    '''
    vid = hos264.video(fpath)
    ft = vid.frameTypes()
    vid.close()
    frames = np.array(ft==ftype).nonzero()
    num_frames = min(num_frames,frames.shape[0])
    frames = np.random.choice(frames, num_frames, replace=False)
    return videoPatches(fpath, frames, w=w)

def featureGenerator(vidlist, feat_func, label, threads=0):
    if threads>0:
        return threadedGetFeatureDataset(vidlist, feat_func, label, threads)
    else:
        return getFeatureDataset(vidlist, feat_func, label)

def getFeatureDataset(vidlist, feat_func, label):
    '''
    '''
    for vid in vidlist:
        feats = feat_func(vid)
        for patch in feats:
            yield patch, label

def threadedGetFeatureDataset(vidlist, feat_func, label, num_threads):
    pool = [None]*num_threads
    ndx=0
    with ThreadPoolExecutor() as exe:
        for vid in vidlist:
            pool[ndx] = exe.submit(vid, feat_func, **kwargs)
            ndx+=1
            ndx%=num_threads
            if pool[ndx] is not None:
                patches = pool[ndx].result()
                pool[ndx] = None
                for p in patches:
                    yield p, label
        for k in range(num_threads)
            ndx+=1
            ndx%=num_threads
            if pool[ndx] is not None:
                patches = pool[ndx].result()
                pool[ndx] = None
                for p in patches:
                    yield p, label





