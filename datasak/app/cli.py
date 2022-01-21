#!/usr/bin/env python3
#Brian Hosler
#January 2021

import os
import sys
import argparse as AP
from random import randrange as randint

from tqdm import tqdm
import tensorflow as tf

import rename
from .. import io
from .. import features
from ..features import featureGenerator

def parse_args():
    parse = AP.ArgumentParser(description='', epilog='')
    #Allow subparsers, and retrieve the generic arguments for them to inherit
    subparsers = parse.add_subparsers(title='Commands', dest='subcomm',
        help='The following functionalities are available.')

    shuffle_parser = subparsers.add_parser('shuffle',
            help="Shuffle one or more tfrecords files.")
    shuffle_parser.add_argument('-s', '--shards', type=int, default=10,
            help="The number of shards to use while shuffling.")
    shuffle_parser.add_argument('-r', '--shuffles', type=int, default=1,
            help="The number of times to repeat shuffle")
    shuffle_parser.add_argument('--keep-original', action='store_true',
            help="By default, original files are overwritten, this flag will not overwrite files not ending in '.shuffle'.")
    shuffle_parser.add_argument('records', nargs='+', default=[],
            help="One or more filepaths to records.")

    summary_parser = subparsers.add_parser('summary',
            help="Compute and report statistics about tfrecords files.")
    summary_parser.add_argument('-c', '--count', action='store_true',
            help="Report the number of examples the file comprises.")
    summary_parser.add_argument('records', nargs='+', default=[],
            help="One or more filepaths to records.")

    writer_parser = subparsers.add_parser('new',
            help="Compile new tfrecords files")
    writer_parser.add_argument('-n', '--num', type=int, default=100,
            help="Number of training samples to be saved per file.")
    writer_parser.add_argument('--classes', default='classes.txt',
            help="Path to a text file containing class labels. If not provided, subdirectories will be used.")
    writer_parser.add_argument('-s', '--shuffle', action='count',
            help="Shuffle the records after creation. Multiple flags result in multiple shuffles.")
    writer_parser.add_argument('--format', nargs=1,
            help="Optional. Path to a module to define record packing and unpacking behavior.")
    writer_parser.add_argument('--feature', nargs=1,
            help="Optional. Path to module which takes a filepath and returns features.")
    writer_parser.add_argument('data',
            help="Path to the dataset or database from which to make records.")

    return parse.parse_args()

def main():
    print(f"Executing function main()")
    args = parse_args()
    if args.subcomm=='shuffle':
        shuffle(args)
    elif args.subcomm=='summary':
        summary(args)
    elif args.subcomm=='new':
        write(args)

def summary(fpath, metrics=['count']):
    ret = {'name':fpath}
    if 'count' in metrics:
        ret['count'] = util.num_patches(fpath)
    return ret

def shuffle(args):
    '''
    '''
    for f in args.records:
        if args.keep_original and not f.endswith('.shuffle'):
            ret = io.shuffle_record(f, args.shards, f"{f}.shuffle")
        else:
            ret = io.shuffle_record(f, args.shards)
        for i in range(1,args.shuffles):
            ret = io.shuffle_record(ret, args.shards)

#TODO: feature extraciton
def write(args):
    '''
    '''
    if args.format is not None:
        rename.record.set_recordFormat(args.format)
    if args.feature is not None:
        pass
    org_data = getDatabase(args.data, classes)
    #Determine appropriate feature extractor
    get_features = lambda x: features.frameTypePathces(x, ftype='I', num_frames=3, w=128)
    for i, (cls,vidlist) in enumerate(org_data.items()):
        datagen = featureGenerator(vidlist['train'], get_features, i, threads=10)
        io.writeTFdataset(datagen, f"train.{cls}.tfrecords", limit=args.num)
        datagen = featureGenerator(vidlist['val'], get_features, i, threads=10)
        io.writeTFdataset(datagen, f"val.{cls}.tfrecords", limit=args.num//10)

if __name__=='__main__':
    main()




