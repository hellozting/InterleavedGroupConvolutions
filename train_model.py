'''
Train model on Cifar10, Cifar100, and SVHN.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
import argparse
import os
import sys
import logging
import numpy as np
import options
import utility

def get_iterator(args, kv):
    base_args=dict(
        num_parts=kv.num_workers,
        part_index=kv.rank,
        # Image normalization
        data_shape=(3, args.data_shape, args.data_shape),
        # subtract mean and divide std
        mean_r=args.mean_rgb[0],
        mean_g=args.mean_rgb[1],
        mean_b=args.mean_rgb[2],
        fill_value_r=int(round(args.mean_rgb[0])),
        fill_value_g=int(round(args.mean_rgb[1])),
        fill_value_b=int(round(args.mean_rgb[2])),
        scale_r=1.0 / args.std_rgb[0],
        scale_g=1.0 / args.std_rgb[1],
        scale_b=1.0 / args.std_rgb[2],
    )
    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir+args.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        #image augmentation
        pad=4 if args.aug_type==1 else 0,
        rand_crop=True if args.aug_type!=0 else False,
        rand_mirror=True if args.aug_type!=0 else False,
        **base_args #base arguments for mxnet
    )
    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir+args.val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        rand_crop=False,
        rand_mirror=False,
        **base_args #base arguments for mxnet
    )
    return (train, val)

class Init(mx.init.Xavier):
    def __init__(self, widen_factor=1, branch_factor=1, rnd_type="uniform", factor_type="avg", magnitude=3):
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)
        self.branch_factor=branch_factor
        self.widen_factor=widen_factor

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)


def train(args):
    network=options.get_network(args)
    #device
    kv = mx.kvstore.create(args.kv_store)    
    devs = [mx.gpu(i) for i in range(len(args.gpus.split(',')))]
    #training data
    (train_data, val_data)=get_iterator(args, kv)
    #model
    model = mx.model.FeedForward(
        ctx=devs,
        symbol=network,
        num_epoch=args.num_epochs,
        learning_rate=args.lr,
        momentum=0.9,
        wd=0.0001,
        optimizer='Nesterov', #'nag',
        initializer=mx.init.Mixed(['.*fc.*','.*'],
            [mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=1),
            Init(widen_factor=args.widen_factor, branch_factor=args.branch_factor,rnd_type='gaussian', factor_type='in', magnitude=2)]),
        lr_scheduler=utility.Scheduler(epoch_step=args.lr_steps, factor=args.lr_factor, epoch_size=args.num_examples / args.batch_size),
        **args.model_args #for retrain
    )
    model.fit(
        X=train_data,
        eval_data=val_data,
        eval_metric=['ce','acc'] if args.dataset!='imagenet' else ['ce','acc',mx.metric.create('top_k_accuracy',top_k=5)],
        kvstore=kv,
        batch_end_callback=utility.InfoCallback(args.batch_size, args.log_iters),
        epoch_end_callback=mx.callback.do_checkpoint(args.model_prefix,args.checkpoint_epochs),
    )

def main(argv):
    args = options.get_args(argv)
    train(args)

if __name__ == '__main__':
    main(sys.argv[1:])