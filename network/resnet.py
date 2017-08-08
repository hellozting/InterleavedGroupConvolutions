'''
Resnet with interleaved group convolutions
'''
import mxnet as mx
from base_gcwf_noshare import FuseNet

def get_conv(name, data, kout, kernel, stride, pad, relu=True):
    #Conv-BN-ReLU style
    data=mx.sym.Convolution(name=name+'_conv', data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)   
    data=mx.sym.BatchNorm(name=name + '_bn', data=data, fix_gamma=False, momentum=0.9, eps=2e-5)
    if relu:
        data=mx.sym.Activation(name=name + '_relu', data=data, act_type='relu')
    return data


def get_two(name, data, kin, kout, primary_partition, secondary_partition):
    data = get_conv(name+'_two1', data, kout, kernel=(3, 3), stride=(1,1) if kin==kout else (2, 2), pad=(1, 1))
    data = get_conv(name+'_two2', data, kout, kernel=(3, 3), stride=(1,1), pad=(1, 1), relu=False)
    return data

#identity shortcut
def get_zero(name, data, kin, kout):
    if kin!=kout:
        data = get_conv(name+'_line', data, kout, kernel=(1, 1), stride=(2, 2), pad=(0, 0), relu=False)
    return data

def get_fusion(name, data, kin, kout):
    shortcut= get_zero(name+'_p0', data, kin, kout)
    two = get_two(name+'_p2', data, kin, kout)
    #resnet style: identity + two convs
    data = shortcut+two
    data = mx.symbol.Activation(name=name+'_relu', data=data, act_type='relu')
    return data

def get_group(name, data, count, kin, kout):
    for idx in range(count):
        data = get_fusion(name=name+'_b%d'%(idx+1), data=data, kin=kin, kout=kout)
        kin=kout
    return data

def get_symbol(num_classes, num_depth, primary_partition, secondary_partition=1):
	# setup model parameters  
    block_depth =2
    num_groups  =3
    exclude_depth=2
    num_blocks=[]
    for i in range(num_groups):
        cur_num=(num_depth-exclude_depth)/(block_depth*(num_groups-i))
        exclude_depth+=cur_num*block_depth
        num_blocks.append(cur_num)
    
    num_channels=1*secondary_partition*primary_partition 
    increase_scale=2
    num_filters=[num_channels, num_channels]
    for i in range(1, len(num_blocks)):
        num_filters.append(num_filters[i]*increase_scale)

	# start network definition
    data = mx.symbol.Variable(name='data')
    # first convolution
    data=get_conv('g0', data, kout=num_filters[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    # different blocks
    data=get_group('g1', data, num_blocks[0], num_filters[0], num_filters[1])
    data=get_group('g2', data, num_blocks[1], num_filters[1], num_filters[2])
    data=get_group('g3', data, num_blocks[2], num_filters[2], num_filters[3])
    # classification layer
    avg = mx.sym.Pooling(name='pool', data=data, kernel=(8, 8), stride=(1, 1), pool_type='avg', global_pool=True)
    flatten = mx.sym.Flatten(name='flatten', data=avg)
    fc = mx.sym.FullyConnected(name='fc', data=flatten, num_hidden=num_classes)
    softmax = mx.sym.SoftmaxOutput(name='softmax', data=fc)
    return softmax