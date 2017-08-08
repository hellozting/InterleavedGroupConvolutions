import mxnet as mx

def get_conv(name, data, kout, kernel, stride, pad, primary_partition=1,relu=True):
    data = mx.symbol.Convolution(name=name+'_conv', data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True,num_group=primary_partition)
    data = mx.symbol.BatchNorm(name=name + '_bn', data=data, fix_gamma=False, momentum=0.9, eps=2e-5)
    if relu:
        data=mx.symbol.Activation(name=name + '_relu', data=data, act_type='relu')
    return data

def get_igc(name, data, kout, kernel, stride, pad, primary_partition=1,relu=True):
    data = mx.symbol.Convolution(name=name+'_conv1', data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True,num_group=primary_partition)
    data = mx.symbol.Reorder(name=name+'_reorder1', data=data, branch_factor=primary_partition)
    data = mx.symbol.Convolution(name=name+'_conv2', data=data, num_filter=kout, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,num_group=kout/primary_partition)
    data = mx.symbol.Reorder(name=name+'_reorder2', data=data, branch_factor=kout/primary_partition)
    data = mx.symbol.BatchNorm(name=name + '_bn', data=data, fix_gamma=False, momentum=0.9, eps=2e-5)
    if relu:
        data=mx.symbol.Activation(name=name + '_relu', data=data, act_type='relu')
    return data

def get_deep(name, data, kin, kout, stride,primary_partition,relu=False):
    data = get_igc(name=name+'_igc1', data=data, kout=kout, kernel=(3, 3), stride=stride, pad=(1, 1),primary_partition=primary_partition)
    data = get_igc(name=name+'_igc2', data=data, kout=kout, kernel=(3, 3), stride=(1, 1), pad=(1, 1),primary_partition=primary_partition, relu=False)
    return data
    
def get_shortcut(name, data, kin, kout, stride,primary_partition):
    if kin==kout:
        shortcut = data
    else:
        shortcut = get_conv(name=name+'_proj', data=data, kout=kout, kernel=(1, 1), stride=stride, pad=(0, 0), primary_partition=primary_partition, relu=False)
    return shortcut
    
def get_fusion(name, data, kin, kout, stride, primary_partition):
    shortcut= get_shortcut(name, data, kin, kout, stride, primary_partition=primary_partition)
    deep   = get_deep(name, data, kin, kout, stride, primary_partition=primary_partition)
    fusion = shortcut + deep
    fusion = mx.symbol.Activation(name=name+'_relu', data=fusion, act_type='relu')
    return fusion

def get_group(name,data,num_block,kin,kout,stride, primary_partition):
    for idx in range(num_block):
        data = get_fusion(name=name+'_b%d'%(idx+1), data=data, kin=kin, kout=kout, stride= stride if idx == 0 else (1, 1), primary_partition=primary_partition)
        kin=kout
    return data


def get_symbol(num_classes=1000, net_depth=18, primary_partition=1, secondary_partition=1):
    # setup model parameters
    model_cfgs = {
        18: (2,2,2,2)
    }
    blocks_num = model_cfgs[net_depth]
    channels=1*primary_partition*secondary_partition
    
    # start network definition
    data = mx.symbol.Variable(name='data')
    # stage conv1_x
    conv1 = get_conv(name='g0', data=data, kout=channels, kernel=(7, 7), stride=(2, 2), pad=(3, 3))
    pool1 = mx.symbol.Pooling(name='g0_pool', data=conv1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    # stage conv2_x, conv3_x, conv4_x, conv5_x
    conv2_x=get_group(name='g1', data=pool1  , num_block=blocks_num[0], kin=channels,  kout=channels*2, stride=(1,1),primary_partition=primary_partition)
    conv3_x=get_group(name='g2', data=conv2_x, num_block=blocks_num[1], kin=channels*2, kout=channels*4, stride=(2,2), primary_partition=primary_partition*2)
    conv4_x=get_group(name='g3', data=conv3_x, num_block=blocks_num[2], kin=channels*4, kout=channels*8, stride=(2,2), primary_partition=primary_partition*4)
    conv5_x=get_group(name='g4', data=conv4_x, num_block=blocks_num[3], kin=channels*8, kout=channels*16, stride=(2,2), primary_partition=primary_partition*8)
    
    avg = mx.symbol.Pooling(name='global_pool', data=conv5_x, kernel=(7, 7), stride=(1, 1), pool_type='avg')
    flatten = mx.sym.Flatten(name="flatten", data=avg)
    fc = mx.symbol.FullyConnected(name='fc_score', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=fc)
    
    return softmax