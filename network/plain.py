'''
Plain network with interleaved group convolutions
'''
import mxnet as mx

def get_conv(name, data, kout, kernel, stride, pad):
    #Conv-BN-ReLU style
    data = mx.symbol.Convolution(name=name+'_conv', data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    data = mx.symbol.BatchNorm(name=name + '_bn', data=data, fix_gamma=False, momentum=0.99, eps=2e-5)
    data = mx.symbol.Activation(name=name + '_relu', data=data, act_type='relu')
    return data    

def get_group(name,data,num_block,kin,kout):
    for idx in range(num_block):
        data = get_conv(name=name+'_b%d'%(idx+1), data=data, kout=kout, kernel=(3,3), stride=(1,1) if kin==kout else (2,2),pad=(1,1))
        kin=kout
    return data


def get_symbol(num_classes, net_depth,primary_partition,secondary_partition=1):
    # setup model parameters
    block3_num=(net_depth-2)/3
    block2_num=(net_depth-2)/3
    block1_num=(net_depth-2)/3
    blocks_num=(block1_num,block2_num,block3_num)
    if net_depth!=((block1_num+block2_num+block3_num)*1+2) or block3_num<=0:
        print 'invalid depth number: %d'%net_depth,', blocks numbers: ',blocks_num
        return
    # start network definition
    data = mx.symbol.Variable(name='data')
    channel=1*secondary_partition*primary_partition
    
    data=get_conv('g0', data, kout=channel, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    data=get_group('g1', data, num_block=blocks_num[0], kin=channel*1, kout=channel*1)
    data=get_group('g2', data, num_block=blocks_num[1], kin=channel*1, kout=channel*2)
    data=get_group('g3', data, num_block=blocks_num[2], kin=channel*2, kout=channel*4)

    avg = mx.symbol.Pooling(name='global_pool', data=data, kernel=(8,8), stride=(1, 1), pool_type='avg')
    flatten = mx.sym.Flatten(name="flatten", data=avg)
    fc = mx.symbol.FullyConnected(name='fc_score', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=fc)
    
    return softmax