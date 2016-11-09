import os
import numpy as np
import sys
import caffe
import matplotlib.pyplot as plt
import google.protobuf
from PIL import Image
from caffe import draw

def train(solver_filename):
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_filename)
    print(solver)
    print(solver_filename)
    solver.solve()

def print_network_paramerters(net):
    print(net)
    print('net.inputs: {0}'.format(net.inputs))
    print('net.outputs: {0}'.format(net.outputs))
    print('net.blobs: {0}'.format(net.blobs))
    print('net.params: {0}'.format(net.params)) 

def get_predicted_output(
        deploy_prototxt_filename,
        caffemodel_filename,
        input, 
        net = None
    ):
    '''
    Get the predicted output, i.e. perform a forward pass
    '''
    if net is None:
        net = caffe.Net(deploy_prototxt_filename,caffemodel_filename, caffe.TEST)

    #input = np.array([[ 5.1,  3.5,  1.4,  0.2]])
    #input = np.random.random((1, 1, 1))
    #print(input)
    #print(input.shape)
    out = net.forward(data=input)
    #print('out: {0}'.format(out))
    return out[net.outputs[0]]

def print_network(prototxt_filename):
    '''
    Draw the ANN architecture
    '''
    _net = caffe.proto.caffe_pb2.NetParameter()
    f = open(prototxt_filename)
    google.protobuf.text_format.Merge(f.read(), _net)
    draw.draw_net_to_file(_net, prototxt_filename + '.png' )
    print('Draw ANN done!')




solver_prototxt_filename = 'solverGazeModel.prototxt'
train_test_prototxt_filename = 'traintestGazeModel.prototxt'
deploy_prototxt_filename = 'deployGazeModel.prototxt'
hdf5_train_data_filename = 'testfile1.hdf5'
hdf5_test_data_filename = 'testfile1.hdf5'

train(solver_prototxt_filename)