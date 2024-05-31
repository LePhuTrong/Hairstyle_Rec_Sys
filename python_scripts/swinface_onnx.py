# libraries
from __future__ import division
import datetime
import os
import os.path as osp
import glob
import numpy as np
import cv2
import sys
import onnxruntime
import onnx
import argparse
from onnx import numpy_helper

class SwinFaceORT:
    def __init__(self, model_path, cpu=False):
        self.model_path = model_path
        # providers = None will use available provider, for onnxruntime-gpu it will be "CUDAExecutionProvider"
        self.providers = ['CPUExecutionProvider'] if cpu else None

    #input_size is (w,h), return error message, return None if success
    def check(self, test_img = None):

        if not os.path.exists(self.model_path):
            return "model_path not exists"
        if not os.path.isdir(self.model_path):
            return "model_path should be directory"
        onnx_files = []
        for _file in os.listdir(self.model_path):
            if _file.endswith('.onnx'):
                onnx_files.append(osp.join(self.model_path, _file))
        if len(onnx_files)==0:
            return "do not have onnx files"
        self.model_file = sorted(onnx_files)[-1]
        print('use onnx-model:', self.model_file)
        try:
            session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)
        except:
            return "load onnx failed"
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        print('input-shape:', input_shape)
        if len(input_shape)!=4:
            return "length of input_shape should be 4"
        if not isinstance(input_shape[0], str):
            #return "input_shape[0] should be str to support batch-inference"
            print('reset input-shape[0] to None')
            model = onnx.load(self.model_file)
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            new_model_file = osp.join(self.model_path, 'zzzzrefined.onnx')
            onnx.save(model, new_model_file)
            self.model_file = new_model_file
            print('use new onnx-model:', self.model_file)
            try:
                session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)
            except:
                return "load onnx failed"
            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
            print('new-input-shape:', input_shape)

        self.image_size = tuple(input_shape[2:4][::-1])
        #print('image_size:', self.image_size)
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
            print(o.name, o.shape)
        # if len(output_names)!=1:
        #     return "number of output nodes should be 1"
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        #print(self.output_names)
        model = onnx.load(self.model_file)
        graph = model.graph
        if len(graph.node)<8:
            return "too small onnx graph"

        input_size = (112,112)
        self.crop = None
        if input_size!=self.image_size:
            return "input-size is inconsistant with onnx model input, %s vs %s"%(input_size, self.image_size)

        self.model_size_mb = os.path.getsize(self.model_file) / float(1024*1024)


        input_mean = None
        input_std = None
        if input_mean is not None or input_std is not None:
            if input_mean is None or input_std is None:
                return "please set input_mean and input_std simultaneously"
        else:
            find_sub = False
            find_mul = False
            for nid, node in enumerate(graph.node[:8]):
                print(nid, node.name)
                if node.name.startswith('Sub') or node.name.startswith('_minus'):
                    find_sub = True
                if node.name.startswith('Mul') or node.name.startswith('_mul') or node.name.startswith('Div'):
                    find_mul = True
            if find_sub and find_mul:
                print("find sub and mul")
                #mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                input_mean = 127.5
                input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        for initn in graph.initializer:
            weight_array = numpy_helper.to_array(initn)
            dt = weight_array.dtype
            if dt.itemsize<4:
                return 'invalid weight type - (%s:%s)' % (initn.name, dt.name)
    
    def forward(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.image_size
        if self.crop is not None:
            nimgs = []
            for img in imgs:
                nimg = img[self.crop[1]:self.crop[3],self.crop[0]:self.crop[2],:]
                if nimg.shape[0]!=input_size[1] or nimg.shape[1]!=input_size[0]:
                    nimg = cv2.resize(nimg, input_size)
                nimgs.append(nimg)
            imgs = nimgs
        blob = cv2.dnn.blobFromImages(imgs, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name : blob})[-1]
        return net_out
    
    def compare(self, embedding1, embedding2):
        embedding1, embedding2 = embedding1.squeeze(), embedding2.squeeze()
        return 1- np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def compare_with_db(self, embedding_1, embeeings_db):
        list_of_dist = []
        for embedding_2 in embeeings_db:
            list_of_dist.append(self.compare(embedding_1, embedding_2))
        return np.array(list_of_dist)
    
def main(args):
    model = SwinFaceORT(args.model_root, cpu=False)
    error = model.check()
    if error is not None:
        print('error:', error)
        return
    img = cv2.imread(args.image_path)
    if img is None:
        print('read image failed:', args.image_path)
        return
    net_out = model.forward(img)
    print('net_out:', net_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do onnx test')
    # general
    parser.add_argument('--model-root', default='', help='path to load model.')
    parser.add_argument('--image-path', default='/train_tmp/IJB_release/IJBC', type=str, help='')

    main(parser.parse_args())