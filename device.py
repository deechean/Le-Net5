#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:23:43 2019

@author: Deechean
"""
from tensorflow.python.client import device_lib as _device_lib

def is_gpu_available(cuda_only=True):
  """
  code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
  Returns whether TensorFlow can access a GPU.
  Args:
    cuda_only: limit the search to CUDA gpus.
  Returns:
    True iff a gpu device of the requested kind is available.
  """
  if cuda_only:
    return any((x.device_type == 'GPU' or x.device_type == 'XLA_GPU')
               for x in _device_lib.list_local_devices())
  else:
    return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
               for x in _device_lib.list_local_devices())

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type in ['GPU', 'XLA_GPU']]