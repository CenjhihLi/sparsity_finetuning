import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from abc import ABC, abstractclassmethod
from typing import Sequence, Tuple
from types import MethodType
from torch_pruning import ops
from torch_pruning.pruner.function import BasePruningFunc

__all__=[
    'BaseFinetuningFunc',
    'FinetunerBox',

    'prune_conv_out_channels',
    'prune_conv_in_channels',
    'prune_depthwise_conv_out_channels',
    'prune_depthwise_conv_in_channels',
    'prune_batchnorm_out_channels',
    'prune_batchnorm_in_channels',
    'prune_linear_out_channels',
    'prune_linear_in_channels',
    'prune_prelu_out_channels',
    'prune_prelu_in_channels',
    'prune_layernorm_out_channels',
    'prune_layernorm_in_channels',
    'prune_embedding_out_channels',
    'prune_embedding_in_channels',
    'prune_parameter_out_channels',
    'prune_parameter_in_channels',
    'prune_multihead_attention_out_channels',
    'prune_multihead_attention_in_channels',
    'prune_groupnorm_out_channels',
    'prune_groupnorm_in_channels',
    'prune_instancenorm_out_channels',
    'prune_instancenorm_in_channels',
]

class BaseFinetuningFunc(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_OTHERS  # None

    def __init__(self, finetuning_dim=1):
        self.finetuning_dim = finetuning_dim
        self.pruning_dim = finetuning_dim

    def _finetune_parameter(self, param, finetune_idxs, finetuning_dim):
        tensor_size = list(param.data.size())
        tensor_size[finetuning_dim] = len(finetune_idxs)
        finetuned_weight = nn.Parameter(torch.zeros(tensor_size, device = param.device))
        return finetuned_weight


class ConvFinetuner(BaseFinetuningFunc):
    TARGET_MODULE = ops.TORCH_CONV

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.out_channels)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        size_mapping = torch.zeros((len(finetune_idxs), layer.out_channels), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        if not layer.transposed:
            finetuning_dim = 0
        else:
            finetuning_dim = 1
        layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim))
        layer.register_buffer('finetuned_out_mapping', size_mapping)

        if layer.bias is not None:
            layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0))
        
        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 1 if self.transposed else 0
                finetuned_out_weight = self.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_out_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_out_weight
            
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 0 if self.transposed else 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_in_weight 

            bias = self.bias
            if bias is not None and hasattr(self, 'finetuned_out_bias'):
                finetuned_out_bias = self.finetuned_out_bias.movedim(0,-1).matmul(self.finetuned_out_mapping).movedim(-1, 0)
                bias = bias + finetuned_out_bias
            return self._conv_forward(input, weight, bias)
        
        layer.forward = MethodType(forward, layer)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.in_channels)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if layer.groups>1:
            finetune_idxs = finetune_idxs[:len(finetune_idxs)//layer.groups]

        size_mapping = torch.zeros((len(finetune_idxs), layer.in_channels//layer.groups), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.
        
        if not layer.transposed:
            finetuning_dim = 1
        else:
            finetuning_dim = 0
        layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim))
        layer.register_buffer('finetuned_in_mapping', size_mapping)
        # no bias because it does not change the output channels
        
        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 1 if self.transposed else 0
                finetuned_out_weight = self.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_out_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_out_weight
            
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 0 if self.transposed else 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_in_weight 

            bias = self.bias
            if bias is not None and hasattr(self, 'finetuned_out_bias'):
                finetuned_out_bias = self.finetuned_out_bias.movedim(0,-1).matmul(self.finetuned_out_mapping).movedim(-1, 0)
                bias = bias + finetuned_out_bias
            return self._conv_forward(input, weight, bias)
        
        layer.forward = MethodType(forward, layer)
        return layer

    def get_out_channels(self, layer):
        return layer.out_channels

    def get_in_channels(self, layer):
        return layer.in_channels

    def get_in_channel_groups(self, layer):
        return layer.groups
    
    def get_out_channel_groups(self, layer):
        return layer.groups


class DepthwiseConvFinetuner(ConvFinetuner):
    TARGET_MODULE = ops.TORCH_CONV

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.out_channels)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        size_mapping = torch.zeros((len(finetune_idxs), layer.out_channels), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0))
        layer.register_buffer('finetuned_mapping', size_mapping)

        if layer.bias is not None:
            layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0))
        
        def forward(self, input):
            weight = self.weight
            finetuning_dim = 0
            if hasattr(self, 'finetuned_weight'):
                finetuned_weight = self.finetuned_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_weight

            bias = self.bias
            if bias is not None and hasattr(self, 'finetuned_bias'):
                finetuned_bias = self.finetuned_bias.movedim(0,-1).matmul(self.finetuned_mapping).movedim(-1, 0)
                bias = bias + finetuned_bias
            return self._conv_forward(input, weight, bias)
        
        layer.forward = MethodType(forward, layer)
        return layer

    prune_in_channels = prune_out_channels


class LinearFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:        
        finetune_idxs = list(set(range(layer.out_features)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        size_mapping = torch.zeros((len(finetune_idxs), layer.out_features), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0))
        layer.register_buffer('finetuned_out_mapping', size_mapping)

        if layer.bias is not None:
            layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0))
        
        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                finetuned_out_weight = self.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_out_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_out_weight
            
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_in_weight 

            bias = self.bias
            if bias is not None and hasattr(self, 'finetuned_out_bias'):
                finetuned_out_bias = self.finetuned_out_bias.movedim(0,-1).matmul(self.finetuned_out_mapping).movedim(-1, 0)
                bias = bias + finetuned_out_bias
            return F.linear(input, weight, bias)
        
        layer.forward = MethodType(forward, layer)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.in_features)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        size_mapping = torch.zeros((len(finetune_idxs), layer.in_features), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.weight, finetune_idxs, 1))
        layer.register_buffer('finetuned_in_mapping', size_mapping)

        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                finetuned_out_weight = self.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_out_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_out_weight
            
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_in_weight 

            bias = self.bias
            if bias is not None and hasattr(self, 'finetuned_out_bias'):
                finetuned_out_bias = self.finetuned_out_bias.movedim(0,-1).matmul(self.finetuned_out_mapping).movedim(-1, 0)
                bias = bias + finetuned_out_bias
            return F.linear(input, weight, bias)
        
        layer.forward = MethodType(forward, layer)
        return layer

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features


class BatchnormFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_BATCHNORM
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.num_features)) - set(idxs))
        finetune_idxs.sort()
        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        size_mapping = torch.zeros((len(finetune_idxs), layer.num_features), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0))
            layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0))
            layer.register_buffer('finetuned_mapping', size_mapping)
        
            def forward(self, input):
                #Reference: torch.nn.module.batchnorm._BatchNorm
                self._check_input_dim(input)
                if self.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = self.momentum

                if self.training and self.track_running_stats:
                    # TODO: if statement only here to tell the jit to skip emitting this when it is None
                    if self.num_batches_tracked is not None:  # type: ignore[has-type]
                        self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                        if self.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                        else:  # use exponential moving average
                            exponential_average_factor = self.momentum
    
                if self.training:
                    bn_training = True
                else:
                    bn_training = (self.running_mean is None) and (self.running_var is None)

                weight = self.weight
                if hasattr(self, 'finetuned_weight'):
                    finetuning_dim = 0
                    finetuned_weight = self.finetuned_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_mapping).movedim(-1, finetuning_dim)
                    weight = weight + finetuned_weight

                bias = self.bias
                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    bias = bias + finetuned_bias

                return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    weight,
                    bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )

            layer.forward = MethodType(forward, layer)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


class LayernormFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_LAYERNORM

    def __init__(self, metrcis=None, finetuning_dim=-1):
        super().__init__(metrcis)
        self.finetuning_dim = finetuning_dim

    def check(self, layer, idxs):
        layer.dim = self.finetuning_dim

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetuning_dim = self.finetuning_dim
        
        num_features = layer.normalized_shape[finetuning_dim]

        finetune_idxs = list(set(range(num_features)) - set(idxs))
        finetune_idxs.sort()
        if layer.elementwise_affine:
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False
        if len(finetune_idxs)==0 or len(layer.normalized_shape) < -finetuning_dim:
            return layer
        size_mapping = torch.zeros((len(finetune_idxs), num_features), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        if layer.elementwise_affine:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim))
            layer.register_buffer('finetuned_mapping', size_mapping)
            layer.register_buffer('finetuning_dim', torch.tensor(finetuning_dim))
            if layer.bias is not None:
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, finetuning_dim))
            
            def forward(self, input):
                weight = self.weight
                if hasattr(self, 'finetuned_weight'):
                    finetuning_dim = self.finetuning_dim.item()
                    finetuned_weight = self.finetuned_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_mapping).movedim(-1, finetuning_dim)
                    weight = weight + finetuned_weight

                bias = self.bias
                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    bias = bias + finetuned_bias
                return F.layer_norm(
                    input, self.normalized_shape, weight, bias, self.eps)
            layer.forward = MethodType(forward, layer)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.normalized_shape[self.finetuning_dim]

    def get_in_channels(self, layer):
        return layer.normalized_shape[self.finetuning_dim]

class GroupNormFinetuner(BaseFinetuningFunc):
    def prune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        finetune_idxs = list(set(range(layer.num_channels)) - set(idxs))
        finetune_idxs.sort()
        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        size_mapping = torch.zeros((len(finetune_idxs), layer.num_channels), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        if layer.affine:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0))
            layer.register_buffer('finetuned_mapping', size_mapping)
            layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0))
            def forward(self, input):
                weight = self.weight
                if hasattr(self, 'finetuned_weight'):
                    finetuned_weight = self.finetuned_weight.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    weight = weight + finetuned_weight

                bias = self.bias
                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    bias = bias + finetuned_bias
                return F.group_norm(
                    input, self.num_groups, weight, bias, self.eps)
            layer.forward = MethodType(forward, layer)
        return layer
    
    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_channels

    def get_in_channels(self, layer):
        return layer.num_channels

    def get_in_channel_groups(self, layer):
        return layer.num_groups
    
    def get_out_channel_groups(self, layer):
        return layer.num_groups
    
class InstanceNormFinetuner(BaseFinetuningFunc):
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.num_features)) - set(idxs))
        finetune_idxs.sort()
        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        size_mapping = torch.zeros((len(finetune_idxs), layer.num_features), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        if layer.affine:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0))
            layer.register_buffer('finetuned_mapping', size_mapping)
            layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0))
            def _apply_instance_norm(self, input):
                weight = self.weight
                if hasattr(self, 'finetuned_weight'):
                    finetuned_weight = self.finetuned_weight.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    weight = weight + finetuned_weight

                bias = self.bias
                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    bias = bias + finetuned_bias
                return F.instance_norm(
                    input, self.running_mean, self.running_var, weight, bias,
                    self.training or not self.track_running_stats, self.momentum, self.eps)

            layer._apply_instance_norm = MethodType(_apply_instance_norm, layer)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


class PReLUFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_PRELU

    def prune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        if layer.num_parameters == 1:
            layer.weight.requires_grad = True
            return layer
        
        finetune_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        size_mapping = torch.zeros((len(finetune_idxs), layer.num_parameters), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0))
        layer.register_buffer('finetuned_mapping', size_mapping)

        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_weight'):
                finetuned_weight = self.finetuned_weight.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                weight = weight + finetuned_weight
            return F.prelu(input, weight)
        layer.forward = MethodType(forward, layer)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        if layer.num_parameters == 1:
            return None
        else:
            return layer.num_parameters

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class EmbeddingFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_EMBED

    def prune_out_channels(self, layer: nn.Embedding, idxs: list) -> nn.Module:
        finetune_idxs = list(set(range(layer.embedding_dim)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        size_mapping = torch.zeros((len(finetune_idxs), layer.embedding_dim), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 1))
        layer.register_buffer('finetuned_mapping', size_mapping)

        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_weight'):
                finetuned_weight = self.finetuned_weight.movedim(1, -1).matmul(self.finetuned_mapping).movedim(-1, 1)
                weight = weight + finetuned_weight
            return F.embedding(
                input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        layer.forward = MethodType(forward, layer)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.embedding_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class LSTMFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_LSTM

    def prune_out_channels(self, layer: nn.LSTM, idxs: list) -> nn.Module:
        assert layer.num_layers==1
        num_features = layer.hidden_size

        finetune_idxs = list(set(range(num_features)) - set(idxs))
        finetune_idxs.sort()
        for param in layer.parameters():
            param.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        finetune_idxs = torch.tensor(finetune_idxs)
        expanded_finetune_idxs = torch.cat([ finetune_idxs+i*num_features for i in range(4) ], dim=0)

        size_mapping = torch.zeros((len(finetune_idxs), num_features), device = layer.weight.device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.
        
        
        
        if layer.bidirectional:
            postfix = ['', '_reverse']
        else:
            postfix = ['']

        layer.register_buffer('finetuned_mapping', size_mapping)

        for pf in postfix:
            layer.register_parameter('finetuned_weight_hh_l0' + pf, 
                                     self._finetune_parameter(getattr(layer, 'weight_hh_l0' + pf), finetune_idxs, 0))
            layer.register_parameter('finetuned_weight_ih_l0' + pf, 
                                     self._finetune_parameter(getattr(layer, 'weight_ih_l0' + pf), expanded_finetune_idxs, 0))
            if layer.bias:
                layer.register_parameter('finetuned_bias_hh_l0' + pf, 
                                         self._finetune_parameter(getattr(layer, 'bias_hh_l0' + pf), finetune_idxs, 0))
                layer.register_parameter('finetuned_bias_ih_l0' + pf, 
                                         self._finetune_parameter(getattr(layer, 'bias_ih_l0' + pf), finetune_idxs, 0))
        #TODO: Figure out what functions should be modified


    def prune_in_channels(self, layer: nn.LSTM, idxs: list):
        num_features = layer.input_size
        finetune_idxs = list(set(range(num_features)) - set(idxs))
        finetune_idxs.sort()
        setattr(layer, 'weight_ih_l0', self._finetune_parameter_and_grad(
                    getattr(layer, 'weight_ih_l0'), finetune_idxs, 1))
        if layer.bidirectional:
            setattr(layer, 'weight_ih_l0_reverse', self._finetune_parameter_and_grad(
                    getattr(layer, 'weight_ih_l0_reverse'), finetune_idxs, 1))
        layer.input_size = len(finetune_idxs)

    def get_out_channels(self, layer):
        return layer.hidden_size
        
    def get_in_channels(self, layer):
        return layer.input_size
    

class ParameterFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_PARAMETER
    def __init__(self, finetuning_dim=-1):
        super().__init__(finetuning_dim=finetuning_dim)
    
    #TODO: try to deal with this in the parameters' belonging module
    def prune_out_channels(self, tensor, idxs: list) -> nn.Module:
        return tensor

    prune_in_channels = prune_out_channels

    def get_out_channels(self, parameter):
        return parameter.shape[self.finetuning_dim]

    def get_in_channels(self, parameter):
        return parameter.shape[self.finetuning_dim]


class MultiheadAttentionFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_MHA

    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        
    def prune_out_channels(self, layer, idxs: list) -> nn.Module:

        finetune_idxs = list(set(range(layer.embed_dim)) - set(idxs))
        finetune_idxs.sort()
        for param in layer.parameters():
            param.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        device = layer.q_proj_weight.device
        size_mapping = torch.zeros((len(finetune_idxs), layer.embed_dim), device = device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        layer.register_buffer('finetuned_in_proj_mapping', size_mapping)

        if layer.q_proj_weight is not None:
            layer.register_parameter('finetuned_q_proj_weight', self._finetune_parameter(layer.q_proj_weight, finetune_idxs, 0))
        if layer.k_proj_weight is not None:
            layer.register_parameter('finetuned_k_proj_weight', self._finetune_parameter(layer.k_proj_weight, finetune_idxs, 0))
        if layer.v_proj_weight is not None:
            layer.register_parameter('finetuned_v_proj_weight', self._finetune_parameter(layer.v_proj_weight, finetune_idxs, 0))
        

        finetune_idxs_3x_repeated = finetune_idxs + \
            [i+layer.embed_dim for i in finetune_idxs] + \
            [i+2*layer.embed_dim for i in finetune_idxs]

        size_mapping = torch.zeros((len(finetune_idxs), 3*layer.embed_dim), device = device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        layer.register_buffer('finetuned_out_proj_mapping', size_mapping)

        if layer.in_proj_weight is not None:
            layer.register_parameter('finetuned_in_proj_weight_out', self._finetune_parameter(layer.in_proj_weight, finetune_idxs_3x_repeated, 0))
            layer.register_parameter('finetuned_in_proj_weight_in', self._finetune_parameter(layer.in_proj_weight, finetune_idxs, 1))
        if layer.in_proj_bias is not None:
            layer.register_parameter('finetuned_in_proj_bias', self._finetune_parameter(layer.in_proj_bias, finetune_idxs_3x_repeated, 0))
        

        if layer.bias_k is not None:
            layer.register_parameter('finetuned_bias_k', self._finetune_parameter(layer.bias_k, finetune_idxs, 2))
        if layer.bias_v is not None:
            layer.register_parameter('finetuned_bias_v', self._finetune_parameter(layer.bias_v, finetune_idxs, 2))

        linear = layer.out_proj
        size_mapping = torch.zeros((len(finetune_idxs), linear.out_features), device = device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        linear.register_buffer('finetuned_out_mapping', size_mapping)
        linear.register_parameter('finetuned_out_weight', self._finetune_parameter(linear.weight, finetune_idxs, 0))

        if linear.bias is not None:
            linear.register_parameter('finetuned_out_bias', self._finetune_parameter(linear.bias, finetune_idxs, 0))

        size_mapping = torch.zeros((len(finetune_idxs), linear.in_features), device = device)

        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.

        linear.register_buffer('finetuned_in_mapping', size_mapping)
        linear.register_parameter('finetuned_in_weight', self._finetune_parameter(linear.weight, finetune_idxs, 1))

        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                finetuned_out_weight = self.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_out_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_out_weight
            
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                weight = weight + finetuned_in_weight 

            bias = self.bias
            if bias is not None and hasattr(self, 'finetuned_out_bias'):
                finetuned_out_bias = self.finetuned_out_bias.movedim(0,-1).matmul(self.finetuned_out_mapping).movedim(-1, 0)
                bias = bias + finetuned_out_bias
            return F.linear(input, weight, bias)
        
        linear.forward = MethodType(forward, linear)

        def forward(
            self,
            query,
            key,
            value,
            key_padding_mask = None,
            need_weights = True,
            attn_mask = None,
            average_attn_weights = True,
            is_causal = False):
        

            why_not_fast_path = ''
            if ((attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
                why_not_fast_path = "floating-point masks are not supported for fast path."

            is_batched = query.dim() == 3

            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="attn_mask",
                target_type=query.dtype
            )

            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            in_proj_weight = self.in_proj_weight
            if hasattr(self, 'finetuned_in_proj_weight_out'):
                finetuning_dim = 0
                finetuned_in_proj_weight_out = self.finetuned_in_proj_weight_out.movedim(finetuning_dim, -1).matmul(self.finetuned_out_proj_mapping).movedim(-1, finetuning_dim)
                in_proj_weight = in_proj_weight + finetuned_in_proj_weight_out
                        
            if hasattr(self, 'finetuned_in_proj_weight_in'):
                finetuning_dim = 1
                finetuned_in_proj_weight_in = self.finetuned_in_proj_weight_in.movedim(finetuning_dim, -1).matmul(self.finetuned_in_proj_mapping).movedim(-1, finetuning_dim)
                in_proj_weight = in_proj_weight + finetuned_in_proj_weight_in 

            in_proj_bias = self.in_proj_bias
            if in_proj_bias is not None and hasattr(self, 'finetuned_in_proj_bias'):
                finetuned_in_proj_bias = self.finetuned_in_proj_bias.movedim(0,-1).matmul(self.finetuned_out_proj_mapping).movedim(-1, 0)
                in_proj_bias = in_proj_bias + finetuned_in_proj_bias
            
            finetuning_dim = 2
            bias_k = self.bias_k
            bias_v = self.bias_v
            if bias_k is not None and hasattr(self, 'finetuned_bias_k'):
                finetuned_bias_k = self.finetuned_bias_k.movedim(finetuning_dim,-1).matmul(self.finetuned_in_proj_mapping).movedim(-1, finetuning_dim)
                bias_k = bias_k + finetuned_bias_k
            if bias_v is not None and hasattr(self, 'finetuned_bias_v'):
                finetuned_bias_v = self.finetuned_bias_v.movedim(finetuning_dim,-1).matmul(self.finetuned_in_proj_mapping).movedim(-1, finetuning_dim)
                bias_v = bias_v + finetuned_bias_v


            if not is_batched:
                why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            elif query is not key or key is not value:
                # When lifting this restriction, don't forget to either
                # enforce that the dtypes all match or test cases where
                # they don't!
                why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
            elif in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
                why_not_fast_path = f"dtypes of query ({query.dtype}) and in_proj_bias ({in_proj_bias.dtype}) don't match"
            elif in_proj_weight is None:
                why_not_fast_path = "in_proj_weight was None"
            elif query.dtype != in_proj_weight.dtype:
                # this case will fail anyway, but at least they'll get a useful error message.
                why_not_fast_path = f"dtypes of query ({query.dtype}) and in_proj_weight ({in_proj_weight.dtype}) don't match"
            elif self.training:
                why_not_fast_path = "training is enabled"
            elif (self.num_heads % 2) != 0:
                why_not_fast_path = "self.num_heads is not even"
            elif not self.batch_first:
                why_not_fast_path = "batch_first was not True"
            elif bias_k is not None:
                why_not_fast_path = "self.bias_k was not None"
            elif bias_v is not None:
                why_not_fast_path = "self.bias_v was not None"
            elif self.add_zero_attn:
                why_not_fast_path = "add_zero_attn was enabled"
            elif not self._qkv_same_embed_dim:
                why_not_fast_path = "_qkv_same_embed_dim was not True"
            elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
                why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                    is not supported with NestedTensor input"
            elif torch.is_autocast_enabled():
                why_not_fast_path = "autocast is enabled"

            linear = self.out_proj
            out_proj_weight = linear.weight
            if hasattr(linear, 'finetuned_out_weight'):
                finetuning_dim = 0
                finetuned_out_weight = linear.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(linear.finetuned_out_mapping).movedim(-1, finetuning_dim)
                out_proj_weight = out_proj_weight + finetuned_out_weight
            
            if hasattr(linear, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = linear.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(linear.finetuned_in_mapping).movedim(-1, finetuning_dim)
                out_proj_weight = out_proj_weight + finetuned_in_weight 

            out_proj_bias = linear.bias
            if out_proj_bias is not None and hasattr(linear, 'finetuned_out_bias'):
                finetuned_out_bias = linear.finetuned_out_bias.movedim(0,-1).matmul(linear.finetuned_out_mapping).movedim(-1, 0)
                out_proj_bias = out_proj_bias + finetuned_out_bias

            if not why_not_fast_path:
                tensor_args = (
                    query,
                    key,
                    value,
                    in_proj_weight,
                    in_proj_bias,
                    out_proj_weight,
                    out_proj_bias,
                )
                # We have to use list comprehensions below because TorchScript does not support
                # generator expressions.
                if torch.overrides.has_torch_function(tensor_args):
                    why_not_fast_path = "some Tensor argument has_torch_function"
                elif nn.modules.activation._is_make_fx_tracing():
                    why_not_fast_path = "we are running make_fx tracing"
                elif not all(nn.modules.activation._check_arg_device(x) for x in tensor_args):
                    why_not_fast_path = ("some Tensor argument's device is neither one of "
                                        f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
                elif torch.is_grad_enabled() and any(nn.modules.activation._arg_requires_grad(x) for x in tensor_args):
                    why_not_fast_path = ("grad is enabled and at least one of query or the "
                                        "input/output projection weights or biases requires_grad")
                if not why_not_fast_path:
                    merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                    if in_proj_bias is not None and in_proj_weight is not None:
                        return torch._native_multi_head_attention(
                            query,
                            key,
                            value,
                            self.embed_dim,
                            self.num_heads,
                            in_proj_weight,
                            in_proj_bias,
                            out_proj_weight,
                            out_proj_bias,
                            merged_mask,
                            need_weights,
                            average_attn_weights,
                            mask_type)

            any_nested = query.is_nested or key.is_nested or value.is_nested
            assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                    f"The fast path was not hit because {why_not_fast_path}")

            if self.batch_first and is_batched:
                # make sure that the transpose op does not affect the "is" property
                if key is value:
                    if query is key:
                        query = key = value = query.transpose(1, 0)
                    else:
                        query, key = (x.transpose(1, 0) for x in (query, key))
                        value = key
                else:
                    query, key, value = (x.transpose(1, 0) for x in (query, key, value))

            q_proj_weight = self.q_proj_weight
            finetuning_dim = 0
            if hasattr(self, 'finetuned_q_proj_weight'):
                finetuned_q_proj_weight = self.finetuned_q_proj_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_proj_mapping).movedim(-1, finetuning_dim)
                q_proj_weight = q_proj_weight + finetuned_q_proj_weight
            
            k_proj_weight = self.k_proj_weight
            finetuning_dim = 0
            if hasattr(self, 'finetuned_k_proj_weight'):
                finetuned_k_proj_weight = self.finetuned_k_proj_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_proj_mapping).movedim(-1, finetuning_dim)
                k_proj_weight = k_proj_weight + finetuned_k_proj_weight

            v_proj_weight = self.v_proj_weight
            finetuning_dim = 0
            if hasattr(self, 'finetuned_v_proj_weight'):
                finetuned_v_proj_weight = self.finetuned_v_proj_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_proj_mapping).movedim(-1, finetuning_dim)
                v_proj_weight = v_proj_weight + finetuned_v_proj_weight

            if not self._qkv_same_embed_dim:
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, in_proj_bias,
                    bias_k, bias_v, self.add_zero_attn,
                    self.dropout, out_proj_weight, out_proj_bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal)
            else:
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, in_proj_bias,
                    bias_k, bias_v, self.add_zero_attn,
                    self.dropout, out_proj_weight, out_proj_bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal)
            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), attn_output_weights
            else:
                return attn_output, attn_output_weights
        
        layer.forward = MethodType(forward, layer)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.embed_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer)

FinetunerBox = {
    ops.OPTYPE.CONV: ConvFinetuner(),
    ops.OPTYPE.LINEAR: LinearFinetuner(),
    ops.OPTYPE.BN: BatchnormFinetuner(),
    ops.OPTYPE.DEPTHWISE_CONV: DepthwiseConvFinetuner(),
    ops.OPTYPE.PRELU: PReLUFinetuner(),
    ops.OPTYPE.LN: LayernormFinetuner(),
    ops.OPTYPE.EMBED: EmbeddingFinetuner(),
    ops.OPTYPE.PARAMETER: ParameterFinetuner(),
    ops.OPTYPE.MHA: MultiheadAttentionFinetuner(),
    ops.OPTYPE.LSTM: LSTMFinetuner(),
    ops.OPTYPE.GN: GroupNormFinetuner(),
    ops.OPTYPE.IN: InstanceNormFinetuner(),
}

# Alias
prune_conv_out_channels = FinetunerBox[ops.OPTYPE.CONV].prune_out_channels
prune_conv_in_channels = FinetunerBox[ops.OPTYPE.CONV].prune_in_channels

prune_depthwise_conv_out_channels = FinetunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_out_channels
prune_depthwise_conv_in_channels = FinetunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_in_channels

prune_batchnorm_out_channels = FinetunerBox[ops.OPTYPE.BN].prune_out_channels
prune_batchnorm_in_channels = FinetunerBox[ops.OPTYPE.BN].prune_in_channels

prune_linear_out_channels = FinetunerBox[ops.OPTYPE.LINEAR].prune_out_channels
prune_linear_in_channels = FinetunerBox[ops.OPTYPE.LINEAR].prune_in_channels

prune_prelu_out_channels = FinetunerBox[ops.OPTYPE.PRELU].prune_out_channels
prune_prelu_in_channels = FinetunerBox[ops.OPTYPE.PRELU].prune_in_channels

prune_layernorm_out_channels = FinetunerBox[ops.OPTYPE.LN].prune_out_channels
prune_layernorm_in_channels = FinetunerBox[ops.OPTYPE.LN].prune_in_channels

prune_embedding_out_channels = FinetunerBox[ops.OPTYPE.EMBED].prune_out_channels
prune_embedding_in_channels = FinetunerBox[ops.OPTYPE.EMBED].prune_in_channels

prune_parameter_out_channels = FinetunerBox[ops.OPTYPE.PARAMETER].prune_out_channels
prune_parameter_in_channels = FinetunerBox[ops.OPTYPE.PARAMETER].prune_in_channels

prune_multihead_attention_out_channels = FinetunerBox[ops.OPTYPE.MHA].prune_out_channels
prune_multihead_attention_in_channels = FinetunerBox[ops.OPTYPE.MHA].prune_in_channels

prune_lstm_out_channels = FinetunerBox[ops.OPTYPE.LSTM].prune_out_channels
prune_lstm_in_channels = FinetunerBox[ops.OPTYPE.LSTM].prune_in_channels

prune_groupnorm_out_channels = FinetunerBox[ops.OPTYPE.GN].prune_out_channels
prune_groupnorm_in_channels = FinetunerBox[ops.OPTYPE.GN].prune_in_channels

prune_instancenorm_out_channels = FinetunerBox[ops.OPTYPE.IN].prune_out_channels
prune_instancenorm_in_channels = FinetunerBox[ops.OPTYPE.IN].prune_in_channels
