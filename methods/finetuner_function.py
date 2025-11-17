import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from copy import deepcopy
from abc import ABC, abstractclassmethod
from typing import Sequence, Tuple
from types import MethodType
from torch_pruning import ops
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm
from typing import Callable, Optional, Union

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.loftq_utils import NFQuantizer


__all__=[
    'BaseFinetuningFunc',
    'FinetunerBox',

    'finetune_conv_out_channels',
    'finetune_conv_in_channels',
    'finetune_depthwise_conv_out_channels',
    'finetune_depthwise_conv_in_channels',
    'finetune_batchnorm_out_channels',
    'finetune_batchnorm_in_channels',
    'finetune_linear_out_channels',
    'finetune_linear_in_channels',
    'finetune_prelu_out_channels',
    'finetune_prelu_in_channels',
    'finetune_layernorm_out_channels',
    'finetune_layernorm_in_channels',
    'finetune_embedding_out_channels',
    'finetune_embedding_in_channels',
    'finetune_parameter_out_channels',
    'finetune_parameter_in_channels',
    'finetune_multihead_attention_out_channels',
    'finetune_multihead_attention_in_channels',
    'finetune_groupnorm_out_channels',
    'finetune_groupnorm_in_channels',
    'finetune_instancenorm_out_channels',
    'finetune_instancenorm_in_channels',

    'finetune_bert_out_channels',
    'finetune_bert_in_channels',
    'finetune_debert_out_channels',
    'finetune_debert_in_channels',
    'finetune_llama_out_channels',
    'finetune_llama_in_channels',
]

class BaseFinetuningFunc(ABC):
    TARGET_MODULES = ops.TORCH_OTHERS  # None

    def __init__(self, finetuning_dim: int = 1, dtype: str = "fp32"):
        self.finetuning_dim = finetuning_dim
        self.dtype = dtype

    @abstractclassmethod
    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def finetune_in_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError
    
    def get_out_channels(self, layer):
        return 1

    def get_in_channels(self, layer):
        return 1

    def _finetune_parameter(self, param, finetune_idxs, finetuning_dim, dtype: str = "fp32"):
        tensor_size = list(param.data.size())
        tensor_size[finetuning_dim] = len(finetune_idxs)
        if dtype == "fp16":
            finetuned_weight = nn.Parameter(torch.zeros(tensor_size, dtype = torch.float16, device = param.device))
        elif dtype == "bf16":
            finetuned_weight = nn.Parameter(torch.zeros(tensor_size, dtype = torch.bfloat16, device = param.device))
        elif dtype == "fp32":
            finetuned_weight = nn.Parameter(torch.zeros(tensor_size, device = param.device))
        init.kaiming_uniform_(finetuned_weight, a=math.sqrt(5))
        return finetuned_weight

    def _create_size_mapping(self, finetune_idxs, out_dim, device, dtype: str = "fp32"):
        if dtype == "fp16":
            size_mapping = torch.zeros((len(finetune_idxs), out_dim), dtype = torch.float16, device = device)
        elif dtype == "bf16":
            size_mapping = torch.zeros((len(finetune_idxs), out_dim), dtype = torch.bfloat16, device = device)
        elif dtype == "fp32":
            size_mapping = torch.zeros((len(finetune_idxs), out_dim), device = device)

        #print(finetune_idxs)
        for idx in zip(range(len(finetune_idxs)), finetune_idxs):
            size_mapping[idx] = 1.
        return size_mapping

class ConvFinetuner(BaseFinetuningFunc):
    TARGET_MODULE = ops.TORCH_CONV

    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.out_channels)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_channels, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_channels, layer.weight.device, dtype = "bf16")

        if not layer.transposed:
            finetuning_dim = 0
        else:
            finetuning_dim = 1
        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim, dtype = "bf16"))
        layer.register_buffer('finetuned_out_mapping', size_mapping)

        if layer.bias is not None:
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = "bf16"))
        
        
        def forward(self, input):
            hidden_output = self._conv_forward(input, self.weight, self.bias)
            output_dtype = hidden_output.dtype
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                input = input.to(self.finetuned_in_weight.dtype)
                hidden_output = hidden_output + self._conv_forward(input, finetuned_in_weight, None).to(output_dtype)

            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                if self.bias is not None and hasattr(self, 'finetuned_out_bias'):
                    finetuned_out_bias = self.finetuned_out_bias.movedim(0,-1).matmul(self.finetuned_out_mapping).movedim(-1, 0)
                else:
                    finetuned_out_bias = None
                input = input.to(self.finetuned_out_weight.dtype)
                finetuned_out_weight = self.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_out_mapping).movedim(-1, finetuning_dim)
                hidden_output = hidden_output + self._conv_forward(input, finetuned_out_weight, finetuned_out_bias).to(output_dtype)

            return hidden_output
        
        layer.forward = MethodType(forward, layer)
        return layer

    def finetune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.in_channels)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if layer.groups>1:
            finetune_idxs = finetune_idxs[:len(finetune_idxs)//layer.groups]

        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.in_channels//layer.groups, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.in_channels//layer.groups, layer.weight.device, dtype = "bf16")
        
        if not layer.transposed:
            finetuning_dim = 1
        else:
            finetuning_dim = 0
        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim, dtype = "bf16"))
        
        layer.register_buffer('finetuned_in_mapping', size_mapping)
        # no bias because it does not change the output channels

        
        def forward(self, input):
            hidden_output = self._conv_forward(input, self.weight, self.bias)
            output_dtype = hidden_output.dtype
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                input = input.to(self.finetuned_in_weight.dtype)
                hidden_output = hidden_output + self._conv_forward(input, finetuned_in_weight, None).to(output_dtype)

            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                if self.bias is not None and hasattr(self, 'finetuned_out_bias'):
                    finetuned_out_bias = self.finetuned_out_bias
                else:
                    finetuned_out_bias = None
                input = input.to(self.finetuned_out_weight.dtype)
                finetuned_out_weight = self.finetuned_out_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_out_mapping).movedim(-1, finetuning_dim)
                hidden_output = hidden_output + self._conv_forward(input, finetuned_out_weight, finetuned_out_bias).to(output_dtype)

            return hidden_output
        
        layer.forward = MethodType(forward, layer)
        return layer
        

    def get_out_channels(self, layer):
        return layer.out_channels

    def get_in_channels(self, layer):
        return layer.in_channels

class DepthwiseConvFinetuner(ConvFinetuner):
    TARGET_MODULE = ops.TORCH_CONV

    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.out_channels)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_channels, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_channels, layer.weight.device, dtype = "bf16")

        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = "bf16"))
        
        layer.register_buffer('finetuned_mapping', size_mapping)

        if layer.bias is not None:
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = "bf16"))
        
        def forward(self, input):
            hidden_output = self._conv_forward(input, self.weight, self.bias)
            output_dtype = hidden_output.dtype

            if hasattr(self, 'finetuned_weight'):
                finetuning_dim = 0
                if self.bias is not None and hasattr(self, 'finetuned_bias'):
                    #finetuned_bias = self.finetuned_bias.movedim(0,-1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    finetuned_bias = self.finetuned_bias
                else:
                    finetuned_bias = None
                input = input.to(self.finetuned_weight.dtype)
                finetuned_weight = self.finetuned_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_mapping).movedim(-1, finetuning_dim)
                hidden_output = hidden_output + self._conv_forward(input, finetuned_weight, finetuned_bias).to(output_dtype)

            return hidden_output
        
        layer.forward = MethodType(forward, layer)
        return layer

    finetune_in_channels = finetune_out_channels

class LinearFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR

    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:        
        finetune_idxs = list(set(range(layer.out_features)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_features, layer.weight.device, dtype = "bf16")
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = "bf16"))
        layer.register_buffer('finetuned_out_mapping', size_mapping)

        if layer.bias is not None:
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = "bf16"))
        
        def forward(self, input):
            hidden_output = F.linear(input, self.weight, self.bias)
            output_dtype = hidden_output.dtype
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                input = input.to(self.finetuned_in_weight.dtype)
                hidden_output = hidden_output + F.linear(input, finetuned_in_weight, None).to(output_dtype)

            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                if self.bias is not None and hasattr(self, 'finetuned_out_bias'):
                    finetuned_out_bias = self.finetuned_out_bias
                else:
                    finetuned_out_bias = None
                input = input.to(self.finetuned_out_weight.dtype)
                hidden_output = hidden_output + \
                    F.linear(F.linear(input, self.finetuned_out_weight, finetuned_out_bias), self.finetuned_out_mapping.transpose(0,1),None ).to(output_dtype)

            return hidden_output
        
        layer.forward = MethodType(forward, layer)
        return layer

    def finetune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.in_features)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.in_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.in_features, layer.weight.device, dtype = "bf16")

        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.weight, finetune_idxs, 1, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.weight, finetune_idxs, 1, dtype = "bf16"))
        layer.register_buffer('finetuned_in_mapping', size_mapping)

        def forward(self, input):
            hidden_output = F.linear(input, self.weight, self.bias)
            output_dtype = hidden_output.dtype
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim)
                input = input.to(self.finetuned_in_weight.dtype)
                hidden_output = hidden_output + F.linear(input, finetuned_in_weight, None).to(output_dtype)

            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                if self.bias is not None and hasattr(self, 'finetuned_out_bias'):
                    finetuned_out_bias = self.finetuned_out_bias
                else:
                    finetuned_out_bias = None
                input = input.to(self.finetuned_out_weight.dtype)
                hidden_output = hidden_output + \
                    F.linear(F.linear(input, self.finetuned_out_weight, finetuned_out_bias), self.finetuned_out_mapping.transpose(0,1),None ).to(output_dtype)

            return hidden_output
        
        layer.forward = MethodType(forward, layer)
        return layer
    
    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features

class myLinear(nn.Module):
    def __init__(self, basic_layer, p_dropout: float = 0.1, variant: str = 'PruFT'):
        super(myLinear, self).__init__()
        self.basic_layer = basic_layer
        self.out_features = basic_layer.out_features
        self.in_features = basic_layer.in_features
        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0.0 else None
        self.variant = variant
        if self.variant == 'dora':
            #Combine the idea of dora with our approach, eventually not use in this work
            self.magnitude = nn.Parameter(basic_layer.weight.norm(dim = 0, keepdim = True))
        self.merge = False

    def merge_and_unload(self,):
        out_mapping = hasattr(self, 'finetuned_out_weight') and hasattr(self, 'finetuned_out_mapping')
        in_mapping = hasattr(self, 'finetuned_in_weight') and hasattr(self, 'finetuned_in_mapping')
        if in_mapping:
            self.basic_layer.weight.data += self.finetuned_in_weight.movedim(1, -1).matmul(self.finetuned_in_mapping).movedim(-1, 1).to(self.basic_layer.weight.dtype)
        
        if out_mapping:
            self.basic_layer.weight.data += self.finetuned_out_weight.movedim(0, -1).matmul(self.finetuned_out_mapping).movedim(-1, 0).to(self.basic_layer.weight.dtype)
        
            if self.basic_layer.bias is not None and hasattr(self, 'finetuned_out_bias'):
                self.basic_layer.bias.data += self.finetuned_out_bias.matmul(self.finetuned_out_mapping)
        self.merge = True
        return self.basic_layer

    def PruFT_forward(self, input):
        h = self.basic_layer(input)
        if self.dropout is not None:
            input = self.dropout(input)
        output_dtype = h.dtype
        out_mapping = hasattr(self, 'finetuned_out_weight') and hasattr(self, 'finetuned_out_mapping')
        in_mapping = hasattr(self, 'finetuned_in_weight') and hasattr(self, 'finetuned_in_mapping')
        if in_mapping:
            input = input.to(self.finetuned_in_mapping.dtype)
            h = h + F.linear(input, 
                             self.finetuned_in_weight.movedim(1, -1).matmul(self.finetuned_in_mapping).movedim(-1, 1), 
                             None,
                             ).to(output_dtype)
        
        if out_mapping:
            input = input.to(self.finetuned_out_weight.dtype)
            if self.basic_layer.bias is not None and hasattr(self, 'finetuned_out_bias'):
                finetuned_out_bias = self.finetuned_out_bias
            else:
                finetuned_out_bias = None
            h = h + F.linear(F.linear(input, self.finetuned_out_weight, finetuned_out_bias), 
                             self.finetuned_out_mapping.transpose(0,1), 
                             None,
                             ).to(output_dtype)
        return h
    
    def DPruFT_forward(self, input):
        weight = self.basic_layer.weight
        bias = self.basic_layer.bias
        if self.dropout is not None:
            input = self.dropout(input)
        output_dtype = input.dtype
        out_mapping = hasattr(self, 'finetuned_out_weight') and hasattr(self, 'finetuned_out_mapping')
        in_mapping = hasattr(self, 'finetuned_in_weight') and hasattr(self, 'finetuned_in_mapping')
        if in_mapping:
            weight = weight.to(self.finetuned_in_mapping.dtype)
            weight = weight + self.finetuned_in_weight.movedim(1, -1).matmul(self.finetuned_in_mapping).movedim(-1, 1)
        
        if out_mapping:
            weight = weight.to(self.finetuned_out_weight.dtype)
            weight = weight + self.finetuned_out_weight.movedim(0, -1).matmul(self.finetuned_out_mapping).movedim(-1, 0)
            if self.basic_layer.bias is not None and hasattr(self, 'finetuned_out_bias'):
                bias = bias.to(self.finetuned_out_bias.dtype)
                bias = bias + self.finetuned_out_bias.matmul(self.finetuned_out_mapping)
                
        weight = weight/(weight.norm(dim = 0, keepdim = True))
        h = (F.linear(input.to(weight.dtype), weight, bias) * self.magnitude).to(output_dtype)
        return h
    
    def forward(self, input):
        if self.merge:
            return self.basic_layer(input)
        if self.variant == 'PruFT':
            return self.PruFT_forward(input)
        elif self.variant == 'dora':
            return self.DPruFT_forward(input)

class LinearFinetuner_dropout(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR
    def __init__(self, finetuning_dim: int = 1, dtype: str = "fp32", p_dropout: float = 0.0, variant: str = 'PruFT'):
        self.finetuning_dim = finetuning_dim
        self.dtype = dtype
        self.p_dropout = p_dropout
        self.variant = variant

    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:        
        finetune_idxs = list(set(range(layer.out_features)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.out_features, layer.weight.device, dtype = "bf16")
        
        if not isinstance(layer, myLinear):
            layer = myLinear(layer, p_dropout = self.p_dropout, variant = self.variant)
        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.basic_layer.weight, finetune_idxs, 0, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_out_weight', self._finetune_parameter(layer.basic_layer.weight, finetune_idxs, 0, dtype = "bf16"))
        layer.register_buffer('finetuned_out_mapping', size_mapping)

        if layer.basic_layer.bias is not None:
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.basic_layer.bias, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_out_bias', self._finetune_parameter(layer.basic_layer.bias, finetune_idxs, 0, dtype = "bf16"))
        
        return layer

    def finetune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.in_features)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.in_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.in_features, layer.weight.device, dtype = "bf16")
        
        if not isinstance(layer, myLinear):
            layer = myLinear(layer, p_dropout = self.p_dropout, variant = self.variant)
        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.basic_layer.weight, finetune_idxs, 1, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_in_weight', self._finetune_parameter(layer.basic_layer.weight, finetune_idxs, 1, dtype = "bf16"))
        
        layer.register_buffer('finetuned_in_mapping', size_mapping)

        return layer
    
    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features

class BatchnormFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_BATCHNORM

    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.num_features)) - set(idxs))
        finetune_idxs.sort()
        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_features, layer.weight.device, dtype = "bf16")

        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = self.dtype))
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = "bf16"))
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = "bf16"))
            
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
                bias = self.bias
                output = F.batch_norm(
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
                output_dtype = output.dtype
                if hasattr(self, 'finetuned_weight'):
                    finetuning_dim = 0
                    finetuned_weight = self.finetuned_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_mapping).movedim(-1, finetuning_dim)

                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                return output + F.batch_norm(
                    input.to(finetuned_weight.dtype),
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean.to(finetuned_weight.dtype)
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var.to(finetuned_weight.dtype) if not self.training or self.track_running_stats else None,
                    finetuned_weight,
                    finetuned_bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                ).to(output_dtype)

            layer.forward = MethodType(forward, layer)
        return layer

    finetune_in_channels = finetune_out_channels

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

    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetuning_dim = self.finetuning_dim

        if layer.elementwise_affine:
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False
        
        num_features = layer.normalized_shape[finetuning_dim]

        finetune_idxs = list(set(range(num_features)) - set(idxs))
        finetune_idxs.sort()
        if len(finetune_idxs)==0 or len(layer.normalized_shape) < -finetuning_dim:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, num_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, num_features, layer.weight.device, dtype = "bf16")
        
        if layer.elementwise_affine:
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, finetuning_dim, dtype = "bf16"))
                
            layer.register_buffer('finetuned_mapping', size_mapping)
            layer.register_buffer('finetuning_dim', torch.tensor(finetuning_dim))
            if layer.bias is not None:
                if self.dtype in ["fp16", "bf16", "fp32"]:
                    layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, finetuning_dim, dtype = self.dtype))
                elif self.dtype in ["fp8", "fp4", "fp2"]:
                    layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, finetuning_dim, dtype = "bf16"))
            
            def forward(self, input):
                weight = self.weight
                bias = self.bias
                output = F.layer_norm(
                    input, self.normalized_shape, weight, bias, self.eps)
                output_dtype = output.dtype
                if hasattr(self, 'finetuned_weight'):
                    finetuning_dim = self.finetuning_dim.item()
                    finetuned_weight = self.finetuned_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_mapping).movedim(-1, finetuning_dim)

                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)  
                return output + F.layer_norm(
                    input.to(finetuned_weight.dtype), self.normalized_shape, finetuned_weight, finetuned_bias, self.eps).to(output_dtype)

            layer.forward = MethodType(forward, layer)
        return layer

    finetune_in_channels = finetune_out_channels

    def get_out_channels(self, layer):
        return layer.normalized_shape[self.finetuning_dim]

    def get_in_channels(self, layer):
        return layer.normalized_shape[self.finetuning_dim]

class GroupNormFinetuner(BaseFinetuningFunc):
    def finetune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        finetune_idxs = list(set(range(layer.num_channels)) - set(idxs))
        finetune_idxs.sort()
        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_channels, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_channels, layer.weight.device, dtype = "bf16")
        
        if layer.affine:
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = "bf16"))
            
            layer.register_buffer('finetuned_mapping', size_mapping)
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = "bf16"))
            
            def forward(self, input):
                weight = self.weight
                bias = self.bias
                output = F.group_norm(
                    input, self.num_groups, weight, bias, self.eps)
                output_dtype = output.dtype

                if hasattr(self, 'finetuned_weight'):
                    finetuned_weight = self.finetuned_weight.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    
                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    
                #************** have not be tested yet **************
                return output + F.group_norm(
                    input.to(finetuned_weight.dtype), self.num_groups, finetuned_weight, finetuned_bias, self.eps).to(output_dtype)
                #************** have not be tested yet **************

            layer.forward = MethodType(forward, layer)
        return layer
    
    finetune_in_channels = finetune_out_channels

    def get_out_channels(self, layer):
        return layer.num_channels

    def get_in_channels(self, layer):
        return layer.num_channels

class InstanceNormFinetuner(BaseFinetuningFunc):
    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        finetune_idxs = list(set(range(layer.num_features)) - set(idxs))
        finetune_idxs.sort()
        if layer.affine:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_features, layer.weight.device, dtype = "bf16")
        
        if layer.affine:
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = "bf16"))
                    
            layer.register_buffer('finetuned_mapping', size_mapping)
            if self.dtype in ["fp16", "bf16", "fp32"]:
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = self.dtype))
            elif self.dtype in ["fp8", "fp4", "fp2"]:
                layer.register_parameter('finetuned_bias', self._finetune_parameter(layer.bias, finetune_idxs, 0, dtype = "bf16"))
            
            def _apply_instance_norm(self, input):
                weight = self.weight
                bias = self.bias
                output = F.instance_norm(
                    input, self.running_mean, self.running_var, weight, bias,
                    self.training or not self.track_running_stats, self.momentum, self.eps)
                output_dtype = output.dtype
                if hasattr(self, 'finetuned_weight'):
                    finetuned_weight = self.finetuned_weight.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    
                if bias is not None and hasattr(self, 'finetuned_bias'):
                    finetuned_bias = self.finetuned_bias.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                    
                #************** have not be tested yet **************
                return output + F.instance_norm(
                    input.to(finetuned_weight.dtype), 
                    self.running_mean.to(finetuned_weight.dtype),
                    self.running_var.to(finetuned_weight.dtype), 
                    finetuned_weight, finetuned_bias,
                    self.training or not self.track_running_stats, self.momentum, self.eps).to(output_dtype)
                #************** have not be tested yet **************

            layer._apply_instance_norm = MethodType(_apply_instance_norm, layer)
        return layer

    finetune_in_channels = finetune_out_channels

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features

class PReLUFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_PRELU

    def finetune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        if layer.num_parameters == 1:
            layer.weight.requires_grad = True
            return layer
        
        finetune_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_parameters, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.num_parameters, layer.weight.device, dtype = "bf16")
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0, dtype = "bf16"))
            
        layer.register_buffer('finetuned_mapping', size_mapping)

        def forward(self, input):
            weight = self.weight
            if hasattr(self, 'finetuned_weight'):
                finetuned_weight = self.finetuned_weight.movedim(0, -1).matmul(self.finetuned_mapping).movedim(-1, 0)
                weight = weight + finetuned_weight
            return F.prelu(input, weight)
        layer.forward = MethodType(forward, layer)
        return layer

    finetune_in_channels = finetune_out_channels

    def get_out_channels(self, layer):
        if layer.num_parameters == 1:
            return None
        else:
            return layer.num_parameters

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class EmbeddingFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_EMBED

    def finetune_out_channels(self, layer: nn.Embedding, idxs: list) -> nn.Module:
        finetune_idxs = list(set(range(layer.embedding_dim)) - set(idxs))
        finetune_idxs.sort()
        layer.weight.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.embedding_dim, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.embedding_dim, layer.weight.device, dtype = "bf16")
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 1, dtype = self.dtype))
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 1, dtype = "bf16"))
        
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

    finetune_in_channels = finetune_out_channels

    def get_out_channels(self, layer):
        return layer.embedding_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class ParameterFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_PARAMETER
    def __init__(self, finetuning_dim=-1):
        super().__init__(finetuning_dim=finetuning_dim)
    
    #TODO: try to deal with this in the parameters' belonging module
    def finetune_out_channels(self, tensor, idxs: list) -> nn.Module:
        return tensor

    def finetune_in_channels(self, tensor, idxs: list) -> nn.Module:
        return tensor
    
    def get_out_channels(self, parameter):
        return parameter.shape[self.finetuning_dim]

    def get_in_channels(self, parameter):
        return parameter.shape[self.finetuning_dim]

class MultiheadAttentionFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = ops.TORCH_MHA

    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        
    def finetune_out_channels(self, layer, idxs: list) -> nn.Module:

        finetune_idxs = list(set(range(layer.embed_dim)) - set(idxs))
        finetune_idxs.sort()
        for param in layer.parameters():
            param.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        device = layer.q_proj_weight.device
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.embed_dim, device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, layer.embed_dim, device, dtype = "bf16")

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

        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, 3*layer.embed_dim, device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, 3*layer.embed_dim, device, dtype = "bf16")

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
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, linear.out_features, device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, linear.out_features, device, dtype = "bf16")
       
        linear.register_buffer('finetuned_out_mapping', size_mapping)
        linear.register_parameter('finetuned_out_weight', self._finetune_parameter(linear.weight, finetune_idxs, 0))

        if linear.bias is not None:
            linear.register_parameter('finetuned_out_bias', self._finetune_parameter(linear.bias, finetune_idxs, 0))

        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, linear.in_features, device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, linear.in_features, device, dtype = "bf16")
        
        linear.register_buffer('finetuned_in_mapping', size_mapping)
        linear.register_parameter('finetuned_in_weight', self._finetune_parameter(linear.weight, finetune_idxs, 1))

        def forward(self, input):
            hidden_output = F.linear(input, self.weight, self.bias)
            output_dtype = hidden_output.dtype
            if hasattr(self, 'finetuned_in_weight'):
                finetuning_dim = 1
                finetuned_in_weight = self.finetuned_in_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_in_mapping).movedim(-1, finetuning_dim).to(output_dtype)
                hidden_output = hidden_output + F.linear(input, finetuned_in_weight, None)

            if hasattr(self, 'finetuned_out_weight'):
                finetuning_dim = 0
                if self.bias is not None and hasattr(self, 'finetuned_out_bias'):
                    finetuned_out_bias = self.finetuned_out_bias
                else:
                    finetuned_out_bias = None
                input = input.to(self.finetuned_out_weight.dtype)
                hidden_output = hidden_output + \
                    F.linear(F.linear(input, self.finetuned_out_weight, finetuned_out_bias), self.finetuned_out_mapping.transpose(0,1),None ).to(output_dtype)

            return hidden_output
        
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

    finetune_in_channels = finetune_out_channels

    def get_out_channels(self, layer):
        return layer.embed_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer)

FinetunerBox = {
    ops.OPTYPE.CONV: ConvFinetuner(),
    #ops.OPTYPE.LINEAR: LinearFinetuner(),
    ops.OPTYPE.LINEAR: LinearFinetuner_dropout(),
    ops.OPTYPE.BN: BatchnormFinetuner(),
    ops.OPTYPE.DEPTHWISE_CONV: DepthwiseConvFinetuner(),
    ops.OPTYPE.PRELU: PReLUFinetuner(),
    ops.OPTYPE.LN: LayernormFinetuner(),
    ops.OPTYPE.EMBED: EmbeddingFinetuner(),
    ops.OPTYPE.PARAMETER: ParameterFinetuner(),
    ops.OPTYPE.MHA: MultiheadAttentionFinetuner(),
    ops.OPTYPE.GN: GroupNormFinetuner(),
    ops.OPTYPE.IN: InstanceNormFinetuner(),
}

# Alias
finetune_conv_out_channels = FinetunerBox[ops.OPTYPE.CONV].finetune_out_channels
finetune_conv_in_channels = FinetunerBox[ops.OPTYPE.CONV].finetune_in_channels

finetune_depthwise_conv_out_channels = FinetunerBox[ops.OPTYPE.DEPTHWISE_CONV].finetune_out_channels
finetune_depthwise_conv_in_channels = FinetunerBox[ops.OPTYPE.DEPTHWISE_CONV].finetune_in_channels

finetune_batchnorm_out_channels = FinetunerBox[ops.OPTYPE.BN].finetune_out_channels
finetune_batchnorm_in_channels = FinetunerBox[ops.OPTYPE.BN].finetune_in_channels

finetune_linear_out_channels = FinetunerBox[ops.OPTYPE.LINEAR].finetune_out_channels
finetune_linear_in_channels = FinetunerBox[ops.OPTYPE.LINEAR].finetune_in_channels

finetune_prelu_out_channels = FinetunerBox[ops.OPTYPE.PRELU].finetune_out_channels
finetune_prelu_in_channels = FinetunerBox[ops.OPTYPE.PRELU].finetune_in_channels

finetune_layernorm_out_channels = FinetunerBox[ops.OPTYPE.LN].finetune_out_channels
finetune_layernorm_in_channels = FinetunerBox[ops.OPTYPE.LN].finetune_in_channels

finetune_embedding_out_channels = FinetunerBox[ops.OPTYPE.EMBED].finetune_out_channels
finetune_embedding_in_channels = FinetunerBox[ops.OPTYPE.EMBED].finetune_in_channels

finetune_parameter_out_channels = FinetunerBox[ops.OPTYPE.PARAMETER].finetune_out_channels
finetune_parameter_in_channels = FinetunerBox[ops.OPTYPE.PARAMETER].finetune_in_channels

finetune_multihead_attention_out_channels = FinetunerBox[ops.OPTYPE.MHA].finetune_out_channels
finetune_multihead_attention_in_channels = FinetunerBox[ops.OPTYPE.MHA].finetune_in_channels

finetune_groupnorm_out_channels = FinetunerBox[ops.OPTYPE.GN].finetune_out_channels
finetune_groupnorm_in_channels = FinetunerBox[ops.OPTYPE.GN].finetune_in_channels

finetune_instancenorm_out_channels = FinetunerBox[ops.OPTYPE.IN].finetune_out_channels
finetune_instancenorm_in_channels = FinetunerBox[ops.OPTYPE.IN].finetune_in_channels

class BertAttnFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = BertSelfAttention
    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:        
        layer.query = finetune_linear_out_channels(layer.query, idxs)
        layer.key = finetune_linear_out_channels(layer.key, idxs)
        layer.value = finetune_linear_out_channels(layer.value, idxs)
        return layer

    def finetune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        layer.query = finetune_linear_in_channels(layer.query, idxs)
        layer.key = finetune_linear_in_channels(layer.key, idxs)
        layer.value = finetune_linear_in_channels(layer.value, idxs)
        return layer
    
    def get_out_channels(self, layer):
        return layer.query.out_features

    def get_in_channels(self, layer):
        return layer.query.in_features
    
class DisentangledAttnFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = DisentangledSelfAttention
    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:        
        layer.query_proj = finetune_linear_out_channels(layer.query_proj, idxs)
        layer.key_proj = finetune_linear_out_channels(layer.key_proj, idxs)
        layer.value_proj = finetune_linear_out_channels(layer.value_proj, idxs)
        return layer

    def finetune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        layer.query_proj = finetune_linear_in_channels(layer.query_proj, idxs)
        layer.key_proj = finetune_linear_in_channels(layer.key_proj, idxs)
        layer.value_proj = finetune_linear_in_channels(layer.value_proj, idxs)
        return layer
    
    def get_out_channels(self, layer):
        return layer.query_proj.out_features

    def get_in_channels(self, layer):
        return layer.query_proj.in_features

    
class LlamaRMSNormFinetuner(BaseFinetuningFunc):
    TARGET_MODULES = LlamaRMSNorm
    def __init__(self, metrcis=None, finetuning_dim=-1):
        super().__init__(metrcis)
        self.finetuning_dim = finetuning_dim
        
    def finetune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:   
        finetune_idxs = list(set(range(layer.out_features)) - set(idxs))
        finetune_idxs.sort()     
        layer.weight.requires_grad = False
        if len(finetune_idxs)==0:
            return layer
        finetuning_dim = self.finetuning_dim
        num_features = layer.weight.size()[finetuning_dim]
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            size_mapping = self._create_size_mapping(finetune_idxs, num_features, layer.weight.device, dtype = self.dtype)
        elif self.dtype in ["fp8", "fp4", "fp2"]:
            size_mapping = self._create_size_mapping(finetune_idxs, num_features, layer.weight.device, dtype = "bf16")
        

        layer.register_parameter('finetuned_weight', self._finetune_parameter(layer.weight, finetune_idxs, 0))
        layer.register_buffer('finetuned_mapping', size_mapping)

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = hidden_states.to(input_dtype)
            finetuned_weight = self.finetuned_weight.movedim(finetuning_dim, -1).matmul(self.finetuned_mapping).movedim(-1, finetuning_dim).to(input_dtype)
            return self.weight * hidden_states + finetuned_weight * hidden_states

        layer.forward = MethodType(forward, layer)
        return layer

    finetune_in_channels = finetune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.size()[self.finetuning_dim]

    def get_in_channels(self, layer):
        return layer.weight.size()[self.finetuning_dim]

myFinetuner = {
    BertSelfAttention: BertAttnFinetuner(),
    DisentangledSelfAttention: DisentangledAttnFinetuner(),
    LlamaRMSNorm: LlamaRMSNormFinetuner(),
}

finetune_bert_out_channels = myFinetuner[BertSelfAttention].finetune_out_channels
finetune_bert_in_channels = myFinetuner[BertSelfAttention].finetune_in_channels
finetune_debert_out_channels = myFinetuner[DisentangledSelfAttention].finetune_out_channels
finetune_debert_in_channels = myFinetuner[DisentangledSelfAttention].finetune_in_channels
finetune_llamanorm_out_channels = myFinetuner[LlamaRMSNorm].finetune_out_channels
finetune_llamanorm_in_channels = myFinetuner[LlamaRMSNorm].finetune_in_channels