import abc
import torch
import torch.nn as nn

import typing
import torch_pruning as tp
from torch_pruning.pruner import function
from torch_pruning.pruner.importance import GroupTaylorImportance, GroupNormImportance, GroupHessianImportance
from torch_pruning.dependency import Group

import methods.tp_finetuner as fntn_function

__all__=[
    'myGroupNormImportance',
    'myGroupTaylorImportance',
    'myGroupHessianImportance',

    'GroupHessianLabelImportance',
    'GroupTaylorLabelImportance',
    'GroupSecondTaylorLabelImportance',

    'LabelAggregator',
]

class LabelAggregator(abc.ABC):
    @abc.abstractclassmethod
    def __call__(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor: 
        raise NotImplementedError

class TopkAvg(LabelAggregator):
    def __init__(self, k: int):
        self.k = k
        
    def __call__(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        size = x.size()
        if dim> (len(size)-1):
            raise ValueError(f"The target dim {dim} is illegal. The tensor is {len(size)}-d, dim should be at most {len(size)-1}.\n")
        if self.k>size[dim]:
            print(f"The tensor ({size}) has only {size[dim]} elements on the target dim {dim}, modify k to be {size[dim]}.\n")
            values, indices = torch.topk(x, k = size[dim], dim = dim)
        else:
            values, indices = torch.topk(x, k = self.k, dim = dim)
        return values.mean(dim = dim)

class TopkMeanAvg(LabelAggregator):
    def __init__(self, k: int):
        self.k = k
        
    def __call__(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        size = x.size()
        if dim> (len(size)-1):
            raise ValueError(f"The target dim {dim} is illegal. The tensor is {len(size)}-d, dim should be at most {len(size)-1}.\n")
        if self.k>size[dim]:
            print(f"The tensor ({size}) has only {size[dim]} elements on the target dim {dim}, modify k to be {size[dim]}.\n")
            values, indices = torch.topk(x, k = size[dim], dim = dim)
        else:
            values, indices = torch.topk(x, k = self.k, dim = dim)
        mean = torch.mean(x, dim = dim)
        return (values.mean(dim = dim) + mean)/(2)

def GreaterQuantilesAvg(x, dim = 0):
    size = x.size()
    if dim> (len(size)-1):
        raise ValueError(f"The target dim {dim} is illegal. The tensor is {len(size)}-d, dim should be at most {len(size)-1}.\n")
    return torch.quantile(x, q = torch.arange(start=0.5, end = 1.01, step=0.1).to(x.device), dim = dim).mean(0)

def QuantilesAvg(x, dim = 0):
    size = x.size()
    if dim> (len(size)-1):
        raise ValueError(f"The target dim {dim} is illegal. The tensor is {len(size)}-d, dim should be at most {len(size)-1}.\n")
    return torch.quantile(x, q = torch.arange(start=0, end = 1.01, step=0.1).to(x.device), dim = dim).mean(0)

LabelAggregators = {
    'max': torch.max,
    'mean':torch.mean,
    'TopkAvg':TopkAvg,
    'TopkMeanAvg':TopkMeanAvg,
    'GreaterQuantilesAvg': GreaterQuantilesAvg,
    'QuantilesAvg': QuantilesAvg,
}


class myGroupNormImportance(GroupNormImportance):
    @torch.no_grad()
    def __call__(self, group: Group):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                fntn_function.prune_conv_out_channels,
                fntn_function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                fntn_function.prune_conv_in_channels,
                fntn_function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)

                # repeat importance for group convolutions
                if prune_fn == fntn_function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                
                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)
            ####################
            # BatchNorm
            ####################
            elif prune_fn == fntn_function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            ####################
            # LayerNorm
            ####################
            elif prune_fn == fntn_function.prune_layernorm_out_channels:

                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class myGroupWandAImportance(myGroupNormImportance):
    def __init__(self, 
                 p: int=2, 
                 wanda_side = "in", #wanda use the norm of input activated values
                 group_reduction: str="mean", 
                 normalizer: str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]):
        self.p = p
        self.wanda_side = wanda_side
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias

    def register_wanda_hook(self, model, n_sample):
        self.module2wanda = {}

        def hook(module, input, output):
            #p-norm, sum along sample size dim
            if isinstance(input, tuple):
                input = input[0]
                
            if isinstance(output, tuple):
                output = output[0]

            if isinstance(module, nn.Conv2d):
                dim = -3
            else:
                dim = -1

            if module in self.module2wanda:
                if self.wanda_side == "in":
                    self.module2wanda[module] += input.pow(self.p).movedim(dim,0).flatten(1).sum(1)/n_sample 
                elif self.wanda_side == "out":
                    self.module2wanda[module] += output.pow(self.p).movedim(dim,0).flatten(1).sum(1)/n_sample
            else:
                if self.wanda_side == "in":
                    print("Module: {}, input size: {}".format(module, input.size()))
                    self.module2wanda[module] = input.pow(self.p).movedim(dim,0).flatten(1).sum(1)/n_sample 
                elif self.wanda_side == "out":
                    print("Module: {}, output size: {}".format(module, output.size()))
                    self.module2wanda[module] = output.pow(self.p).movedim(dim,0).flatten(1).sum(1)/n_sample

        for name, module in model.named_modules():
            #self.name2module[name] = module
            self.handles = []
            self.handles.append(module.register_forward_hook(hook))

    def remove_hooks(self, ):
        for h in self.handles:
            h.remove()
    def __call__(self, group: Group):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                fntn_function.prune_conv_out_channels,
                fntn_function.prune_linear_out_channels,
            ]:
                if self.wanda_side == "in":
                    wanda = self.module2wanda[layer].unsqueeze(0).unsqueeze(-1) #size = (dout, din, -1 )
                elif self.wanda_side == "out":
                    wanda = self.module2wanda[layer].unsqueeze(1).unsqueeze(-1) #size = (dout, din, -1 )

                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].unsqueeze(-1).flatten(2) #size = (dout, din, -1 )
                else:
                    w = layer.weight.data[idxs].unsqueeze(-1).flatten(2) #size = (dout, din, -1 )

                local_imp = w.abs()*wanda.flatten(1).sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                fntn_function.prune_conv_in_channels,
                fntn_function.prune_linear_in_channels,
            ]:  
                if self.wanda_side == "in":
                    wanda = self.module2wanda[layer].unsqueeze(1).unsqueeze(-1) #size = (din, dout, -1 )
                elif self.wanda_side == "out":
                    wanda = self.module2wanda[layer].unsqueeze(0).unsqueeze(-1) #size = (din, dout, -1 )

                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.unsqueeze(-1).flatten(2) #size = (din, dout, -1 )
                else:
                    w = layer.weight.data.transpose(0, 1).unsqueeze(-1).flatten(2) #size = (din, dout, -1 )

                local_imp = w.abs()*wanda.flatten(1).sum(1)
                # repeat importance for group convolutions
                if prune_fn == fntn_function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                
                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # BatchNorm
            ####################
            elif prune_fn == fntn_function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    if self.wanda_side == "in":
                        wanda = self.module2wanda[layer] #size = (din, )
                    elif self.wanda_side == "out":
                        wanda = self.module2wanda[layer] #size = (dout, )
                    w = layer.weight.data
                    local_imp = w.abs()*wanda
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            ####################
            # LayerNorm
            ####################
            elif prune_fn == fntn_function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if self.wanda_side == "in":
                        wanda = self.module2wanda[layer] #size = (din, )
                    elif self.wanda_side == "out":
                        wanda = self.module2wanda[layer] #size = (dout, )
                    w = layer.weight.data
                    local_imp = w.abs()*wanda
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class myGroupTaylorImportance(GroupTaylorImportance):
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            pruning_dim = dep.target.pruning_dim
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            # Conv/Linear Output
            if prune_fn in [
                fntn_function.prune_conv_out_channels,
                fntn_function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "out_channels"):
                    num_features = layer.out_channels
                elif hasattr(layer, "out_features"):
                    num_features = layer.out_features
                idxs = [idx % num_features for idx in idxs]
                
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    b = layer.bias.data[idxs]
                    db = layer.bias.grad.data[idxs]
                    local_imp = (b * db).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv/Linear Input
            elif prune_fn in [
                fntn_function.prune_conv_in_channels,
                fntn_function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "in_channels"):
                    num_features = layer.in_channels
                elif hasattr(layer, "in_features"):
                    num_features = layer.in_features
                idxs = [idx % num_features for idx in idxs]
                
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                    dw = (layer.weight.grad).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                
                # repeat importance for group convolutions
                if prune_fn == fntn_function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                local_imp = local_imp[idxs]

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # BN
            elif prune_fn == fntn_function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == fntn_function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    num_features = layer.normalized_shape[pruning_dim]
                    idxs = [idx % num_features for idx in idxs]
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class myGroupHessianImportance(GroupHessianImportance):
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []

        if len(self._accu_grad) > 0: # fill gradients so that we can re-use the implementation for Taylor
            for p, g in self._accu_grad.items():
                p.grad.data = g / self._counter[p]
            self.zero_grad()

        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            pruning_dim = dep.target.pruning_dim
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                fntn_function.prune_conv_out_channels,
                fntn_function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "out_channels"):
                    num_features = layer.out_channels
                elif hasattr(layer, "out_features"):
                    num_features = layer.out_features
                idxs = [idx % num_features for idx in idxs]
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                        h = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                        h = layer.weight.grad.data[idxs].flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    h = layer.bias.grad.data[idxs]
                    local_imp = (b**2 * h)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv in_channels
            elif prune_fn in [
                fntn_function.prune_conv_in_channels,
                fntn_function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "in_channels"):
                    num_features = layer.in_channels
                elif hasattr(layer, "in_features"):
                    num_features = layer.in_features
                idxs = [idx % num_features for idx in idxs]
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = (layer.weight).flatten(1)
                        h = (layer.weight.grad).flatten(1)
                    else:
                        w = (layer.weight).transpose(0, 1).flatten(1)
                        h = (layer.weight.grad).transpose(0, 1).flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    if prune_fn == fntn_function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == fntn_function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == fntn_function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    num_features = layer.normalized_shape[pruning_dim]
                    idxs = [idx % num_features for idx in idxs]
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class GroupTaylorLabelImportance(myGroupNormImportance):
    pass

class GroupHessianLabelImportance(myGroupNormImportance):
    pass

class GroupSecondTaylorLabelImportance(myGroupNormImportance):
    pass