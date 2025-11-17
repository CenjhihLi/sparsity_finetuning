import abc
import torch
import torch.nn as nn

import typing
import torch_pruning as tp
from torch_pruning.pruner import function
from torch_pruning.pruner.importance import GroupNormImportance, GroupTaylorImportance, GroupHessianImportance
from torch_pruning.dependency import Group
import methods.tp_finetuner as fntn_function

__all__=[
    'GroupHessianLabelImportance',
    'GroupTaylorLabelImportance',

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

class GroupTaylorLabelImportance(GroupNormImportance):
    """ Grouped first-order taylor expansion of the loss function.
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf

        Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                loss = loss_fn(model(inputs), labels)
                loss.backward() # compute gradients
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = GroupTaylorImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 n_label = 200,
                 label_aggregator = torch.max,
                 multivariable:bool=False, 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias
        self.n_label = n_label
        self.label_aggregator = label_aggregator

    def accumulate_grad(self, model, label, ignored_layers = None, n_data = 500):
        ignored_layers = ignored_layers if ignored_layers is not None else []
        for m in model.modules():
            if m not in ignored_layers:
                if hasattr(m, 'weight'):
                    if m.weight.grad is not None:
                        m.weight_grad_labelwise[..., label] += m.weight.grad.data.clone()/n_data
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        if m.bias.grad is not None:
                            m.bias_grad_labelwise[..., label] += m.bias.grad.data.clone()/n_data

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
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1).unsqueeze(0)
                    dw = layer.weight_grad_labelwise.movedim(-1,0).data.transpose(2, 1)[:,idxs].flatten(2)
                else:
                    w = layer.weight.data[idxs].flatten(1).unsqueeze(0)
                    dw = layer.weight_grad_labelwise.movedim(-1,0).data[:,idxs].flatten(2)
                if self.multivariable:
                    local_imp = self.label_aggregator((w * dw).sum(2).abs(), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
                else:
                    local_imp = self.label_aggregator((w * dw).abs().sum(2), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
                    
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    b = layer.bias.data[idxs].unsqueeze(0)
                    db = layer.bias_grad_labelwise.movedim(-1,0).data[:,idxs]
                    local_imp = self.label_aggregator((b * db).abs(), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
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
                    w = layer.weight.flatten(1).unsqueeze(0)
                    dw = layer.weight_grad_labelwise.movedim(-1,0).flatten(2)
                else:
                    w = layer.weight.transpose(0, 1).flatten(1).unsqueeze(0)
                    dw = layer.weight_grad_labelwise.movedim(-1,0).transpose(1, 2).flatten(2)
                if self.multivariable:
                    local_imp = self.label_aggregator((w * dw).sum(2).abs(), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
                else:
                    local_imp = self.label_aggregator((w * dw).abs().sum(2), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
                    
                
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
                    w = layer.weight.data[idxs].unsqueeze(0)
                    dw = layer.weight_grad_labelwise.movedim(-1,0).data[:,idxs]
                    local_imp = self.label_aggregator((w*dw).abs(), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs].unsqueeze(0)
                        db = layer.bias_grad_labelwise.movedim(-1,0).data[:,idxs]
                        local_imp = self.label_aggregator((b * db).abs(), dim=0)
                        if self.label_aggregator in [torch.max]:
                            local_imp, _ = local_imp
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == fntn_function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    num_features = layer.normalized_shape[pruning_dim]
                    idxs = [idx % num_features for idx in idxs]
                    w = layer.weight.data[idxs].unsqueeze(0)
                    dw = layer.weight_grad_labelwise.movedim(-1,0).data[:,idxs]
                    local_imp = self.label_aggregator((w*dw).abs(), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs].unsqueeze(0)
                        db = layer.bias_grad_labelwise.movedim(-1,0).data[:,idxs]
                        local_imp = self.label_aggregator((b * db).abs(), dim=0)
                        if self.label_aggregator in [torch.max]:
                            local_imp, _ = local_imp
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
        
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class GroupHessianLabelImportance(GroupNormImportance):
    """Grouped Optimal Brain Damage:
       https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html

       Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                scorer = GroupHessianImportance()   
                scorer.zero_grad() # clean the acuumulated gradients if necessary
                loss = loss_fn(model(inputs), labels, reduction='none') # compute loss for each sample
                for l in loss:
                    model.zero_grad() # clean the model gradients
                    l.backward(retain_graph=True) # compute gradients for each sample
                    scorer.accumulate_grad(model) # accumulate gradients of each sample
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 n_label = 200,
                 label_aggregator = torch.max,
                 #DEVICE = 'cuda',
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self.n_label = n_label
        self.label_aggregator = label_aggregator
    
    def accumulate_grad(self, model, label, ignored_layers = None, n_data = 500):
        ignored_layers = ignored_layers if ignored_layers is not None else []
        for m in model.modules():
            if m not in ignored_layers:
                if hasattr(m, 'weight'):
                    if m.weight.grad is not None:
                        m.weight_grad_labelwise[..., label] += m.weight.grad.data.clone().pow(2)/n_data
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        if m.bias.grad is not None:
                            m.bias_grad_labelwise[..., label] += m.bias.grad.data.clone().pow(2)/n_data
    
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
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1).unsqueeze(0)
                        h = layer.weight_grad_labelwise.movedim(-1,0).data.transpose(2, 1)[:,idxs].flatten(2)
                    else:
                        w = layer.weight.data[idxs].flatten(1).unsqueeze(0)
                        h = layer.weight_grad_labelwise.movedim(-1,0).data[:,idxs].flatten(2)

                    local_imp = self.label_aggregator((w**2 * h).sum(2), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp

                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs].unsqueeze(0)
                    h = layer.bias_grad_labelwise.movedim(-1,0).data[:,idxs]
                    local_imp = self.label_aggregator((b**2 * h), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
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
                        w = layer.weight.flatten(1).unsqueeze(0)
                        h = layer.weight_grad_labelwise.movedim(-1,0).flatten(2)
                    else:
                        w = layer.weight.transpose(0, 1).flatten(1).unsqueeze(0)
                        h = layer.weight_grad_labelwise.movedim(-1,0).transpose(1, 2).flatten(2)

                    local_imp = self.label_aggregator((w**2 * h).sum(2), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp

                    if prune_fn == fntn_function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == fntn_function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs].unsqueeze(0)
                        h = layer.weight_grad_labelwise.movedim(-1,0).data[:,idxs]
                        local_imp = self.label_aggregator((w**2 * h), dim=0)
                        if self.label_aggregator in [torch.max]:
                            local_imp, _ = local_imp
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is None:
                        b = layer.bias.data[idxs].unsqueeze(0)
                        h = layer.bias_grad_labelwise.movedim(-1,0).data[:,idxs]
                        local_imp = self.label_aggregator((b**2 * h).abs(), dim=0)
                        if self.label_aggregator in [torch.max]:
                            local_imp, _ = local_imp
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == fntn_function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    num_features = layer.normalized_shape[pruning_dim]
                    idxs = [idx % num_features for idx in idxs]
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs].unsqueeze(0)
                        h = layer.weight_grad_labelwise.movedim(-1,0).data[:,idxs]
                        local_imp = self.label_aggregator((w**2 * h), dim=0)
                        if self.label_aggregator in [torch.max]:
                            local_imp, _ = local_imp
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        b = layer.bias.data[idxs].unsqueeze(0)
                        h = layer.bias_grad_labelwise.movedim(-1,0).data[:,idxs]
                        local_imp = self.label_aggregator((b**2 * h), dim=0)
                        if self.label_aggregator in [torch.max]:
                            local_imp, _ = local_imp
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp
        
class GroupSecondTaylorLabelImportance(GroupTaylorLabelImportance):
    """ Grouped first-order taylor expansion of the loss function.
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf

        Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
             
    """
    def accumulate_grad(self, model, label, ignored_layers = None, n_data = 500):
        ignored_layers = ignored_layers if ignored_layers is not None else []
        for m in model.modules():
            if m not in ignored_layers:
                # w * dw + w**2 * h = w * (dw + w * h)
                if hasattr(m, 'weight'):
                    if m.weight.grad is not None:
                        m.weight_grad_labelwise[..., label] += (m.weight.grad.data + m.weight.data * m.weight.grad.data.pow(2)).clone()/n_data
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        if m.bias.grad is not None:
                            m.bias_grad_labelwise[..., label] += (m.bias.grad.data + m.bias.data * m.bias.grad.data.pow(2)).clone()/n_data

