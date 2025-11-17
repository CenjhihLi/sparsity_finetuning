import abc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
import torch_pruning.pruner.importance as tp_imp_fn
import methods.finetuner_function as fntn_function

from types import MethodType
from torch.autograd import Variable
from numpy import ndarray

__all__=[
    'NormImportance',
    'TaylorImportance',
    'HessianImportance',
    'ZerothOrderTaylorImportance',

    'HessianLabelImportance',
    'TaylorLabelImportance',
    'SecondTaylorLabelImportance',

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

def GreaterQuantilesAvg(x: torch.Tensor, dim: int = 0):
    size = x.size()
    if dim> (len(size)-1):
        raise ValueError(f"The target dim {dim} is illegal. The tensor is {len(size)}-d, dim should be at most {len(size)-1}.\n")
    return torch.quantile(x, q = torch.arange(start=0.5, end = 1.01, step=0.1).to(x.device), dim = dim).mean(0)

def QuantilesAvg(x: torch.Tensor, dim: int = 0):
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

class RandomImportance(tp_imp_fn.Importance):
    """ Random importance estimator
    Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = RandomImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    @torch.no_grad()
    def __call__(self, layer: nn.Module, finetuning_fn, finetuning_dim: int = 0):
        ####################
        # Conv Output
        ####################
        if finetuning_fn == fntn_function.finetune_conv_out_channels:
            local_imp = torch.rand((layer.out_channels))
        ####################
        # Linear Output
        ####################
        elif finetuning_fn == fntn_function.finetune_linear_out_channels:
            local_imp = torch.rand((layer.out_features))
        ####################
        # Conv input
        ####################
        elif finetuning_fn == fntn_function.finetune_conv_in_channels:
            local_imp = torch.rand((layer.in_channels))
        ####################
        # Linear input
        ####################
        elif finetuning_fn == fntn_function.finetune_linear_in_channels:
            local_imp = torch.rand((layer.in_features))
        ####################
        # BatchNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            local_imp = torch.rand((layer.num_features))
        ####################
        # LayerNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            local_imp = torch.rand((layer.normalized_shape[-1]))
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            local_imp = torch.rand((layer.weight.size()[-1]))
        return local_imp

class NormImportance(tp_imp_fn.GroupNormImportance):
    def __call__(self, target, finetuning_fn, finetuning_dim: int = 0):
        if isinstance(target, nn.Parameter):
            return self.param_imp(target, finetuning_dim)
        elif isinstance(target, tuple(self.target_types)):
            return self.layer_imp(target, finetuning_fn)
        else: return None
    
    @torch.no_grad()
    def param_imp(self, param: nn.Parameter, finetuning_dim: int = 0):
        return self._normalize(param.movedim(finetuning_dim, 0).abs().pow(self.p).sum(1), self.normalizer)
    
    @torch.no_grad()
    def dequantize_weight(self, layer: nn.Module):
        weight = layer.weight.clone()
        import bitsandbytes as bnb
        #implementation for only 4 bits dequantize now
        weight_dequantized = bnb.functional.dequantize_4bit(
            weight.data, weight.quant_state
            ).to(torch.float32)
        return weight_dequantized

    @torch.no_grad()
    def layer_imp(self, layer: nn.Module, finetuning_fn):
        ####################
        # Conv/Linear Output
        ####################
        if finetuning_fn in [
            fntn_function.finetune_conv_out_channels,
            fntn_function.finetune_linear_out_channels,
        ]:
            if layer.weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                w = self.dequantize_weight(layer)
            else:
                w = layer.weight
            if hasattr(layer, "transposed") and layer.transposed:
                w = w.data.transpose(1, 0).flatten(1)
            else:
                w = w.data.flatten(1)
            local_imp = w.abs().pow(self.p).sum(1)
        ####################
        # Conv/Linear Input
        ####################
        elif finetuning_fn in [
            fntn_function.finetune_conv_in_channels,
            fntn_function.finetune_linear_in_channels,
        ]:
            if layer.weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                w = self.dequantize_weight(layer)
            else:
                w = layer.weight
            if hasattr(layer, "transposed") and layer.transposed:
                w = (w.data).flatten(1)
            else:
                w = (w.data).transpose(0, 1).flatten(1)
            local_imp = w.abs().pow(self.p).sum(1)
            # repeat importance for group convolutions
            if finetuning_fn == fntn_function.finetune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                local_imp = local_imp.repeat(layer.groups)
            
        ####################
        # BatchNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            # regularize BN
            if layer.affine:
                w = layer.weight.data
                local_imp = w.abs().pow(self.p)
        ####################
        # LayerNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            if layer.elementwise_affine:
                w = layer.weight.data
                local_imp = w.abs().pow(self.p)
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            w = layer.weight.data
            local_imp = w.abs().pow(self.p)

        if isinstance(local_imp, torch.Tensor):
            local_imp = self._normalize(local_imp, self.normalizer)
        return local_imp

    
class WandAImportance(NormImportance):
    """
    refer from: https://github.com/locuslab/wanda/tree/main
    """
    def __init__(self, 
                 p: int=2, 
                 group_reduction: str="mean", 
                 normalizer: str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, ]):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
    
    def register_module(self, model, n_sample):
        p = self.p
        for name, module in model.named_modules():
            if isinstance(module, tuple(self.target_types)):
                if isinstance(module, nn.Conv2d):
                    dim = -3
                    module.register_buffer("wanda", torch.zeros((module.weight.size()[dim]), dtype = torch.float16))
                    module.wanda = module.wanda.to(module.weight.device)
                    def forward(self, input):
                        act = (torch.norm(input.movedim(dim,0).flatten(1).to(torch.float16), p=p, dim=1) ** p)/n_sample
                        self.wanda += act
                        del act
                        return self._conv_forward(input, self.weight, self.bias)
                    module.forward = MethodType(forward, module)
                elif isinstance(module, nn.Linear):
                    dim = -1
                    module.register_buffer("wanda", torch.zeros((module.weight.size()[dim]), dtype = torch.float16))
                    module.wanda = module.wanda.to(module.weight.device)
                    def forward(self, input):
                        act = (torch.norm(input.movedim(dim,0).flatten(1).to(torch.float16), p=p, dim=1) ** p)/n_sample
                        self.wanda += act
                        del act
                        return F.linear(input, self.weight, self.bias)
                    module.forward = MethodType(forward, module)
    
    def remove_module(self, model): 
        for name, module in model.named_modules():
            if hasattr(module, "wanda"):
                delattr(module, "wanda")
                if isinstance(module, nn.Conv2d):
                    def forward(self, input):
                        return self._conv_forward(input, self.weight, self.bias)
                    module.forward = MethodType(forward, module)
                elif isinstance(module, nn.Linear):
                    def forward(self, input):
                        return F.linear(input, self.weight, self.bias)
                    module.forward = MethodType(forward, module)

    @torch.no_grad()
    def param_imp(self, param: nn.Parameter, finetuning_dim: int = 0):
        return None 
    
    @torch.no_grad()
    def layer_imp(self, layer: nn.Module, finetuning_fn):
        ####################
        # Conv/Linear Output
        ####################
        if finetuning_fn in [
            fntn_function.finetune_conv_out_channels,
            fntn_function.finetune_linear_out_channels,
        ]:
            wanda = layer.wanda.unsqueeze(0).unsqueeze(-1) #size = (dout, din, -1 )
            if layer.weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                w = self.dequantize_weight(layer)
            else:
                w = layer.weight
            if hasattr(layer, "transposed") and layer.transposed:
                w = w.data.transpose(1, 0).unsqueeze(-1).flatten(2) #size = (dout, din, -1 )
            else:
                w = w.data.unsqueeze(-1).flatten(2) #size = (dout, din, -1 )
                
            local_imp = w.abs() * (wanda.flatten(1).sum(1).to(w.dtype))
        ####################
        # Conv/Linear Input
        ####################
        elif finetuning_fn in [
            fntn_function.finetune_conv_in_channels,
            fntn_function.finetune_linear_in_channels,
        ]:  
            wanda = layer.wanda.unsqueeze(1).unsqueeze(-1) #size = (din, dout, -1 )

            if layer.weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                w = self.dequantize_weight(layer)
            else:
                w = layer.weight
            if hasattr(layer, "transposed") and layer.transposed:
                w = w.data.unsqueeze(-1).flatten(2) #size = (din, dout, -1 )
            else:
                w = w.data.transpose(0, 1).unsqueeze(-1).flatten(2) #size = (din, dout, -1 )

            local_imp = w.abs() * (wanda.flatten(1).sum(1).to(w.dtype))
            # repeat importance for group convolutions
            if finetuning_fn == fntn_function.finetune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                local_imp = local_imp.repeat(layer.groups)
            
        ####################
        # BatchNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            # regularize BN
            if layer.affine:
                wanda = layer.wanda
                w = layer.weight.data
                local_imp = w.abs() * (wanda.to(w.dtype))
        ####################
        # LayerNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            if layer.elementwise_affine:
                wanda = layer.wanda
                w = layer.weight.data
                local_imp = w.abs() * (wanda.to(w.dtype))
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            wanda = layer.wanda
            w = layer.weight.data
            local_imp = w.abs() * (wanda.to(w.dtype))

        if isinstance(local_imp, torch.Tensor):
            local_imp = self._normalize(local_imp, self.normalizer)
        return local_imp


class ZerothOrderTaylorImportance(NormImportance):
    """ 
    Gradient estimator: Simultaneous Perturbation Stochastic Approximation or SPSA 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 n_estimate: int = 5,
                 multivariable:bool=False, 
                 bias=False,
                 zo_eps = 1e-3,
                 zo_scale = 1.0,
                 target_types: list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias
        self.n_estimate = n_estimate
        self.zo_eps = zo_eps
        self.zo_scale = zo_scale

    def random_seed_init(self, model, ignored_layers, random_seed: int = 2024):
        zo_random_seed_list = []
        random.seed(random_seed)
        random_factors = list(range(10000))
        random_constants = list(range(10000))
        random.shuffle(random_factors)
        random.shuffle(random_constants)
        i=1
        while len(set(zo_random_seed_list))< self.n_estimate * self.data_split:
            random_factor = random_factors.pop(0)
            random_constant = random_constants.pop(0)
            s = random_factor * i + random_constant
            if s not in zo_random_seed_list:
                zo_random_seed_list.append(s)
                i+=1
        
        self.named_parameters_to_optim = []
        self.param2name = {}
        self.zo_random_seeds = {}
        j = 0
        for name, param in model.named_parameters():
            path = name.split('.')
            module = model
            if len(path)>1:
                for p in path[:-1]:
                    module = getattr(module, p)
            if module in ignored_layers:
                continue
            self.named_parameters_to_optim.append((name, param))
            self.param2name[param] = name
            self.zo_random_seeds[name] = [j + zo_seed for zo_seed in zo_random_seed_list]
            j += 1

    def zo_perturb_parameters(self, model, estimate_idx=0, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: to controll the random vector z
            e.g. 
            set seed 
            theta + 1 * z * eps
            set seed
            theta - 1 * z * eps # recover the parameters to be the orginal ones 

        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        # Set the random seed to ensure that we sample the same z for perturbation/estimate  
        # for name, param in model.named_parameters():
        for name, param in self.named_parameters_to_optim:
            random_seed = self.zo_random_seeds[name][estimate_idx]
            torch.manual_seed(random_seed)
            z = torch.normal(mean=0, std=self.zo_scale, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.zo_eps

    @torch.no_grad()
    def zo_forward(self, model, device, data_loader, loss_function = None):
        model.eval()
        sum_loss = 0
        with torch.no_grad():
            for batch_idx, item in enumerate(data_loader):
                if loss_function is None:
                    if isinstance(item, list):
                        # Prepare inputs and move to device
                        inputs, targets = item
                        inputs = Variable(inputs).to(device)
                        targets = Variable(targets).to(device)
                        # Forward pass through the model
                        lm_output = model(inputs, labels = targets)
                        loss = lm_output.loss #negative log likelihood
                    elif isinstance(item, dict):
                        for k, v in item.items():
                            item[k] = Variable(v).to(device)
                        lm_output = model(**item)
                        targets = item["labels"]
                        loss = lm_output.loss #negative log likelihood
                else:
                    data, target = Variable(data).to(device), Variable(target).to(device)
                    out = model(data)
                    loss = loss_function(out, target)
                sum_loss += loss.data.item()
            avgloss = sum_loss / len(data_loader)
        return avgloss
    
    @torch.no_grad()
    def zo_one_batch_forward(self, model, device, item, loss_function = None):
        if loss_function is None:
            if isinstance(item, list):
                # Prepare inputs and move to device
                inputs, targets = item
                inputs = Variable(inputs).to(device)
                targets = Variable(targets).to(device)
                # Forward pass through the model
                lm_output = model(inputs, labels = targets)
                loss = lm_output.loss #negative log likelihood
            elif isinstance(item, dict):
                for k, v in item.items():
                        item[k] = Variable(v).to(device)
                lm_output = model(**item)
                targets = item["labels"]
                loss = lm_output.loss #negative log likelihood
        else:
            data, target = Variable(data).to(device), Variable(target).to(device)
            out = model(data)
            loss = loss_function(out, target)
        return loss.data.item()
    
    def zo_forward_split(self, model, device, data_loader, estimate_idx, scaling_factor, loss_function = None):
        self.zo_perturb_parameters(model = model, estimate_idx = estimate_idx, scaling_factor = scaling_factor)
        model.eval()
        n_batch_split = len(data_loader)//self.data_split
        avgloss_list = []
        sum_loss = 0
        with torch.no_grad():
            for batch_idx, item in enumerate(data_loader):
                loss = self.zo_one_batch_forward(model, device, item, loss_function = None)
                sum_loss += loss
                if (batch_idx + 1) % n_batch_split == 0:
                    avgloss = sum_loss / n_batch_split
                    avgloss_list.append(avgloss)
                    sum_loss = 0
                    self.zo_perturb_parameters(model = model, estimate_idx = estimate_idx, scaling_factor = -scaling_factor)
                    estimate_idx += 1
                    if estimate_idx % self.data_split == 0:
                        break
                    else:
                        self.zo_perturb_parameters(model = model, estimate_idx = estimate_idx, scaling_factor = scaling_factor)
        return avgloss_list
    
    def projected_grad_estimate(self, model, device, data_loader, ignored_layers, loss_function = None, random_seed: int = 2024, data_split: int = 256):        
        self.data_split = data_split
        self.random_seed_init(model = model, ignored_layers = ignored_layers, random_seed = random_seed)
        self.zo_projected_grad_list = []
        
        for estimate_idx in range(self.n_estimate):
            if data_split > 1:
                loss_pos_list = self.zo_forward_split(model, device, data_loader, 
                                                      estimate_idx = estimate_idx*self.data_split, scaling_factor = 1, 
                                                      loss_function = loss_function)
                loss_neg_list = self.zo_forward_split(model, device, data_loader, 
                                                      estimate_idx = estimate_idx*self.data_split, scaling_factor = -1, 
                                                      loss_function = loss_function)
                
                for i in range(data_split):
                    loss_pos = loss_pos_list[i]
                    loss_neg = loss_neg_list[i]
                    projected_grad = ((loss_pos - loss_neg) / (2 * self.zo_eps))
                    print("The {}-th estimate, loss_pos = {}, loss_neg = {}, projected_grad = {}.".format(
                        estimate_idx*data_split + i, loss_pos, loss_neg, projected_grad))
                    self.zo_projected_grad_list.append(projected_grad)
            else:
                # loss of parameters + eps*z
                self.zo_perturb_parameters(model = model, estimate_idx = estimate_idx, scaling_factor = 1)
                loss_pos = self.zo_forward(model, device, data_loader, loss_function)

                # loss of parameters - eps*z
                # (parameters + eps*z - 2*eps*z)
                self.zo_perturb_parameters(model = model, estimate_idx = estimate_idx, scaling_factor = -2)
                loss_neg = self.zo_forward(model, device, data_loader, loss_function)

                projected_grad = ((loss_pos - loss_neg) / (2 * self.zo_eps))
                print("The {}-th estimate, loss_pos = {}, loss_neg = {}, projected_grad = {}.".format(
                        estimate_idx, loss_pos, loss_neg, projected_grad))
                self.zo_projected_grad_list.append(projected_grad)
                # Reset model back to original parameters
                # (parameters + eps*z - 2*eps*z + eps*z)
                self.zo_perturb_parameters(model = model, estimate_idx = estimate_idx, scaling_factor = 1)
    
    def zo_estimate(self, param: nn.Parameter, original_SPSA: bool = False, eps = 1e-7):
        name = self.param2name[param] 
        seeds = self.zo_random_seeds[name]
        dw = torch.zeros(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        for estimate_idx in range(self.n_estimate * self.data_split):
            projected_grad = self.zo_projected_grad_list[estimate_idx]
            random_seed = seeds[estimate_idx]
            torch.manual_seed(random_seed)
            z = torch.normal(mean=0, std=self.zo_scale, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if original_SPSA:
                z = torch.where(z>0, torch.maximum(z, eps), torch.minimum(z, -eps))  
                est = projected_grad / (self.n_estimate * self.data_split * z)
                dw += est
            else:
                est = projected_grad * z /(self.n_estimate * self.data_split)
                dw += est
        """
        A better way:
        Store each est above
        Unsqueeze a new dim
        Then concat est on the new dim
        Then compute variance on the new dim
        """
        var = torch.zeros(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        for estimate_idx in range(self.n_estimate * self.data_split):
            projected_grad = self.zo_projected_grad_list[estimate_idx]
            random_seed = seeds[estimate_idx]
            torch.manual_seed(random_seed)
            z = torch.normal(mean=0, std=self.zo_scale, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if original_SPSA:
                z = torch.where(z>0, torch.maximum(z, eps), torch.minimum(z, -eps))  
                est = (projected_grad / z - dw).pow(2) / (self.n_estimate * self.data_split)
                var += est
            else:
                est = (projected_grad * z- dw).pow(2) /(self.n_estimate * self.data_split)
                var += est
        
        print ("Param: {}, ZO Gradient statistics: min {}, median {}, mean {}, max {}.".format(
                name, dw.min(), dw.median(), dw.mean(), dw.max()))
        print ("Variance statistics: min {}, median {}, mean {}, max {}.".format(
                var.min(), var.median(), var.mean(), var.max()))
        return dw

    @torch.no_grad()
    def param_imp(self, param: nn.Parameter, finetuning_dim: int = 0):
        dw = self.zo_estimate(param)

        w = param.data.movedim(finetuning_dim, 0).flatten(1)
        dw = dw.data.movedim(finetuning_dim, 0).flatten(1)
        if self.multivariable:
            local_imp = (w * dw).sum(1).abs()
        else:
            local_imp = (w * dw).abs().sum(1)
        return local_imp
    
    @torch.no_grad()
    def layer_imp(self, layer: nn.Module, finetuning_fn):
        if not isinstance(layer, tuple(self.target_types)):
            return None
        ####################
        # Conv/Linear Output
        ####################
        if finetuning_fn in [
            fntn_function.finetune_conv_out_channels,
            fntn_function.finetune_linear_out_channels,
        ]:
            dw = self.zo_estimate(layer.weight)
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.data.transpose(1, 0).flatten(1)
                dw = dw.data.transpose(1, 0).flatten(1)
            else:
                w = layer.weight.data.flatten(1)
                dw = dw.data.flatten(1)
            if self.multivariable:
                local_imp = (w * dw).sum(1).abs()
            else:
                local_imp = (w * dw).abs().sum(1)

        ####################
        # Conv/Linear Input
        ####################
        elif finetuning_fn in [
            fntn_function.finetune_conv_in_channels,
            fntn_function.finetune_linear_in_channels,
        ]:
            dw = self.zo_estimate(layer.weight)
            if hasattr(layer, "transposed") and layer.transposed:
                w = (layer.weight).flatten(1)
                dw = dw.flatten(1)
            else:
                w = (layer.weight).transpose(0, 1).flatten(1)
                dw = dw.transpose(0, 1).flatten(1)
            if self.multivariable:
                local_imp = (w * dw).sum(1).abs()
            else:
                local_imp = (w * dw).abs().sum(1)
                
            # repeat importance for group convolutions
            if finetuning_fn == fntn_function.finetune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                local_imp = local_imp.repeat(layer.groups)
        ####################
        # BatchNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            # regularize BN
            if layer.affine:
                w = layer.weight.data
                dw = self.zo_estimate(layer.weight)
                local_imp = (w * (dw.data)).abs()
        ####################
        # LayerNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            if layer.elementwise_affine:
                w = layer.weight.data
                dw = self.zo_estimate(layer.weight)
                local_imp = (w * (dw.data)).abs()
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            w = layer.weight.data
            dw = self.zo_estimate(layer.weight)
            local_imp = (w * (dw.data)).abs()

        if isinstance(local_imp, torch.Tensor):
            local_imp = self._normalize(local_imp, self.normalizer)
        return local_imp

class TaylorImportance(NormImportance):
    """ Grouped first-order taylor expansion of the loss function.
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf

        Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                loss = loss_fn(model(inputs), labels)
                loss.backward() # compute gradients
                scorer = GroupTaylorImportance()    
                for m in model.modules():
                    imp_score = scorer(m, finetuner_function)    
                    idx = torch.argsort()[:ratio]
                    new_layer = finetuner_function(m, idx)
            ``` 
    """
    def __init__(self, 
                 group_reduction: str = "mean", 
                 normalizer: str = 'mean', 
                 multivariable: bool = False, 
                 bias = False,
                 target_types: list = [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias

    @torch.no_grad()
    def param_imp(self, param: nn.Parameter, finetuning_dim: int = 0):
        w = param.data.movedim(finetuning_dim, 0).flatten(1)
        dw = param.grad.data.movedim(finetuning_dim, 0).flatten(1)
        if self.multivariable:
            local_imp = (w * dw).sum(1).abs()
        else:
            local_imp = (w * dw).abs().sum(1)
        return self._normalize(local_imp, self.normalizer)
    
    @torch.no_grad()
    def layer_imp(self, layer: nn.Module, finetuning_fn):
        if not isinstance(layer, tuple(self.target_types)):
            return None
        ####################
        # Conv/Linear Output
        ####################
        if finetuning_fn in [
            fntn_function.finetune_conv_out_channels,
            fntn_function.finetune_linear_out_channels,
        ]:
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.data.transpose(1, 0).flatten(1)
                dw = layer.weight.grad.data.transpose(1, 0).flatten(1)
            else:
                w = layer.weight.data.flatten(1)
                dw = layer.weight.grad.data.flatten(1)
            if self.multivariable:
                local_imp = (w * dw).sum(1).abs()
            else:
                local_imp = (w * dw).abs().sum(1)
        ####################
        # Conv/Linear Input
        ####################
        elif finetuning_fn in [
            fntn_function.finetune_conv_in_channels,
            fntn_function.finetune_linear_in_channels,
        ]:
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
            if finetuning_fn == fntn_function.finetune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                local_imp = local_imp.repeat(layer.groups)
        ####################
        # BatchNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            # regularize BN
            if layer.affine:
                w = layer.weight.data
                dw = layer.weight.grad.data
                local_imp = (w*dw).abs()
        ####################
        # LayerNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            if layer.elementwise_affine:
                w = layer.weight.data
                dw = layer.weight.grad.data
                local_imp = (w*dw).abs()
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            w = layer.weight.data
            dw = layer.weight.grad.data
            local_imp = (w*dw).abs()

        if isinstance(local_imp, torch.Tensor):
            local_imp = self._normalize(local_imp, self.normalizer)
        return local_imp

class HessianImportance(NormImportance):
    """Grouped Optimal Brain Damage:
       https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html

       Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python                
                inputs, labels = ...
                scorer = GroupHessianImportance()   
                scorer.zero_grad() # clean the acuumulated gradients if necessary
                loss = loss_fn(model(inputs), labels)
                for l in loss:
                    model.zero_grad() # clean the model gradients
                    l.backward(retain_graph=True) # compute gradients for each sample
                    scorer.accumulate_grad(model) # accumulate gradients of each sample
                scorer = GroupTaylorImportance()    
                for m in model.modules():
                    imp_score = scorer(m, finetuner_function)    
                    idx = torch.argsort()[:ratio]
                    new_layer = finetuner_function(m, idx)
            ``` 
    """
    def __call__(self, target, finetuning_fn, finetuning_dim: int = 0):
        if len(self._accu_grad) > 0: # fill gradients so that we can re-use the implementation for Taylor
            for p, g in self._accu_grad.items():
                p.grad.data = g / self._counter[p]
            self.zero_grad()

        if isinstance(target, nn.Parameter):
            return self.param_imp(target, finetuning_dim)
        elif isinstance(target, tuple(self.target_types)):
            return self.layer_imp(target, finetuning_fn)
        else: return None

    def zero_grad(self):
        self._accu_grad = {}
        self._counter = {}

    def accumulate_grad(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._accu_grad:
                    self._accu_grad[param] = param.grad.data.clone().pow(2)
                else:
                    self._accu_grad[param] += param.grad.data.clone().pow(2)
                
                if name not in self._counter:
                    self._counter[param] = 1
                else:
                    self._counter[param] += 1

    @torch.no_grad()
    def param_imp(self, param: nn.Parameter, finetuning_dim: int = 0):
        w = param.data.movedim(finetuning_dim, 0).flatten(1)
        h = param.grad.data.movedim(finetuning_dim, 0).flatten(1)
        return self._normalize((w**2 * h).sum(1), self.normalizer)
    
    @torch.no_grad()
    def layer_imp(self, layer: nn.Module, finetuning_fn):
        if not isinstance(layer, tuple(self.target_types)):
            return None
        ####################
        # Conv/Linear Output
        ####################
        if finetuning_fn in [
            fntn_function.finetune_conv_out_channels,
            fntn_function.finetune_linear_out_channels,
        ]:
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.data.transpose(1, 0).flatten(1)
                h = layer.weight.grad.data.transpose(1, 0).flatten(1)
            else:
                w = layer.weight.data.flatten(1)
                h = layer.weight.grad.data.flatten(1)
            local_imp = (w**2 * h).sum(1)
        ####################
        # Conv/Linear Input
        ####################
        elif finetuning_fn in [
            fntn_function.finetune_conv_in_channels,
            fntn_function.finetune_linear_in_channels,
        ]:
            if hasattr(layer, "transposed") and layer.transposed:
                w = (layer.weight).flatten(1)
                h = (layer.weight.grad).flatten(1)
            else:
                w = (layer.weight).transpose(0, 1).flatten(1)
                h = (layer.weight.grad).transpose(0, 1).flatten(1)
            local_imp = (w**2 * h).sum(1)
                
            # repeat importance for group convolutions
            if finetuning_fn == fntn_function.finetune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                local_imp = local_imp.repeat(layer.groups)
        ####################
        # BatchNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            # regularize BN
            if layer.affine:
                w = layer.weight.data
                h = layer.weight.grad.data
                local_imp = (w**2 * h)
        ####################
        # LayerNorm
        ####################
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            if layer.elementwise_affine:
                w = layer.weight.data
                h = layer.weight.grad.data
                local_imp = (w**2 * h)
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            w = layer.weight.data
            h = layer.weight.grad.data
            local_imp = (w**2 * h)

        if isinstance(local_imp, torch.Tensor):
            local_imp = self._normalize(local_imp, self.normalizer)
        return local_imp


class TaylorLabelImportance(NormImportance):
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

    def __call__(self, target, finetuning_fn, finetuning_dim: int = 0, labelwise_grad: torch.Tensor = None):
        if isinstance(target, nn.Parameter):
            return self.param_imp(target, labelwise_grad, finetuning_dim)
        elif isinstance(target, tuple(self.target_types)):
            return self.layer_imp(target, finetuning_fn)
        else: return None

    @torch.no_grad()
    def param_imp(self, param: nn.Parameter, labelwise_grad: torch.Tensor , finetuning_dim: int = 0):
        w = param.data.movedim(finetuning_dim, 0).flatten(1).unsqueeze(0)
        dw = labelwise_grad.data.movedim(finetuning_dim, 0).movedim(-1, 0).flatten(2)
        local_imp = self.label_aggregator((w * dw).sum(2).abs(), dim=0)
        if self.label_aggregator in [torch.max]:
            local_imp, _ = local_imp
        return self._normalize(local_imp, self.normalizer)
    
    @torch.no_grad()
    def layer_imp(self, layer: nn.Module, finetuning_fn):      
        if finetuning_fn in [
            fntn_function.finetune_conv_out_channels,
            fntn_function.finetune_linear_out_channels,
        ]:  
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.data.transpose(1, 0).flatten(1).unsqueeze(0)
                dw = layer.weight_grad_labelwise.movedim(-1,0).data.transpose(2, 1).flatten(2)
            else:
                w = layer.weight.data.flatten(1).unsqueeze(0)
                dw = layer.weight_grad_labelwise.movedim(-1,0).data.flatten(2)
            if self.multivariable:
                local_imp = self.label_aggregator((w * dw).sum(2).abs(), dim=0)
                if self.label_aggregator in [torch.max]:
                    local_imp, _ = local_imp
            else:
                local_imp = self.label_aggregator((w * dw).abs().sum(2), dim=0)
                if self.label_aggregator in [torch.max]:
                    local_imp, _ = local_imp                
        # Conv/Linear Input
        elif finetuning_fn in [
            fntn_function.finetune_conv_in_channels,
            fntn_function.finetune_linear_in_channels,
        ]:  
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
            if finetuning_fn == fntn_function.finetune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                local_imp = local_imp.repeat(layer.groups)
        # BN
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            # regularize BN
            if layer.affine:
                w = layer.weight.data.unsqueeze(0)
                dw = layer.weight_grad_labelwise.movedim(-1,0).data
                local_imp = self.label_aggregator((w*dw).abs(), dim=0)
                if self.label_aggregator in [torch.max]:
                    local_imp, _ = local_imp  
        # LN
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            print("finetuning_fn in [layernorm.out]")   
            if layer.elementwise_affine:
                w = layer.weight.data.unsqueeze(0)
                dw = layer.weight_grad_labelwise.movedim(-1,0).data
                local_imp = self.label_aggregator((w*dw).abs(), dim=0)
                if self.label_aggregator in [torch.max]:
                    local_imp, _ = local_imp
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            w = layer.weight.data.unsqueeze(0)
            dw = layer.weight_grad_labelwise.movedim(-1,0).data
            local_imp = self.label_aggregator((w*dw).abs(), dim=0)
            if self.label_aggregator in [torch.max]:
                local_imp, _ = local_imp

        if isinstance(local_imp, torch.Tensor):
            local_imp = self._normalize(local_imp, self.normalizer)
        return local_imp

class HessianLabelImportance(TaylorLabelImportance):
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
    def param_imp(self, param: nn.Parameter, labelwise_grad: torch.Tensor , finetuning_dim: int = 0):
        w = param.data.movedim(finetuning_dim, 0).flatten(1).unsqueeze(0)
        h = labelwise_grad.data.movedim(finetuning_dim, 0).movedim(-1, 0).flatten(2)
        local_imp = self.label_aggregator((w**2 * h).sum(2), dim=0)
        if self.label_aggregator in [torch.max]:
            local_imp, _ = local_imp
        return self._normalize(local_imp, self.normalizer)
    
    @torch.no_grad()
    def layer_imp(self, layer: nn.Module, finetuning_fn):       
        if finetuning_fn in [
            fntn_function.finetune_conv_out_channels,
            fntn_function.finetune_linear_out_channels,
        ]:
            if layer.weight.grad is not None:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0).flatten(1).unsqueeze(0)
                    h = layer.weight_grad_labelwise.movedim(-1,0).data.transpose(2, 1).flatten(2)
                else:
                    w = layer.weight.data.flatten(1).unsqueeze(0)
                    h = layer.weight_grad_labelwise.movedim(-1,0).data.flatten(2)
                local_imp = self.label_aggregator((w**2 * h).sum(2), dim=0)
                if self.label_aggregator in [torch.max]:
                    local_imp, _ = local_imp                
        # Conv in_channels
        elif finetuning_fn in [
            fntn_function.finetune_conv_in_channels,
            fntn_function.finetune_linear_in_channels,
        ]:
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
                if finetuning_fn == fntn_function.finetune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)

        # BN
        elif finetuning_fn == fntn_function.finetune_batchnorm_out_channels:
            if layer.affine:
                if layer.weight.grad is not None:
                    w = layer.weight.data.unsqueeze(0)
                    h = layer.weight_grad_labelwise.movedim(-1,0).data
                    local_imp = self.label_aggregator((w**2 * h), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp        
        # LN
        elif finetuning_fn == fntn_function.finetune_layernorm_out_channels:
            if layer.elementwise_affine:
                if layer.weight.grad is not None:
                    w = layer.weight.data.unsqueeze(0)
                    h = layer.weight_grad_labelwise.movedim(-1,0).data
                    local_imp = self.label_aggregator((w**2 * h), dim=0)
                    if self.label_aggregator in [torch.max]:
                        local_imp, _ = local_imp
        elif finetuning_fn == fntn_function.finetune_llamanorm_out_channels:
            if layer.weight.grad is not None:
                w = layer.weight.data.unsqueeze(0)
                h = layer.weight_grad_labelwise.movedim(-1,0).data
                local_imp = self.label_aggregator((w**2 * h), dim=0)
                if self.label_aggregator in [torch.max]:
                    local_imp, _ = local_imp

        if isinstance(local_imp, torch.Tensor):
            local_imp = self._normalize(local_imp, self.normalizer)
        return local_imp
        
class SecondTaylorLabelImportance(TaylorLabelImportance):
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
                # w * dw + w**2 * h = w * (dw + w * h)
                if hasattr(m, 'weight'):
                    if m.weight.grad is not None:
                        m.weight_grad_labelwise[..., label] += (m.weight.grad.data + m.weight.data * m.weight.grad.data.pow(2)).clone()/n_data
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        if m.bias.grad is not None:
                            m.bias_grad_labelwise[..., label] += (m.bias.grad.data + m.bias.data * m.bias.grad.data.pow(2)).clone()/n_data
