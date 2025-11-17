import torch
import torch.nn as nn
import typing, warnings

import torch_pruning as tp
from torch_pruning.pruner.algorithms.scheduler import linear_scheduler
from torch_pruning.pruner import function
from torch_pruning import ops, dependency
from numbers import Number
from torch_pruning import _helpers

import methods.finetuner_function as fntn_function
from methods.finetuner_function import myLinear 

class Finetuner:
    """
    Finetuner for applying structural pruning to finetuning. 
    """

    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        global_finetuning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        finetuning_ratio: float = 0.5,  # channel/dim finetuning ratio, same as pruning ratio
        finetuning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific finetuning ratio, will cover finetuning_ratio if specified
        rank: int = None, # control the neuron by rank but not ratio
        # ========================= Might not need in finetuning ========================= 
        max_finetuning_ratio: float = 1.0, # might not need to set this in finetuning, just keep it to be 1
        # ========================= Might not need in finetuning ========================= 

        customized_finetuner: typing.Dict[typing.Any, fntn_function.BaseFinetuningFunc] = None, # finetuners for customized layers. E.g., {nn.Linear: my_linear_finetuners}
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to
        finetuning_channel: str = "out",
        # Advanced
        channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        finetune_num_heads: bool = False, # remove entire heads in multi-head attention
        finetune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_finetuning_ratio: float = 0.0, # head finetuning ratio
        head_finetuning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head finetuning ratio
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & finetuning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs
        dtype: str = "fp32", 
        p_dropout: float = 0.0,
        variant: str = 'PruFT',
    ):
        self.model = model
        self.importance = importance

        self.finetuning_ratio = finetuning_ratio
        self.max_finetuning_ratio = min(1, max_finetuning_ratio)
        self.rank = rank
        self.global_finetuning = global_finetuning
        
        if len(num_heads) > 0:
            channel_groups.update(num_heads)

        self.channel_groups = channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to
        self.finetuning_channel = finetuning_channel

        # MHA
        self.num_heads = num_heads
        self.finetune_num_heads = finetune_num_heads
        self.finetune_head_dims = finetune_head_dims
        self.head_finetuning_ratio = head_finetuning_ratio

        ###############################################
        # functions for finetuning

        self.finetune_function = fntn_function.FinetunerBox
        for k, v in self.finetune_function.items():
            self.finetune_function[k].dtype = dtype
        if p_dropout > 0.0:
            self.finetune_function[ops.OPTYPE.LINEAR].p_dropout = p_dropout
        if variant in ['DoRA', 'dora', 'Dora']:
            self.finetune_function[ops.OPTYPE.LINEAR].variant = 'dora'

        self.customized_finetuner = customized_finetuner if customized_finetuner is not None else {}
        ###############################################
        # Count the number of total channels
        # Ignored layers and sub-modules of layers included by finetuning_function since they will be handled by finetuning_function
        self.ignored_layers = []
        self.ignored_params = []
        if ignored_layers is not None:
            for layer in ignored_layers:
                if isinstance(layer, nn.Module):
                    self.ignored_layers.extend(list(layer.modules()))
                elif isinstance(layer, nn.Parameter):
                    self.ignored_params.append(layer)

        total_channels = 0
        total_heads = 0            
        self.module2name = {}

        iter_list = set([ops.type2class(finetuning_type) for finetuning_type in self.finetune_function.keys()] + list(self.customized_finetuner.keys()))
        for layer_type_or_instance in iter_list:      
            for n, m in self.model.named_modules():
                self.module2name[m] = n                
                if isinstance(m, tuple(iter_list)):
                    finetuner = self.get_finetuning_fn(m)
                    if self.finetuning_channel == "out":
                        total_channels += finetuner.get_out_channels(m)
                        if m in self.num_heads:
                            total_heads += self.num_heads[m]
                    elif self.finetuning_channel == "in":
                        total_channels += finetuner.get_in_channels(m)

                # a layer instance or a layer type
                if (m==layer_type_or_instance) or (not isinstance(layer_type_or_instance, torch.nn.Module) and isinstance(m, layer_type_or_instance)):
                    for sub_module in m.modules(): 
                        if sub_module != m:
                            self.ignored_layers.append(sub_module)

        self.total_channels = total_channels
        self.total_heads = total_heads

        #pos_embed, cls_token, dis_token, etc in transformers
        self.individual_parameters = {}
        for n, p in model.named_parameters():
            if '.' not in n:
                self.individual_parameters[n] = p

        ###############################################
        # Layer-specific finetuning ratios. Will cover the global ratio if specified
        finetuning_types = tuple([finetuning_type for finetuning_type in self.finetune_function.keys()])
        self.finetuning_ratio_dict = {}
        if finetuning_ratio_dict is not None:
            for module in finetuning_ratio_dict:
                ratio = finetuning_ratio_dict[module]
                for submodule in module.modules():
                    if isinstance(submodule, finetuning_types):
                        self.finetuning_ratio_dict[submodule] = ratio

        # Head finetuning ratio
        self.head_finetuning_ratio_dict = {}
        if head_finetuning_ratio_dict is not None:
            for module in head_finetuning_ratio_dict:
                ratio = head_finetuning_ratio_dict[module]
                for submodule in module.modules():
                    if isinstance(submodule, finetuning_types):
                        self.head_finetuning_ratio_dict[submodule] = ratio

        ###############################################
        # Detect group convs & group norms
        for m in self.model.modules():
            n_groups = 1
            if hasattr(m, 'groups'):
                n_groups = getattr(m, 'groups')
            elif hasattr(m, 'num_groups'):
                n_groups = getattr(m, 'num_groups')
            else: continue

            if isinstance(m, ops.TORCH_CONV) and n_groups == m.out_channels:
                continue
            if n_groups > 1:
                self.channel_groups[m] = n_groups
    
    def run(self)-> None:
        if self.global_finetuning:
            self.set_finetune_global() 
        else:    
            self.set_finetune_local()

    def estimate_importance(self, layer, finetining_fn) -> torch.Tensor:
        return self.importance(layer, finetining_fn)

    def get_target_finetuning_ratio(self, module) -> float:
        s = self.finetuning_ratio_dict.get(module, self.finetuning_ratio)
        return min(s, self.max_finetuning_ratio)

    def get_target_head_finetuning_ratio(self, module) -> float:
        s = self.head_finetuning_ratio_dict.get(module, self.head_finetuning_ratio)
        return min(s, 1)

    def _is_attn_head(self, layer) -> bool:
        if self.finetuning_channel == "out" and layer in self.num_heads:
            return True
        return False

    def _get_channel_groups(self, module) -> int:
        if module in self.channel_groups:
            return self.channel_groups[module]
        return 1

    def _round_to(self, n_freezed, n_channels, round_to):
        """
        example: 
        n_freezed = 35
        n_channels = 768
        round_to = 2

        768 - 35 = 733
        733 - 733 % 2 = 733-1 = 732
        768 - 732 = 36
        """
        rounded_channels = n_channels - n_freezed #n_finetuned
        rounded_channels = rounded_channels - rounded_channels % round_to 
        n_freezed = n_channels - rounded_channels
        return max(n_freezed, 0)

    def setting_finetuning_layer(self, old_module, finetuning_fn, freezing_idxs):
        name = self.module2name[old_module]
        path = name.split('.')
        module = self.model
        if len(path)>1:
            for p in path[:-1]:
                module = getattr(module, p)
        setattr(module, path[-1], finetuning_fn(old_module, freezing_idxs))  

    def merge_and_unload(self, model):
        for name, layer in model.named_modules():
            if isinstance(layer, myLinear): 
                path = name.split('.')
                module = model
                if len(path)>1:
                    for p in path[:-1]:
                        module = getattr(module, p)
                setattr(module, path[-1], layer.merge_and_unload())  
        return model

    def get_finetuning_fn(self, layer: nn.Module):
        for finetuning_class in self.customized_finetuner.keys():
            if isinstance(layer, finetuning_class):
                return self.customized_finetuner[finetuning_class]
        for finetuning_type in self.finetune_function.keys():
            if isinstance(layer, ops.type2class(finetuning_type)):
                return self.finetune_function[finetuning_type]
        return None

    def set_finetune_local(self):        
        for layer in self.model.modules():
            ##################################
            # Compute raw importance score
            ##################################
            if layer not in self.ignored_layers:
                imp = None
                finetuner = self.get_finetuning_fn(layer)
                bert_attn = (
                    fntn_function.BertAttnFinetuner, 
                    fntn_function.DisentangledAttnFinetuner,
                    )
                if isinstance(finetuner, bert_attn):
                    if self.finetuning_channel == "out":
                        finetuning_fn = fntn_function.finetune_linear_out_channels
                    elif self.finetuning_channel == "in":
                        finetuning_fn = fntn_function.finetune_linear_in_channels         
                    for submodule in layer.modules():
                        if isinstance(submodule, nn.Linear):
                            if imp is None:
                                imp = self.estimate_importance(submodule, finetuning_fn)/3 
                            else: 
                                imp = imp + self.estimate_importance(submodule, finetuning_fn)/3
                elif finetuner is not None:
                    if self.finetuning_channel == "out":
                        imp = self.estimate_importance(layer, finetuner.finetune_out_channels)
                    elif self.finetuning_channel == "in":
                        imp = self.estimate_importance(layer, finetuner.finetune_in_channels)                        
                if imp is None: continue

                ##################################
                # Compute the number of dims/channels to finetune
                ##################################
                if self.finetuning_channel == "out":
                    n_channels = finetuner.get_out_channels(layer)
                    target_finetuning_ratio = self.get_target_finetuning_ratio(layer)
                    if self.rank is not None:
                        n_freezed = max(n_channels - self.rank, 0)
                    else:
                        n_freezed = n_channels - int(n_channels * (target_finetuning_ratio))
                elif self.finetuning_channel == "in":
                    n_channels = finetuner.get_in_channels(layer)
                    target_finetuning_ratio = self.get_target_finetuning_ratio(layer)
                    if self.rank is not None:
                        n_freezed = max(n_channels - self.rank,0)
                    else:
                        n_freezed = n_channels - int(n_channels * (target_finetuning_ratio)) 

                # round to the nearest multiple of round_to
                if self.round_to and self.rank is None:
                    n_freezed = self._round_to(n_freezed, n_channels, self.round_to)
                n_finetuned = n_channels - n_freezed

                ##################################
                # collect freezing idxs
                ##################################
                freezing_idxs = []
                ch_groups = self._get_channel_groups(layer) 
                group_size = n_channels // ch_groups
                attn_head = self._is_attn_head(layer)
                # dims/channels
                if n_finetuned > 0:
                    if (self.finetune_head_dims and attn_head) or (not attn_head):
                        n_freezed_per_group = n_freezed // ch_groups 
                        n_finetuned_per_group = (n_channels // ch_groups) - n_freezed_per_group
                        if self.round_to:
                            n_freezed_per_group = self._round_to(n_freezed_per_group, group_size, self.round_to)
                        n_finetuned_per_group = (n_channels // ch_groups) - n_freezed_per_group
                        if n_finetuned_per_group>0:
                            for chg in range(ch_groups):
                                sub_group_imp = imp[chg*group_size: (chg+1)*group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_freezing_idxs = sub_imp_argsort[:n_freezed_per_group] + chg*group_size # offset
                                freezing_idxs.append(sub_freezing_idxs)
                else: # no channel grouping
                    imp_argsort = torch.argsort(imp)
                    freezing_idxs.append( imp_argsort[:n_freezed] )
                # num heads
                if attn_head and self.finetune_num_heads: # finetune entire attn heads
                    target_head_finetuning_ratio = self.get_target_head_finetuning_ratio(layer)
                    n_head = self.num_heads[layer]
                    n_heads_freezed = n_head - int(n_head * (target_head_finetuning_ratio))
                    n_heads_finetuned = n_head - n_heads_freezed
                    if n_heads_finetuned>0:
                        head_imp = imp.view(ch_groups, -1).mean(1)
                        for head_id in torch.argsort(head_imp)[:n_heads_freezed]:
                            freezing_idxs.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=head_imp.device) )      
                          
                if len(freezing_idxs)==0: continue
                freezing_idxs = torch.unique( torch.cat(freezing_idxs, 0) ).tolist()
                #finetuning setting
                if self.finetuning_channel == "out":
                    self.setting_finetuning_layer(layer, finetuner.finetune_out_channels, freezing_idxs)
                elif self.finetuning_channel == "in":
                    self.setting_finetuning_layer(layer, finetuner.finetune_in_channels, freezing_idxs)                

    def set_finetune_global(self):      
        ##############################################
        # 1. Pre-compute importance for each layer
        ##############################################
        global_importance = []
        global_head_importance = {} # for attn head finetuning
        for layer in self.model.modules():
            if layer not in self.ignored_layers:
                imp = None
                finetuner = self.get_finetuning_fn(layer)
                bert_attn = (
                    fntn_function.BertAttnFinetuner, 
                    fntn_function.DisentangledAttnFinetuner,
                    )
                if isinstance(finetuner, bert_attn):
                    if self.finetuning_channel == "out":
                        finetuning_fn = fntn_function.finetune_linear_out_channels
                    elif self.finetuning_channel == "in":
                        finetuning_fn = fntn_function.finetune_linear_in_channels
                    for submodule in layer.modules():
                        if isinstance(submodule, nn.Linear):
                            if imp is None:
                                imp = self.estimate_importance(submodule, finetuning_fn)/3 
                            else: 
                                imp = imp + self.estimate_importance(submodule, finetuning_fn)/3
                elif finetuner is not None:
                    if self.finetuning_channel == "out":
                        imp = self.estimate_importance(layer, finetuner.finetune_out_channels)
                    elif self.finetuning_channel == "in":
                        imp = self.estimate_importance(layer, finetuner.finetune_in_channels)
                if imp is None: continue
                ch_groups = self._get_channel_groups(layer)
                group_size = len(imp) // ch_groups
                if ch_groups > 1:
                    # Corresponding elements of each group will be removed together.
                    # So we average importance across groups here. For example:
                    # imp = [1, 2, 3, 4, 5, 6] with ch_groups=2.
                    # We have two groups [1,2,3] and [4,5,6].
                    # The average importance should be [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
                    dim_imp = imp.view(ch_groups, -1).mean(dim=0) 
                else:
                    # no grouping
                    dim_imp = imp
                global_importance.append((layer, ch_groups, group_size, dim_imp))
            
                # pre-compute head importance for attn heads
                attn_head = self._is_attn_head(layer)
                if attn_head and self.finetune_num_heads and self.get_target_head_finetuning_ratio(layer)>0:
                    # average importance of each group. For example:
                    # the importance score of the group
                    # imp = [1, 2, 3, 4, 5, 6] with num_heads=2
                    # Note: head1 = [1, 2, 3], head2 = [4, 5, 6]
                    # the average importance is [(1+2+3)/3, (4+5+6)/3] = [2, 5]
                    head_imp = imp.view(ch_groups, -1).mean(1) # average importance by head.
                    global_head_importance[layer] = head_imp

        if len(global_importance) == 0 and len(global_head_importance)==0:
            return
        
        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################
        # Find the threshold for global finetuning
        if len(global_importance)>0:
            concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
            target_finetuning_ratio = self.finetuning_ratio
            n_freezed = len(concat_imp) - int(self.total_channels * (target_finetuning_ratio))
            n_finetuned = len(concat_imp) - n_freezed
            if n_finetuned>0:
                topk_imp, _ = torch.topk(concat_imp, k=n_freezed, largest=False)
                thres = topk_imp[-1]

        # Find the threshold for head finetuning
        if len(global_head_importance)>0:
            concat_head_imp = torch.cat([local_imp[-1] for local_imp in global_head_importance.values()], dim=0)
            target_head_finetuning_ratio = self.head_finetuning_ratio
            n_heads_freezed = len(concat_head_imp) - int(self.total_heads * (target_head_finetuning_ratio))
            n_head_finetuned = len(concat_head_imp) - n_heads_freezed
            if n_head_finetuned>0:
                topk_head_imp, _ = torch.topk(concat_head_imp, k=n_heads_freezed, largest=False)
                head_thres = topk_head_imp[-1]
        
        ##############################################
        # 3. set finetuned parameters
        ##############################################
        for layer, ch_groups, group_size, imp in global_importance:
            finetuner = self.get_finetuning_fn(layer)
            if self.finetuning_channel == "out":
                get_channel_fn = finetuner.get_out_channels
            elif self.finetuning_channel == "in":
                get_channel_fn = finetuner.get_in_channels
            # finetuned feature dims/channels
            freezing_indices = []
            if len(global_importance)>0 and n_finetuned>0:
                if ch_groups > 1: # re-compute importance for each channel group if channel grouping is enabled
                    n_freezed_per_group = len((imp <= thres).nonzero().view(-1))
                    n_finetuned_per_group = len((imp > thres).nonzero().view(-1))
                    if n_finetuned_per_group>0:
                        if self.round_to:
                            n_freezed_per_group = self._round_to(n_freezed_per_group, group_size, self.round_to)
                        attn_head = self._is_attn_head(layer)
                        if not attn_head or self.finetune_head_dims==True:
                            raw_imp = self.estimate_importance(layer) # re-compute importance
                            for chg in range(ch_groups): # determine finetuning indices for each channel group independently
                                sub_group_imp = raw_imp[chg*group_size: (chg+1)*group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_freezing_idxs = sub_imp_argsort[:n_freezed_per_group] + chg*group_size
                                freezing_indices.append(sub_freezing_idxs)
                else:
                    _freezing_indices = (imp <= thres).nonzero().view(-1)
                    imp_argsort = torch.argsort(imp)
                    if len(_freezing_indices)>0 and self.round_to: 
                        n_freezed = len(_freezing_indices)
                        n_channels = get_channel_fn(layer)
                        n_freezed = self._round_to(n_freezed, n_channels, self.round_to)
                        _freezing_indices = imp_argsort[:n_freezed]
                    freezing_indices.append(_freezing_indices)
                        
            # finetune heads
            if len(global_head_importance)>0 and n_head_finetuned>0:
                if layer in global_head_importance:
                    head_imp = global_head_importance[layer]
                    head_freezing_indices = (head_imp <= head_thres).nonzero().view(-1)
                    if len(head_freezing_indices)>0:
                        for head_id in head_freezing_indices:
                            freezing_indices.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=head_imp.device) )

            if len(freezing_indices)==0: continue
            freezing_indices = torch.unique(torch.cat(freezing_indices, 0)).tolist()
            #finetuning setting
            if self.finetuning_channel == "out":
                self.setting_finetuning_layer(layer, finetuner.finetune_out_channels, freezing_indices)
            elif self.finetuning_channel == "in":
                self.setting_finetuning_layer(layer, finetuner.finetune_in_channels, freezing_indices)         
