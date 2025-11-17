import torch
import torch.nn as nn
import typing, warnings

import torch_pruning as tp
from torch_pruning.pruner.algorithms.scheduler import linear_scheduler
from torch_pruning.pruner import function
from torch_pruning import ops, dependency
from numbers import Number
from torch_pruning import _helpers

import methods.tp_finetuner as fntn_function

class myDependencyGraph(dependency.DependencyGraph):
    def __init__(self):
        _dummy_pruners = {
            ops.OPTYPE.CONCAT: ops.DummyPruner(),
            ops.OPTYPE.SPLIT: ops.DummyPruner(),
            ops.OPTYPE.ELEMENTWISE: ops.ElementWisePruner(),
            ops.OPTYPE.RESHAPE: ops.ReshapePruner(),
            ops.OPTYPE.UNBIND: ops.UnbindPruner(),
            ops.OPTYPE.CUSTOMIZED: ops.CustomizedPruner(), # just a placeholder
        }
        self.REGISTERED_PRUNERS = fntn_function.FinetunerBox.copy()  # shallow copy
        self.REGISTERED_PRUNERS.update(_dummy_pruners) # merge dummy pruners
        self.CUSTOMIZED_PRUNERS = {} # user-customized pruners

        self.IGNORED_LAYERS_IN_TRACING = []

        # cache pruning functions for fast lookup
        self._in_channel_pruning_fn = set([p.prune_in_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_in_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None])
        self._out_channel_pruning_fn = set([p.prune_out_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_out_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None])
        self._op_id = 0 # operatior id, will be increased by 1 for each new operator

        # Pruning History
        self._pruning_history = []

    def get_pruning_group(
        self,
        module: nn.Module,
        pruning_fn: typing.Callable,
        idxs: typing.Sequence[int],
    ) -> dependency.Group:
        """
        Get the pruning group for a given module.

            Args:
                module (nn.Module): The module to be pruned.
                pruning_fn (Callable): The pruning function.
                idxs (list or tuple): The indices of channels/dimensions.

            Returns:
                Group: The pruning group containing the dependencies and indices.

            Raises:
                ValueError: If the module is not in the dependency graph.
        """
        if module not in self.module2node:
            raise ValueError(
                "Module {} is not in the dependency graph.".format(module)
            )
        if isinstance(module, tp.ops.TORCH_CONV) and module.groups == module.out_channels and module.out_channels>1:
            pruning_fn = fntn_function.finetune_depthwise_conv_out_channels
        if isinstance(idxs, Number):
            idxs = [idxs]
        
        idxs = [ _helpers._HybridIndex(idx=i, root_idx=i) for i in idxs ] # idxs == root_idxs for the root layer

        self.update_index_mapping()
        group = dependency.Group()

        #  the user pruning operation
        root_node = self.module2node[module]
        group.add_dep(
            dep=dependency.Dependency(pruning_fn, pruning_fn, source=root_node, target=root_node), 
            idxs=idxs,
        )

        visited_node = set()

        def _fix_dependency_graph_non_recursive(dep, idxs, *args):
            processing_stack = [(dep, idxs)]
            while len(processing_stack) > 0:
                dep, idxs = processing_stack.pop(-1)
                node, fn = dep.target, dep.handler
                visited_node.add(node)
    
                for new_dep in node.dependencies:
                    if new_dep.is_triggered_by(fn):
                        new_indices = idxs
                        for mapping in new_dep.index_mapping:
                            if mapping is not None:
                                new_indices = mapping(new_indices)

                        if len(new_indices) == 0:
                            continue
                        if (new_dep.target in visited_node) and group.has_pruning_op(
                            new_dep, new_indices
                        ):
                            continue
                        else:
                            group.add_dep(new_dep, new_indices)
                            processing_stack.append(
                                (new_dep, new_indices)
                            )

        _fix_dependency_graph_non_recursive(*group[0])
        # merge pruning ops
        merged_group = dependency.Group()
        for dep, idxs in group.items:
            if isinstance(dep.target.module, nn.Parameter): #and dep.target.module in self.ignored_params:
                skip=False
                for ignored_p in self.ignored_params:
                    if dep.target.module is ignored_p:
                        skip=True
                        break
                if skip:
                    continue
            merged_group.add_and_merge(dep, idxs)
        merged_group._DG = self
        for i in range(len(merged_group)):
            hybrid_idxs = merged_group[i].idxs
            idxs = _helpers.to_plain_idxs(hybrid_idxs)
            root_idxs = _helpers.to_root_idxs(hybrid_idxs)
            merged_group[i] = _helpers.GroupItem(merged_group[i].dep, idxs) # transform _helpers._HybridIndex to plain index
            merged_group[i].root_idxs = root_idxs
        return merged_group

class MetaFinetuner:
    """
        Finetuner for applying structural pruning to finetuning. 
    """

    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        global_finetuning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        finetuning_ratio: float = 0.5,  # channel/dim finetuning ratio, same as pruning ratio
        finetuning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific finetuning ratio, will cover finetuning_ratio if specified
        rank: int = None, # control the neuron by rank but not ratio
        
        # ========================= Might not need in finetuning ========================= 
        max_finetuning_ratio: float = 1.0, # maximum pruning ratio. useful if over-pruning happens. 
        # ========================= Might not need in finetuning ========================= 

        customized_finetuner: typing.Dict[typing.Any, fntn_function.BaseFinetuningFunc] = None, # finetuners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        finetune_num_heads: bool = False, # remove entire heads in multi-head attention
        finetune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_finetuning_ratio: float = 0.0, # head finetuning ratio
        head_finetuning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head finetuning ratio
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs
    ):
        self.model = model
        self.importance = importance

        self.finetuning_ratio = finetuning_ratio
        self.max_finetuning_ratio = min(1, max_finetuning_ratio)
        self.rank = rank
        self.global_finetuning = global_finetuning
        
        if len(num_heads) > 0:
            out_channel_groups.update(num_heads)

        self.in_channel_groups = in_channel_groups
        self.out_channel_groups = out_channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to

        # MHA
        self.num_heads = num_heads
        self.finetune_num_heads = finetune_num_heads
        self.finetune_head_dims = finetune_head_dims
        self.head_finetuning_ratio = head_finetuning_ratio

        ###############################################
        # Ignored layers and submodules
        self.ignored_layers = []
        self.ignored_params = []
        if ignored_layers is not None:
            for layer in ignored_layers:
                if isinstance(layer, nn.Module):
                    self.ignored_layers.extend(list(layer.modules()))
                elif isinstance(layer, nn.Parameter):
                    self.ignored_params.append(layer)

        ###############################################
        # Build dependency graph
        self.customized_finetuner = customized_finetuner if customized_finetuner is not None else {}

        self.DG = myDependencyGraph().build_dependency(
            model,
            example_inputs = example_inputs,
            forward_fn = forward_fn,
            output_transform = output_transform,
            unwrapped_parameters = unwrapped_parameters,
            customized_pruners = self.customized_finetuner,
            ignored_params = self.ignored_params,
        )

        ###############################################
        # TODO: Might not need some codes for pruning
        # Layer-specific finetuning ratios. Will cover the global ratio if specified
        self.finetuning_ratio_dict = {}
        if finetuning_ratio_dict is not None:
            for module in finetuning_ratio_dict:
                ratio = finetuning_ratio_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    if isinstance(submodule, prunable_types):
                        self.finetuning_ratio_dict[submodule] = ratio

        # Head finetuning ratio
        self.head_finetuning_ratio_dict = {}
        if head_finetuning_ratio_dict is not None:
            for module in head_finetuning_ratio_dict:
                ratio = head_finetuning_ratio_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    if isinstance(submodule, prunable_types):
                        self.head_finetuning_ratio_dict[submodule] = ratio

        ###############################################
        # Detect group convs & group norms
        for m in self.model.modules():
            layer_pruner = self.DG.get_pruner_of_module(m)
            in_ch_group = layer_pruner.get_in_channel_groups(m)
            out_ch_group = layer_pruner.get_out_channel_groups(m)
            if isinstance(m, ops.TORCH_CONV) and m.groups == m.out_channels:
                continue
            if in_ch_group > 1:
                self.in_channel_groups[m] = in_ch_group
            if out_ch_group > 1:
                self.out_channel_groups[m] = out_ch_group
            
        ###############################################
        # Initial channels/dims of each layer
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        self.init_num_heads = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)
                if m in self.num_heads:
                    self.init_num_heads[m] = self.num_heads[m]
        
        ###############################################
        # Count the number of total channels at initialization
        if self.global_finetuning:
            initial_total_channels = 0
            initial_total_heads = 0
            for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
                group = self._downstream_node_as_root_if_attention(group)
                initial_total_channels += ( (self.DG.get_out_channels(group[0][0].target.module) ) // self._get_channel_groups(group) )
                for dep, _ in group:
                    if dep.target.module in self.num_heads and self.DG.is_out_channel_pruning_fn(dep.handler):
                        initial_total_heads += self.num_heads[dep.target.module]
                        break # only count heads once
            self.initial_total_channels = initial_total_channels
            self.initial_total_heads = initial_total_heads

    def run(self, interactive=False)-> typing.Union[typing.Generator, None]:
        pruning_method = self.set_finetune_global if self.global_finetuning else self.set_finetune_local
        if interactive: # yield groups for interactive pruning
            return pruning_method
        else:
            for group in pruning_method():
                group.prune()

    def estimate_importance(self, group) -> torch.Tensor:
        return self.importance(group)

    def get_target_finetuning_ratio(self, module) -> float:
        s = self.finetuning_ratio_dict.get(module, self.finetuning_ratio)
        return min(s, self.max_finetuning_ratio)

    def get_target_head_finetuning_ratio(self, module) -> float:
        s = self.head_finetuning_ratio_dict.get(module, self.head_finetuning_ratio)
        return min(s, 1)

    def update_regularizer(self) -> None:
        pass

    def regularize(self, model, loss) -> typing.Any:
        """ Model regularizor for sparse training
        """
        pass

    def _is_attn_group(self, group) -> bool:
        is_attn = False
        qkv_layers = []
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if self.DG.is_out_channel_pruning_fn(pruning_fn) and module in self.num_heads:
                qkv_layers.append(module)
                is_attn = True
        return is_attn, qkv_layers

    def _get_channel_groups(self, group) -> int:
        ch_groups = 1

        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            channel_groups = self.out_channel_groups if self.DG.is_out_channel_pruning_fn(pruning_fn) else self.in_channel_groups

            if module in channel_groups:
                ch_groups = channel_groups[module]

        return ch_groups  # no channel grouping

    def _downstream_node_as_root_if_attention(self, group):
        # Use a downstream node as the root if torch.unbind exists. TODO: find a general way to handle torch.unbind in timm
        is_attention = False
        downstream_dep = None
        for _dep, _idxs in group:
            if _dep.source.module in self.num_heads and self.DG.is_out_channel_pruning_fn(_dep.handler):
                is_attention = True
            if isinstance(_dep.target.module, tuple(self.root_module_types)) and self.DG.is_in_channel_pruning_fn(_dep.handler):
                downstream_dep = _dep
        if is_attention and downstream_dep is not None: # use a downstream node as the root node for attention layers
            group = self.DG.get_pruning_group(downstream_dep.target.module, downstream_dep.handler, _idxs)
        return group

    def _round_to(self, n_freezed, current_channels, round_to):
        rounded_channels = current_channels - n_freezed #n_finetuned
        rounded_channels = rounded_channels - rounded_channels % round_to 
        n_freezed = current_channels - rounded_channels
        return max(n_freezed, 0)

    def set_finetune_local(self) -> typing.Generator:        
        #print("Computing importance for each layer...")        
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            ##################################
            # Compute raw importance score
            ##################################
            group = self._downstream_node_as_root_if_attention(group)
            module = group[0][0].target.module
            finetuning_fn = group[0][0].handler
            ch_groups = self._get_channel_groups(group) 
            imp = self.estimate_importance(group)
            if imp is None: continue

            ##################################
            # Compute the number of dims/channels to prune
            ##################################
            if self.DG.is_out_channel_pruning_fn(finetuning_fn):
                current_channels = self.DG.get_out_channels(module)
                target_finetuning_ratio = self.get_target_finetuning_ratio(module)
                if self.rank is not None:
                    n_freezed = max(current_channels - self.rank, 0)
                else:
                    n_freezed = current_channels - int(
                        self.layer_init_out_ch[module] *
                        (target_finetuning_ratio)
                    )
            else:
                current_channels = self.DG.get_in_channels(module)
                target_finetuning_ratio = self.get_target_finetuning_ratio(module)
                if self.rank is not None:
                    n_freezed = max(current_channels - self.rank, 0)
                else:
                    n_freezed = current_channels - int(
                        self.layer_init_in_ch[module] *
                        (target_finetuning_ratio)
                    )

            # round to the nearest multiple of round_to
            if self.round_to and self.rank is None:
                n_freezed = self._round_to(n_freezed, current_channels, self.round_to)
            #print("rounded n_freezed: {}".format(n_freezed))

            ##################################
            # collect freezing idxs
            ##################################
            freezing_idxs = []
            _is_attn, qkv_layers = self._is_attn_group(group)
            group_size = current_channels // ch_groups
            # dims/channels
            if n_freezed > 0:
                if (self.finetune_head_dims and _is_attn) or (not _is_attn):
                    n_freezed_per_group = n_freezed // ch_groups 
                    if self.round_to:
                        n_freezed_per_group = self._round_to(n_freezed_per_group, group_size, self.round_to)
                    if n_freezed_per_group>0:
                        for chg in range(ch_groups):
                            sub_group_imp = imp[chg*group_size: (chg+1)*group_size]
                            sub_imp_argsort = torch.argsort(sub_group_imp)
                            sub_freezing_idxs = sub_imp_argsort[:n_freezed_per_group] + chg*group_size # offset
                            freezing_idxs.append(sub_freezing_idxs)
            else: # no channel grouping
                imp_argsort = torch.argsort(imp)
                freezing_idxs.append( imp_argsort[:n_freezed] )
            # num heads
            if _is_attn and self.finetune_num_heads: # Prune entire attn heads
                target_head_finetuning_ratio = self.get_target_head_finetuning_ratio(qkv_layers[0])
                n_heads_freezed = self.num_heads[qkv_layers[0]] - int(self.init_num_heads[qkv_layers[0]] * (target_head_finetuning_ratio))
                if n_heads_freezed>0:
                    head_imp = imp.view(ch_groups, -1).mean(1)
                    for head_id in torch.argsort(head_imp)[:n_heads_freezed]:
                        freezing_idxs.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=head_imp.device) )      
                          
            if len(freezing_idxs)==0: continue
            freezing_idxs = torch.unique( torch.cat(freezing_idxs, 0) ).tolist()
            #print("freezing_idxs: {}".format(freezing_idxs))
            group = self.DG.get_pruning_group(
                module, finetuning_fn, freezing_idxs)
            if self.DG.check_pruning_group(group):
                yield group 

    def set_finetune_global(self) -> typing.Generator:        
        ##############################################
        # 1. Pre-compute importance for each group
        ##############################################
        global_importance = []
        global_head_importance = {} # for attn head finetuning
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            group = self._downstream_node_as_root_if_attention(group) # use a downstream node as the root node for attention layers
            ch_groups = self._get_channel_groups(group)
            imp = self.estimate_importance(group) # raw importance score
            group_size = len(imp) // ch_groups
            if imp is None: continue
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
            global_importance.append((group, ch_groups, group_size, dim_imp))
            
            # pre-compute head importance for attn heads
            _is_attn, qkv_layers = self._is_attn_group(group)
            if _is_attn and self.finetune_num_heads and self.get_target_head_finetuning_ratio(qkv_layers[0])>0:
                # average importance of each group. For example:
                # the importance score of the group
                # imp = [1, 2, 3, 4, 5, 6] with num_heads=2
                # Note: head1 = [1, 2, 3], head2 = [4, 5, 6]
                # the average importance is [(1+2+3)/3, (4+5+6)/3] = [2, 5]
                head_imp = imp.view(ch_groups, -1).mean(1) # average importance by head.
                global_head_importance[group] = (qkv_layers, head_imp)

        if len(global_importance) == 0 and len(global_head_importance)==0:
            return
        
        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################
        
        # Find the threshold for global finetuning
        if len(global_importance)>0:
            concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
            target_finetuning_ratio = self.finetuning_ratio
            n_freezed = len(concat_imp) - int(
                self.initial_total_channels *
                (target_finetuning_ratio)
            )
            if n_freezed>0:
                topk_imp, _ = torch.topk(concat_imp, k=n_freezed, largest=False)
                thres = topk_imp[-1]

        # Find the threshold for head finetuning
        if len(global_head_importance)>0:
            concat_head_imp = torch.cat([local_imp[-1] for local_imp in global_head_importance.values()], dim=0)
            target_head_finetuning_ratio = self.head_finetuning_ratio
            n_heads_freezed = len(concat_head_imp) - int(
                self.initial_total_heads *
                (target_head_finetuning_ratio)
            )
            if n_heads_freezed>0:
                topk_head_imp, _ = torch.topk(concat_head_imp, k=n_heads_freezed, largest=False)
                head_thres = topk_head_imp[-1]
        
        ##############################################
        # 3. set finetuned parameters
        ##############################################
        for group, ch_groups, group_size, imp in global_importance:
            module = group[0].dep.target.module
            pruning_fn = group[0].dep.handler
            get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(pruning_fn) else self.DG.get_in_channels
            
            # Prune feature dims/channels
            freezing_indices = []
            if len(global_importance)>0 and n_freezed>0:
                if ch_groups > 1: # re-compute importance for each channel group if channel grouping is enabled
                    n_freezed_per_group = len((imp <= thres).nonzero().view(-1))
                    if n_freezed_per_group>0:
                        if self.round_to:
                            n_freezed_per_group = self._round_to(n_freezed_per_group, group_size, self.round_to)
                        _is_attn, _ = self._is_attn_group(group)
                        if not _is_attn or self.finetune_head_dims==True:
                            raw_imp = self.estimate_importance(group) # re-compute importance
                            for chg in range(ch_groups): # determine pruning indices for each channel group independently
                                sub_group_imp = raw_imp[chg*group_size: (chg+1)*group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_freezing_idxs = sub_imp_argsort[:n_freezed_per_group] + chg*group_size
                                freezing_indices.append(sub_freezing_idxs)
                else:
                    _freezing_indices = (imp <= thres).nonzero().view(-1)
                    imp_argsort = torch.argsort(imp)
                    if len(_freezing_indices)>0 and self.round_to: 
                        n_freezed = len(_freezing_indices)
                        current_channels = get_channel_fn(module)
                        n_freezed = self._round_to(n_freezed, current_channels, self.round_to)
                        _freezing_indices = imp_argsort[:n_freezed]
                    freezing_indices.append(_freezing_indices)
                        
            # Prune heads
            if len(global_head_importance)>0 and n_heads_freezed>0:
                if group in global_head_importance:
                    qkv_layers, head_imp = global_head_importance[group]
                    head_freezing_indices = (head_imp <= head_thres).nonzero().view(-1)
                    if len(head_freezing_indices)>0:
                        for head_id in head_freezing_indices:
                            freezing_indices.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=head_imp.device) )
            
            if len(freezing_indices)==0: continue
            freezing_indices = torch.unique(torch.cat(freezing_indices, 0)).tolist()
            # create pruning group
            group = self.DG.get_pruning_group(
                module, pruning_fn, freezing_indices)
            if self.DG.check_pruning_group(group):
                yield group 
