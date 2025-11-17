import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import argparse
import time
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.nn import Linear
from torch.autograd import Variable
from methods.data_loader import flanv2, wikitext2, alpaca_llama, alpacagpt4_llama, gsm8k, zeroshot_llama
from evaluate import load
from methods.finetuner_runner import Finetuner
from methods.finetuner_importance import RandomImportance, NormImportance, TaylorImportance, HessianImportance, WandAImportance, ZerothOrderTaylorImportance
import methods.finetuner_function as fntn_function
from methods.finetuner_function import myLinear 

from methods.tp_MetaFinetuner import MetaFinetuner
from methods.tp_finetuner_importance import myGroupNormImportance, myGroupTaylorImportance, myGroupHessianImportance, myGroupWandAImportance
from torch.utils.tensorboard import SummaryWriter
#from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

auth_token="your_token"
hf_cache = '/your_path/huggingface_cache'

def parse_args():
    parser = argparse.ArgumentParser(description='LLM finetuning')
    parser.add_argument('--seed', default=2023, type=int, help='The random seed.')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='model name')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset choosing.')
    parser.add_argument('--task_name', type=str, default='text-generation', help='Task choosing.')
    parser.add_argument('--n_samples', type=int, default=256, help='Number of samples for finetuning text generation model.')
    parser.add_argument('--max_length', type=int, default=2048, help=(
            "The maximum length of total input sequence tokenization. Sequences longer will be truncated, sequences shorter will be padded."))
    parser.add_argument( "--pad_to_max_length", default=False, action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",)
    parser.add_argument('--data_path', default='/your_path', type=str, help='Data path.')
    parser.add_argument('--model_checkpoint', type=str, default='/your_path/sparsity_finetuning', help='The folder for storing model checkpoints.')
    parser.add_argument('--store_frequency', type=int, default=5, help='Storing model checkpoints each how much epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for model training.')
    parser.add_argument('--warmup_steps', type=float, default=0, help='Warm up steps (Learning rate will not dacay) for model training.')
    parser.add_argument('--taylor_batchs', default=50, type=int, help='Number of batchs for taylor criterion.')
    parser.add_argument('--finetune_attn_as_linear', default=False, action='store_true', help='Finetune attention layer as independent linear layers.')
    parser.add_argument('--dependency', default=False, action='store_true', help='If True, use tp_fintuner to find the dependency graph.')
    #parser.add_argument('--quantization', default=False, action='store_true', help='If True, use quantization for fine-tuning parameters.')
    parser.add_argument('--param_dtype', default="fp32", type=str, help="dtype of fine-tuning parameters")
    parser.add_argument('--finetuning_ratio', default=0.024, type=float, help='finetuning ratio')
    parser.add_argument('--p_dropout', default=0.0, type=float, help='finetuning dropout prob.')
    parser.add_argument('--n_estimate', default=5, type=int, help='n_estimate for ZOtaylor')
    parser.add_argument('--variant', default='PruFT', type=str, help='combining other method')
    parser.add_argument('--finetuning_rank', default=0, type=int, help='control the fine-tuning features by a fixed rank instead of using ratio')
    parser.add_argument('--finetune_num_heads', default=False, action='store_true', help='Finetuning attention head.')
    parser.add_argument('--head_finetuning_ratio', default=0.05, type=float, help='Head finetuning ratio, work if finetune_num_heads is True.')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='Bottleneck or uniform')
    parser.add_argument('--finetuning_type', default='l2', type=str, help='Finetuning type', choices=['random', 'ZOtaylor', 'taylor', 'l2', 'l1', 'wanda', 'hessian', '2ndtaylor'])
    parser.add_argument('--label_aggregator', default='None', type=str, help='Aggregator of labelwise importance score', choices=['max', 'mean', 'GreaterQuantileAvg', 'QuantilesAvg', 'Top5Avg', 'Top5MeanAvg'])
    parser.add_argument('--finetuning_channel', default='out', type=str, help='Finetuning channel. "in": in channel, "out": out channel, "half": half in and half out', choices=['out', 'in', 'half'])
    parser.add_argument('--global_finetuning', default=False, action='store_true', help='Global finetuning')  
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='Batch size for model training.')
    parser.add_argument('--EPOCHS', type=int, default=10, help='Max epochs for model training.')    
    parser.add_argument('--START_EPOCH', type=int, default=1, help='Start epochs for model training. (continue last training)')    
    parser.add_argument('--ratio_store', default="005", type=str, help='Save the pruned model in subfolder')
    parser.add_argument('--gpu', type=str, default='0', help='GPU using, i.e. \'0,1,2\'')     
    parser.add_argument('--parallel', default=False, action='store_true', help='paralleled computing')
    args = parser.parse_args()
    return args

def train_text_generation(model, device, data_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    # List to store negative log likelihoods, for computing perplexity
    #nlls = []
    total_num = len(data_loader.dataset)
    print("Training sample size: {}, number of batchs: {}.".format(total_num, len(data_loader)))
    for batch_idx, item in enumerate(data_loader):
        if isinstance(item, list):
            # Prepare inputs and move to device
            inputs, targets = item
            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)
            # Forward pass through the model
            lm_output = model(inputs, labels = targets)
        elif isinstance(item, dict):
            for k, v in item.items():
                item[k] = Variable(v).to(device)
            lm_output = model(**item)
            targets = item["labels"]
        loss = lm_output.loss #negative log likelihood
        optimizer.zero_grad()
        loss.backward()
        #accelerator.backward(loss)
        optimizer.step()
        sum_loss += loss.data.item()
        #sum_loss += accelerator.gather(loss).data.item()
        # Append to list of negative log likelihoods
        #nlls.append(loss)
        
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(targets), total_num,
                       100. * (batch_idx + 1) / len(data_loader), loss.item()))
        torch.cuda.empty_cache()
    avg_loss = sum_loss / len(data_loader)
    #ppl = torch.exp(torch.stack(nlls).mean())
    print('epoch:{},loss:{}'.format(epoch, avg_loss))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    return avg_loss#, ppl.item()


@torch.no_grad()
def val_text_generation(model, device, data_loader, metric):
    model.eval()
    sum_loss = 0
    # List to store negative log likelihoods, for computing perplexity
    output_metric = None
    if metric == "perplexity":
        nlls = []
    elif metric in ["F1", "accuracy", "spearmanr" ]:
        metric_fn = load(metric)
    total_num = len(data_loader.dataset)
    print("Training sample size: {}, number of batchs: {}.".format(total_num, len(data_loader)))    
    with torch.no_grad():
        for batch_idx, item in enumerate(data_loader):
            if isinstance(item, list):
                # Prepare inputs and move to device
                inputs, targets = item
                inputs = Variable(inputs).to(device)
                targets = Variable(targets).to(device) 
                # Forward pass through the model
                lm_output = model(inputs, labels = targets)
            elif isinstance(item, dict):
                for k, v in item.items():
                    item[k] = Variable(v).to(device)
                lm_output = model(**item)
                targets = item["labels"]
            loss = lm_output.loss #negative log likelihood
            #pred = out.logits
            sum_loss += loss.data.item()
            # Append to list of negative log likelihoods
            if metric == "perplexity":
                nlls.append(loss)
            elif metric in ["F1", "accuracy", "spearmanr" ]:
                metric_fn.add_batch(
                    predictions = lm_output.logits.argmax(dim=-1).to(torch.float32),
                    references = targets.to(torch.float32),
                )

        if metric == "perplexity":
            output_metric = torch.exp(torch.stack(nlls).mean()).item()
        elif metric in ["F1", "accuracy", "spearmanr" ]:
            output_metric = metric_fn.compute()[metric]

        avgloss = sum_loss / len(data_loader)
        print('\nVal set: Average loss: {:.4f}, perplexity: {:.4f}\n'.format(avgloss, output_metric))
        torch.cuda.empty_cache()
        return avgloss, output_metric

def main():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['PYHTONHASHSEED'] = str(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #np.random.seed(args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch_cache = '/your_path/torch/hub'
    if not os.path.isdir(torch_cache):
        os.makedirs(torch_cache, exist_ok=True)
    if not os.path.isdir(args.model_checkpoint):
        os.makedirs(args.model_checkpoint, exist_ok=True)
    torch.hub.set_dir(torch_cache)
    #import datasets
    #datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    #########################################
    # load importance estimator
    #########################################

    if args.dependency:
        taylorImp = myGroupTaylorImportance
        hessianImp = myGroupHessianImportance
        normImp = myGroupNormImportance
        wandaImp = myGroupWandAImportance
        ZOtaylor = ZerothOrderTaylorImportance #TODO
    else:
        taylorImp = TaylorImportance
        hessianImp = HessianImportance
        normImp = NormImportance
        wandaImp = WandAImportance
        ZOtaylor = ZerothOrderTaylorImportance
    
    if args.finetuning_type == 'random':
        imp = RandomImportance()
    elif args.finetuning_type == 'taylor':
        imp = taylorImp()
    elif args.finetuning_type == 'ZOtaylor':
        imp = ZOtaylor(n_estimate = args.n_estimate)
    elif args.finetuning_type == 'l2':
        imp = normImp(p=2)
    elif args.finetuning_type == 'l1':
        imp = normImp(p=1)
    elif args.finetuning_type == 'wanda':
        imp = wandaImp()
    elif args.finetuning_type == 'hessian':
        imp = hessianImp()
    else: raise NotImplementedError

    #########################################
    # check model list
    #########################################
    model_list = [
        "meta-llama/Meta-Llama-3-8B","meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "google/gemma-7b",
    ]
    if args.model not in model_list:
        raise ValueError("Only support the language classification models including: ".format(model_list))
    #########################################
    # load dataloader
    #########################################
    
    if args.dataset == "flanv2":
        train_loader, val_loader = flanv2(args)
        #if args.task_name == 'MMLU':
        #    metric = "accuracy" 
        #elif args.task_name == 'QA':
        #    metric = "F1" 
        #MMLU: 5 shot setting, accuracy
        #QA: 1-shot setting, F1
        #TODO: how to compute that
    elif args.dataset == "wikitext2":
        train_loader, val_loader = wikitext2(args)
        #if args.task_name == 'text-generation':
        #metric = "perplexity" 
    elif args.dataset == "alpaca":
        train_loader, val_loader = alpaca_llama(args)
        #if args.task_name == 'text-generation':
        #metric = "spearmanr" 
        #args.learning_rate = 1e-4
    elif args.dataset == "alpacagpt4":
        train_loader, val_loader = alpacagpt4_llama(args)
        #args.learning_rate = 1e-4
    elif args.dataset == "zeroshot":
        train_loader, val_loader = zeroshot_llama(args)
        #args.learning_rate = 1e-4
    elif args.dataset == "gsm8k":
        train_loader, val_loader = gsm8k(args)

    metric = "perplexity" 

    print("Dataset: {}, task: {}, metric: {}".format(args.dataset, args.task_name, metric))
    config = AutoConfig.from_pretrained(args.model, finetuning_task="text-generation", token = auth_token, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, cache_dir = hf_cache)
    #Do not need to reset the classifier, the pretrained parameters does not include classification layer.
    model.to(DEVICE)

    #########################################
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    text = "Hello, this is an example input."
    example_inputs = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(DEVICE)

    for p in model.parameters():
        p.requires_grad_(True)        

    model_name = args.model.split("/")[-1]
    print("{} to be finetuned, importance are evaluated by {}.".format(model_name, 
                                    args.finetuning_type + ("_bottleneck" if args.bottleneck else "") +\
                                    ("_global" if args.global_finetuning else "") +\
                                    (("_" + args.label_aggregator) if args.label_aggregator != "None" else "")
                                    ))
    print(model)

    #########################################
    # Build network finetuner
    #########################################
    start_time = time.time()
    num_heads = {}
    ignored_layers = []
    ignored_layers.append(model.lm_head)
    if args.dependency:
        # All heads should be pruned simultaneously, so we group channels by head.        
        from torch_pruning import ops
        for name, module in model.named_modules():
            print("module name: {}".format(name))
            if name.endswith("self_attn"):
                num_heads[module.q_proj] = model.config.num_attention_heads
                num_heads[module.k_proj] = model.config.num_key_value_heads
                num_heads[module.v_proj] = model.config.num_key_value_heads
            if not isinstance(module, Linear):
                ignored_layers.append(module)
            for finetuning_type in fntn_function.FinetunerBox.keys():
                if isinstance(module, ops.type2class(finetuning_type)):
                    print("finetuner: {}".format(fntn_function.FinetunerBox[finetuning_type]))
        output_transform = lambda out: out.logits.sum()
        finetuner = MetaFinetuner(
            model, 
            example_inputs, 
            global_finetuning=args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
            importance=imp, # importance criterion for parameter selection
            finetuning_ratio=args.finetuning_ratio,
            rank = args.finetuning_rank if args.finetuning_rank>0 else None,
            num_heads=num_heads,
            finetune_num_heads=args.finetune_num_heads, # reduce num_heads by pruning entire heads (default: False)
            finetune_head_dims=not args.finetune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
            head_finetuning_ratio=0.5, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
            output_transform=output_transform,
            round_to = 1 if args.finetuning_rank>0 else 4,
            ignored_layers=ignored_layers,
        )
    else:
        #########################################
        # Ignore classification modules
        # All heads should be pruned simultaneously, so we group channels by head.
        # Since requires_grad are set to be True, all parameters in the ignored layers will be finetuned.
        #########################################

        if args.finetune_attn_as_linear:
            print("Finetune attention layer as independent linear layers.")
        
        customized_finetuner = None
        for name, module in model.named_modules():
            if (not args.finetune_attn_as_linear) and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                ignored_layers.append(module)
            if 'norm' in name or 'embed_tokens' in name:
                ignored_layers.append(module)
        if args.finetuning_channel == "in":
            print("Finetuning in channels/features.")
        unwrapped_parameters = None
        finetuner = Finetuner(
            model, 
            global_finetuning=args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
            importance=imp, # importance criterion for parameter selection
            finetuning_ratio=args.finetuning_ratio, # target pruning ratio
            rank = args.finetuning_rank if args.finetuning_rank>0 else None,
            ignored_layers=ignored_layers,
            num_heads=num_heads, # number of heads in self attention
            finetune_num_heads=args.finetune_num_heads, # finetuning entire heads (default: False)
            finetune_head_dims=not args.finetune_num_heads, # finetuning head_dim (default: True)
            head_finetuning_ratio = args.head_finetuning_ratio, # only works when finetune_num_heads=True (default: 0.0)
            round_to = 1 if args.finetuning_rank>0 else 4,
            finetuning_channel = args.finetuning_channel,
            #dtype = "fp16" if args.quantization else args.param_dtype,
            dtype = args.param_dtype,
            p_dropout = args.p_dropout,
            variant = args.variant,
            customized_finetuner = customized_finetuner
        )
    torch.cuda.reset_peak_memory_stats(0)
    if isinstance(imp, (wandaImp, taylorImp, hessianImp)):
        if isinstance(imp, (taylorImp, hessianImp)):
            model.zero_grad()
            print("Accumulating gradients for finetuning settings...")
        if isinstance(imp, hessianImp):
            imp.zero_grad()    

        if isinstance(imp, wandaImp):        
            imp.register_module(model, len(train_loader.dataset))

        for k, item_dict in enumerate(train_loader):
            #print(item_dict)
            if k>=args.taylor_batchs: break
            for key, tensor in item_dict.items():
                item_dict[key] = tensor.to(DEVICE)

            output = model(**item_dict)

            lbls = item_dict["labels"]
            if isinstance(imp, hessianImp):
                loss = F.cross_entropy(output.logits, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model)
            elif isinstance(imp, taylorImp):
                loss = output.loss
                loss.backward()
        if isinstance(imp, wandaImp):        
            imp.remove_module(model)
        
    elif isinstance(imp, (ZOtaylor)):
        imp.projected_grad_estimate(model, DEVICE, train_loader, ignored_layers, random_seed = args.seed)
        
    finetuner.run() 
    end_time = time.time()
    print("Max memory allocated in fine-tuning settings: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
    print("Completed setting the finetuned parameters, took {:.2f}s.".format(end_time - start_time))
    for name, parameters in model.named_parameters():
        if "finetuned" not in name:
            parameters.requires_grad_(False)  
    print("Completed setting the finetuned parameters.")
    torch.cuda.empty_cache()

    optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0,
        )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
    decay_type = 'step linear decay'
    #optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    #scheduler = get_scheduler(
    #    name = "linear",
    #    optimizer=optimizer,
    #    num_warmup_steps = int(args.EPOCHS * len(train_loader) * 0.03),
    #    num_training_steps = args.EPOCHS * len(train_loader), 
    #    )
    #decay_type = 'linear decay'

    print("Optimizer settings: \nlearning rate: {}\nscheduler: {} with warmup steps: {}\nnum_training_steps: {}".format(
        args.learning_rate, decay_type, int(args.EPOCHS * len(train_loader) * 0.03), args.EPOCHS * len(train_loader)
    ))

    #finetune the model here
    print("\n----------------------------------------")
    n_trainable_params = 0
    n_params = 0
    for name, param in model.named_parameters():
        print ("Parameters: {}, requires_grad: {}, size: {}.".format(name, param.requires_grad, param.size()))
        n_params += param.numel()
        if param.requires_grad:
            n_trainable_params += param.numel()
    print ("Number of all parameters: {}.\nNumber of trainable parameters: {}.".format(n_params, n_trainable_params))
    print("----------------------------------------\n")

    if args.dependency:
        checkpoint_folder = os.path.join(args.model_checkpoint, "tp_finetuning")
    else:
        checkpoint_folder = args.model_checkpoint
    checkpoint_folder = os.path.join(os.path.join(checkpoint_folder, model_name), args.dataset + "_" + args.task_name)  
    if args.finetuning_rank>0:
        checkpoint_folder = os.path.join(checkpoint_folder, "rank" + str(args.finetuning_rank))
    else:
        checkpoint_folder = os.path.join(checkpoint_folder, "ratio" + str(args.finetuning_ratio).replace(".", ""))
        
    checkpoint_folder = os.path.join(checkpoint_folder,
                                    ("inchannel_" if args.finetuning_channel == "in" else "") +\
                                    "by_" + args.finetuning_type + ("_bottleneck" if args.bottleneck else "") +\
                                    ("_global" if args.global_finetuning else "") +\
                                    (("_" + args.label_aggregator) if args.label_aggregator != "None" else "") )
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    print("Testing perplexity of the pretrained foundation model {}".format(model_name))
    loss_ori, ppl_ori = val_text_generation(model, DEVICE, val_loader, metric)
    Best_metric = ppl_ori
    checkpoint = {'model': model.state_dict(), 'best_metric': Best_metric}
    checkpoint_name = model_name + ("_attn_as_linear" if args.finetune_attn_as_linear else "")
    checkpoint_name = checkpoint_name + ("_dropout" if args.p_dropout>0.0 else "")
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name + '_epoch_{}.pth'.format(args.START_EPOCH-1))
    if args.START_EPOCH==1:
        torch.save(checkpoint, os.path.join(checkpoint_folder, model_name + '_best.pth') )
        training_loss_history = []
        val_loss_history = []
        metric_history = []
    else:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])   
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        training_loss_history = checkpoint['training_loss']
        val_loss_history = checkpoint['val_loss']
        metric_history = checkpoint['metric']
        Best_metric = checkpoint['best_metric']

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=checkpoint_folder)

    for epoch in range(args.START_EPOCH, args.EPOCHS + 1):
        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()
        train_loss = train_text_generation(model, DEVICE, train_loader, optimizer, epoch)
        scheduler.step()
        end_time = time.time()
        # Check the maximum memory allocated
        print("Max memory allocated in training: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
        print('Training took {:.2f}s'.format(end_time - start_time))
        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()
        val_loss, val_metric = val_text_generation(model, DEVICE, val_loader, metric)
        end_time = time.time()
        # Check the maximum memory allocated
        print("Max memory allocated in validation: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
        print('Inference took {:.2f}s'.format(end_time - start_time))
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar(metric + "/val", val_metric, epoch)
        
        training_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        metric_history.append(val_metric)
        #accelerator.wait_for_everyone()
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'training_loss': training_loss_history,
            'val_loss': val_loss_history,
            'metric': metric_history,
            'best_metric': Best_metric,
            }
        if (metric == "perplexity" and val_metric < Best_metric) or (metric in ["F1", "accuracy", "spearmanr" ] and val_metric > Best_metric):
            checkpoint['best_metric'] = val_metric
            torch.save(checkpoint, os.path.join(checkpoint_folder, checkpoint_name + '_best.pth') )
            Best_metric = val_metric
        print('Best metric: {}\n'.format( Best_metric))
        if epoch % args.store_frequency == 0:
            torch.save(checkpoint, os.path.join(checkpoint_folder, checkpoint_name + '_epoch_{}.pth'.format(epoch)) )
    #accelerator.end_training()
    for name, layer in model.named_modules():
        if isinstance(layer, myLinear): 
            path = name.split('.')
            module = model
            if len(path)>1:
                for p in path[:-1]:
                    module = getattr(module, p)
            setattr(module, path[-1], layer.merge_and_unload())  
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, checkpoint_name + '_final_merged.pth') )
    writer.close()

    #=============================================================================================

    print("Testing perplexity of the finel finetuned model...")
    loss_Finetuned, metric_Finetuned = val_text_generation(model, DEVICE, val_loader, metric)
    print('Best perplexity: {}\n'.format( Best_metric))

    print("----------------------------------------")
    print("Now evaluating the model by EleutherAI's LM-Evaluation-Harness")
    print("Note that we only use this for gsm8k in the paper.")
    from lm_eval import evaluator 
    from lm_eval.models.huggingface import HFLM 
    num_fewshot = 0
    if "70b" in args.model or "65b" in args.model:
        limit = 2000
    else:
        limit = None

    if args.dataset == "zeroshot":
        task_list=["boolq","piqa","social_iqa","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]
    elif args.dataset == "gsm8k":
        #task_list = ["gsm8k", "hendrycks_math"] 
        task_list = ["gsm8k"] 
    elif args.dataset == "coding":
        task_list=["code2text_python",]
    else:
        #task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa", "ifeval"]
        task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]

    model = HFLM(pretrained=model, tokenizer = tokenizer, backend = "causal", trust_remote_code = True)
    results = evaluator.simple_evaluate(
        model = model,
        tasks = task_list,
        num_fewshot=num_fewshot,
        batch_size=args.BATCH_SIZE,
        device = DEVICE,
        limit = limit,
        check_integrity = False,
    )
    printing_results = results["results"]
    print("{} shot evaluation".format(num_fewshot))
    for k,v in printing_results.items():
        print("{}: {}".format(k,v))

if __name__=='__main__':
    main()