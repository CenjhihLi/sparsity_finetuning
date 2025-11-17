import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import transformers
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp

from torch.nn import Linear
from methods.data_loader import glue
from evaluate import load
from methods.finetuner_runner import Finetuner
from methods.finetuner_importance import NormImportance, TaylorImportance, HessianImportance, WandAImportance
import methods.finetuner_function as fntn_function

from methods.tp_MetaFinetuner import MetaFinetuner
from methods.tp_finetuner_importance import myGroupNormImportance, myGroupTaylorImportance, myGroupHessianImportance, myGroupWandAImportance
from torch.utils.tensorboard import SummaryWriter
#from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

task2metrc = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mnli_mismatched": "accuracy", 
    "mnli_matched": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
    "hans": "accuracy",
}

hf_cache = '/your_path/huggingface_cache'

def parse_args():
    parser = argparse.ArgumentParser(description='Timm ViT Pruning')
    parser.add_argument('--seed', default=2023, type=int, help='The random seed.')
    parser.add_argument('--model', default='microsoft/deberta-v3-base', type=str, help='model name')
    parser.add_argument('--dataset', type=str, default='glue', help='Dataset choosing.')
    parser.add_argument('--task_name', type=str, default='mnli', help='Task choosing.')
    parser.add_argument('--max_length', type=int, default=256, help=(
            "The maximum length of total input sequence tokenization. Sequences longer will be truncated, sequences shorter will be padded."))
    parser.add_argument( "--pad_to_max_length", default=False, action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",)
    parser.add_argument('--data_path', default='/your_path', type=str, help='Data path')
    parser.add_argument('--model_checkpoint', type=str, default='/your_path/sparsity_finetuning', help='The folder for storing model checkpoints.')
    parser.add_argument('--store_frequency', type=int, default=5, help='Storing model checkpoints each how much epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for model training.')
    parser.add_argument('--warmup_steps', type=float, default=0, help='Warm up steps (Learning rate will not dacay) for model training.')
    parser.add_argument('--taylor_batchs', default=50, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--finetune_attn_as_linear', default=False, action='store_true', help='Finetune attention layer as independent linear layers.')
    parser.add_argument('--dependency', default=False, action='store_true', help='If True, use tp_fintuner to find the dependency graph.')
    parser.add_argument('--finetuning_ratio', default=0.05, type=float, help='finetune ratio')
    parser.add_argument('--finetune_num_heads', default=False, action='store_true', help='global finetuning')
    parser.add_argument('--head_finetuning_ratio', default=0.05, type=float, help='head finetuning ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--finetuning_type', default='taylor', type=str, help='finetuning type', choices=['random', 'taylor', 'l2', 'l1', 'wanda', 'hessian', '2ndtaylor'])
    parser.add_argument('--label_aggregator', default='None', type=str, help='aggregator of labelwise importance score', choices=['max', 'mean', 'GreaterQuantileAvg', 'QuantilesAvg', 'Top5Avg', 'Top5MeanAvg'])
    parser.add_argument('--finetuning_channel', default='out', type=str, help='finetuning channel. "in": in channel, "out": out channel, "half": half in and half out', choices=['out', 'in', 'half'])
    parser.add_argument('--global_finetuning', default=False, action='store_true', help='global finetuning')
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size for model training.')
    parser.add_argument('--EPOCHS', type=int, default=15, help='Max epochs for model training.')    
    parser.add_argument('--START_EPOCH', type=int, default=1, help='Start epochs for model training. (continue last training)')    
    parser.add_argument('--ratio_store', default="005", type=str, help='save the finetuned model in subfolder')
    parser.add_argument('--gpu', type=str, default='0', help='GPU using, i.e. \'0,1,2\'')     
    parser.add_argument('--parallel', default=False, action='store_true', help='paralleled computing')
    args = parser.parse_args()
    return args

def batch_train(model, device, data_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(data_loader.dataset)
    print(total_num, len(data_loader))
    for batch_idx, item_dict in enumerate(data_loader):
        for key, tensor in item_dict.items():
            item_dict[key] = tensor.to(device)
        out = model(**item_dict)
        target = item_dict["labels"]            
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(target), total_num,
                       100. * (batch_idx + 1) / len(data_loader), loss.item()))
    avg_loss = sum_loss / len(data_loader)
    print('epoch:{},loss:{}'.format(epoch, avg_loss))
    return avg_loss

@torch.no_grad()
def val(model, device, data_loader, metric, is_regression: bool = False):
    model.eval()
    sum_loss = 0
    total_num = len(data_loader.dataset)
    print(total_num, len(data_loader))
    with torch.no_grad():
        for batch_idx, item_dict in enumerate(data_loader):
            for key, tensor in item_dict.items():
                item_dict[key] = tensor.to(device)
            out = model(**item_dict)
            target = item_dict["labels"]   
            loss = out.loss
            if is_regression:
                pred = out.logits
            else:
                pred = out.logits.argmax(dim=-1)
            sum_loss += loss.data.item()
            metric.add_batch(
                predictions = pred,
                references = target,
            )

        eval_metric = metric.compute()
        avgloss = sum_loss / len(data_loader)
        
        print('\nVal set: Average loss: {:.4f}, metric: {}\n'.format(
            avgloss, eval_metric))
        return avgloss, eval_metric


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
    metric_name = task2metrc[args.task_name]

    #########################################
    # load importance estimator
    #########################################

    if args.dependency:
        taylorImp = myGroupTaylorImportance
        hessianImp = myGroupHessianImportance
        normImp = myGroupNormImportance
        wandaImp = myGroupWandAImportance
    else:
        taylorImp = TaylorImportance
        hessianImp = HessianImportance
        normImp = NormImportance
        wandaImp = WandAImportance
    
    if args.finetuning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.finetuning_type == 'taylor':
        imp = taylorImp()
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
        "google-bert/bert-large-uncased", "google-bert/bert-large-cased", 
        "google-bert/bert-base-uncased", "google-bert/bert-base-cased", 
        "microsoft/deberta-base", "microsoft/deberta-large",  
        "microsoft/deberta-v3-base", "microsoft/deberta-v3-large", 
    ]
    if args.model not in model_list:
        raise ValueError("Only support the language classification models including: ".format(model_list))
    #########################################
    # load dataloader
    #########################################
    
    if args.dataset == "glue":
        train_loader, val_loader = glue(args)
    if args.task_name == "stsb":
        n_labels = 1
    else:
        label_list = train_loader.dataset.unique("labels")
        label_list.sort()
        n_labels = len(label_list)
    n_sample = len(train_loader.dataset)

    print("Dataset: {}, task: {}".format(args.dataset, args.task_name))
    config = AutoConfig.from_pretrained(args.model, num_labels=n_labels, finetuning_task=args.task_name, cache_dir = hf_cache)
    #tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir = hf_cache)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, cache_dir = hf_cache)
    #Do not need to reset the classifier, the pretrained parameters does not include classification layer.
    model.to(DEVICE)

    for p in model.parameters():
        p.requires_grad_(True)
    model_name = args.model.split("/")[-1]
    print("{} to be finetuned, importance are evaluated by {}.".format(model_name, 
                                    args.finetuning_type + ("_bottleneck" if args.bottleneck else "") +\
                                    ("_global" if args.global_finetuning else "") +\
                                    (("_" + args.label_aggregator) if args.label_aggregator != "None" else "") +\
                                    ("_finetune_attn_as_linears" if args.finetune_attn_as_linear else "") 
                                    ))
    print(model)

    #########################################
    # Build network finetuner
    #########################################
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    hf_inputs = tokenizer("Hello, this is an example input.", return_tensors="pt")
    example_inputs = {
        'input_ids': hf_inputs['input_ids'].to(DEVICE), 
        'token_type_ids': hf_inputs['token_type_ids'].to(DEVICE), 
        'attention_mask': hf_inputs['attention_mask'].to(DEVICE),
        }

    bert_attn = (
        transformers.models.bert.modeling_bert.BertSelfAttention, 
        transformers.models.deberta_v2.modeling_deberta_v2.DisentangledSelfAttention
        )
    num_heads = {}
    ignored_layers = []
    ignored_layers.append(model.pooler)
    if args.dependency:
        # All heads should be pruned simultaneously, so we group channels by head.
        for m in model.modules():
            if isinstance(m, transformers.models.bert.modeling_bert.BertSelfAttention):
                num_heads[m.query] = m.num_attention_heads
                num_heads[m.key] = m.num_attention_heads
                num_heads[m.value] = m.num_attention_heads
            if isinstance(m, transformers.models.deberta_v2.modeling_deberta_v2.DisentangledSelfAttention):
                num_heads[m.query_proj] = m.num_attention_heads
                num_heads[m.key_proj] = m.num_attention_heads
                num_heads[m.value_proj] = m.num_attention_heads
            if isinstance(m, Linear) and m.out_features == n_labels:
                ignored_layers.append(m)
        output_transform = lambda out: out.logits.sum()
        finetuner = MetaFinetuner(
            model, 
            example_inputs, 
            global_finetuning=args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
            importance=imp, # importance criterion for parameter selection
            finetuning_ratio=(args.finetuning_ratio/2),
            num_heads=num_heads,
            finetune_num_heads=args.finetune_num_heads, # reduce num_heads by pruning entire heads (default: False)
            finetune_head_dims=not args.finetune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
            head_finetuning_ratio=0.5, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
            output_transform=output_transform,
            ignored_layers=ignored_layers,
        )
    else:
        #########################################
        # Ignore classification modules
        # All heads should be pruned simultaneously, so we group channels by head.
        # Since requires_grad are set to be True, all parameters in the ignored layers will be finetuned.
        #########################################
        for m in model.modules():
            if isinstance(m, bert_attn):
                num_heads[m] = m.num_attention_heads
            if isinstance(m, Linear) and m.out_features == n_labels:
                ignored_layers.append(m)

        if args.finetune_attn_as_linear:
            customized_finetuner = None
            print("Finetune attention layer as independent linear layers.")
        else:
            customized_finetuner = fntn_function.myFinetuner
        if args.finetuning_channel == "in":
            print("Finetuning in channels/features.")
        unwrapped_parameters = None
        finetuner = Finetuner(
            model, 
            global_finetuning=args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
            importance=imp, # importance criterion for parameter selection
            finetuning_ratio=args.finetuning_ratio, # target pruning ratio
            ignored_layers=ignored_layers,
            num_heads=num_heads, # number of heads in self attention
            finetune_num_heads=args.finetune_num_heads, # finetuning entire heads (default: False)
            finetune_head_dims=not args.finetune_num_heads, # finetuning head_dim (default: True)
            head_finetuning_ratio = args.head_finetuning_ratio, # only works when finetune_num_heads=True (default: 0.0)
            round_to=2,
            finetuning_channel = args.finetuning_channel,
            customized_finetuner = customized_finetuner,
            quantization = False,
        )
    
    if isinstance(imp, (wandaImp, taylorImp, hessianImp)):
        model.zero_grad()
        if isinstance(imp, hessianImp):
            imp.zero_grad()
        elif isinstance(imp, wandaImp):
            imp.register_wanda_hook(model, n_sample)
        print("Accumulating gradients for finetuning...")
        for k, item_dict in enumerate(train_loader):
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
            elif isinstance(imp, wandaImp):
                imp.remove_hooks()
            
    finetuner.run()

    print("Completed setting the finetuned parameters.")

    # Optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": 0.01,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    #model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    scheduler = get_scheduler(
        #name = "cosine_with_min_lr",
        name = "linear",
        optimizer=optimizer,
        num_warmup_steps = args.warmup_steps,
        num_training_steps = args.EPOCHS * len(train_loader), 
        #scheduler_specific_kwargs = {"min_lr": 1e-9},
    )
    print("num_training_steps: {}".format(args.EPOCHS * len(train_loader)))
    metric = load(args.dataset, args.task_name)


    #finetune the model here
    n_trainable_params = 0
    n_params = 0
    for name, param in model.named_parameters():
        print ("Parameters: {}, requires_grad: {}, size: {}.".format(name, param.requires_grad, param.size()))
        n_params += param.numel()
        if param.requires_grad:
            n_trainable_params += param.numel()
    print ("Number of all parameters: {}.\nNumber of trainable parameters: {}.".format(n_params, n_trainable_params))

    if args.dependency:
        checkpoint_folder = os.path.join(args.model_checkpoint, "tp_finetuning")
    else:
        checkpoint_folder = args.model_checkpoint
    checkpoint_folder = os.path.join(os.path.join(checkpoint_folder, model_name), args.dataset + "_" + args.task_name)  
    checkpoint_folder = os.path.join(os.path.join(checkpoint_folder, "ratio" + args.ratio_store), 
                                    ("inchannel_" if args.finetuning_channel == "in" else "") +\
                                    "by_" + args.finetuning_type + ("_bottleneck" if args.bottleneck else "") +\
                                    ("_global" if args.global_finetuning else "") +\
                                    (("_" + args.label_aggregator) if args.label_aggregator != "None" else "") +\
                                    ("_finetune_attn_as_linears" if args.finetune_attn_as_linear else "") 
                                    )
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    print("Testing accuracy of the pretrained foundation model {}".format(model_name))
    loss_ori, metric_ori = val(model, DEVICE, val_loader, metric, args.task_name == "stsb")
    Best_metric = metric_ori
    checkpoint = {'model': model.state_dict(), 'best_metric': Best_metric}
    
    checkpoint_file = os.path.join(checkpoint_folder, model_name + '_epoch_{}.pth'.format(args.START_EPOCH-1))
    if args.START_EPOCH==1:
        torch.save(checkpoint, os.path.join(checkpoint_folder, model_name + '_best.pth') )
        training_loss_history = []
        val_loss_history = []
        accuracy_history = []
    else:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])   
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        training_loss_history = checkpoint['training_loss']
        val_loss_history = checkpoint['val_loss']
        accuracy_history = checkpoint['accuracy']
        Best_metric = checkpoint['best_metric']

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=checkpoint_folder)

    for epoch in range(args.START_EPOCH, args.EPOCHS + 1):
        train_loss = batch_train(model, DEVICE, train_loader, optimizer, epoch)
        scheduler.step()
        val_loss, val_metric = val(model, DEVICE, val_loader, metric, args.task_name == "stsb")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_metric[metric_name], epoch)
        
        training_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        accuracy_history.append(val_metric[metric_name])
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'training_loss': training_loss_history,
            'val_loss': val_loss_history,
            'accuracy': accuracy_history,
            'best_metric': Best_metric,
            }
        if val_metric[metric_name] > Best_metric[metric_name]:
            checkpoint['best_metric'] = val_metric
            torch.save(checkpoint, os.path.join(checkpoint_folder, model_name + '_best.pth') )
            Best_metric = val_metric
        print('Best metric: {}\n'.format( Best_metric))
        if epoch % args.store_frequency == 0:
            torch.save(checkpoint, os.path.join(checkpoint_folder, model_name + '_epoch_{}.pth'.format(epoch)) )
    torch.save(checkpoint, os.path.join(checkpoint_folder, model_name + '_final.pth') )
    writer.close()

    #=============================================================================================

    print("Testing accuracy of the finel finetuned model...")
    loss_Finetuned, metric_Finetuned = val(model, DEVICE, val_loader, metric, args.task_name == "stsb")
    print('Best metric: {}\n'.format( Best_metric))

    print("----------------------------------------")

if __name__=='__main__':
    main()