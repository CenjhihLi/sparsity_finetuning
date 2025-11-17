import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from methods.data_loader import glue
from evaluate import load
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
    parser.add_argument('--train_attnqv_only', default=False, action='store_true', help='If True, train classifier only')
    parser.add_argument('--max_length', type=int, default=256, help=(
            "The maximum length of total input sequence tokenization. Sequences longer will be truncated, sequences shorter will be padded."))
    parser.add_argument('--pad_to_max_length', default=False, action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",)
    parser.add_argument('--data_path', default='/your_path', type=str, help='Data path')
    parser.add_argument('--model_checkpoint', type=str, default='/your_path', help='The folder for storing model checkpoints.')
    parser.add_argument('--store_frequency', type=int, default=5, help='Storing model checkpoints each how much epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for model training.')
    parser.add_argument('--warmup_steps', type=float, default=0, help='Warm up steps (Learning rate will not dacay) for model training.')
    parser.add_argument('--train_classifier_only', default=False, action='store_true', help='If True, train classifier only')
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size for model training.')
    parser.add_argument('--EPOCHS', type=int, default=10, help='Max epochs for model training.')    
    parser.add_argument('--START_EPOCH', type=int, default=1, help='Start epochs for model training. (continue last training)')    
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
    #accelerator = Accelerator()
    metric_name = task2metrc[args.task_name]

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
    print("Dataset: {}, task: {}".format(args.dataset, args.task_name))
    config = AutoConfig.from_pretrained(args.model, num_labels=n_labels, finetuning_task = args.task_name, cache_dir = hf_cache)
    #tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir = hf_cache)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, cache_dir = hf_cache)
    #Do not need to reset the classifier, the pretrained parameters does not include classification layer.
    model.to(DEVICE)

    #########################################
    if args.train_classifier_only:
        print("Training the classifier only.")
        for name, param in model.named_parameters():
            if "classifier" in name or "pooler" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    elif args.train_attnqv_only:
        for n, m in model.named_modules():
            if "query" in n or "value" in n:
                for p in m.parameters():
                    p.requires_grad_(True)
            else:
                for p in m.parameters():
                    p.requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    model_name = args.model.split("/")[-1]
    print("{} to be finetuned.".format(model_name))
    print(model)

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

    checkpoint_folder = os.path.join(os.path.join(args.model_checkpoint, model_name if not args.train_attnqv_only else model_name + "_attnqv"), 
                                     args.dataset + "_" + args.task_name)  
    
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