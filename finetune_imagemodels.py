import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import timm
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
import numpy as np
import pandas as pd

from torch.nn import CrossEntropyLoss, Linear
from torch.autograd import Variable
from torchvision import datasets
from methods.finetuner_runner import Finetuner
from methods.finetuner_importance import RandomImportance,NormImportance, TaylorImportance, HessianImportance, WandAImportance
from methods.tp_MetaFinetuner import MetaFinetuner
from methods.tp_finetuner_importance import myGroupNormImportance, myGroupTaylorImportance, myGroupHessianImportance, myGroupWandAImportance
from methods.finetuner_function import myLinear 
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Timm ViT Pruning')
    parser.add_argument('--seed', default=2023, type=int, help='The random seed.')
    parser.add_argument('--model', default='regnetx_160', type=str, help='model name')
    parser.add_argument('--dataset', type=str, default='tiny-imagenet', help='Dataset choosing.')
    parser.add_argument('--data_path', default='/your_path', type=str, help='Data path')
    parser.add_argument('--model_checkpoint', type=str, default='/your_path/sparsity_finetuning', help='The folder for storing model checkpoints.')
    parser.add_argument('--store_frequency', type=int, default=5, help='Storing model checkpoints each how much epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for model training.')
    parser.add_argument('--taylor_batchs', default=50, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--dependency', default=False, action='store_true', help='If True, use tp_fintuner to find the dependency graph.')
    parser.add_argument('--freeze_attn', default=False, action='store_true', help='If True, not training attention blocks.')
    parser.add_argument('--finetuning_ratio', default=0.05, type=float, help='prune ratio')
    parser.add_argument('--finetune_num_heads', default=False, action='store_true', help='global pruning')
    parser.add_argument('--head_finetuning_ratio', default=0.05, type=float, help='head pruning ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--finetuning_type', default='taylor', type=str, help='pruning type', choices=['random', 'taylor', 'l2', 'l1', 'wanda', 'hessian', '2ndtaylor'])
    parser.add_argument('--label_aggregator', default='None', type=str, help='aggregator of labelwise importance score', choices=['max', 'mean', 'GreaterQuantileAvg', 'QuantilesAvg', 'Top5Avg', 'Top5MeanAvg'])
    parser.add_argument('--finetuning_channel', default='out', type=str, help='pruning channel. "in": in channel, "out": out channel, "half": half in and half out', choices=['out', 'in', 'half'])
    parser.add_argument('--global_finetuning', default=False, action='store_true', help='global pruning')
    parser.add_argument('--use_imagenet_mean_std', default=False, action='store_true', help='use imagenet mean and std')
    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='Batch size for model training.')
    parser.add_argument('--EPOCHS', type=int, default=5, help='Max epochs for model training.')    
    parser.add_argument('--START_EPOCH', type=int, default=1, help='Start epochs for model training. (continue last training)')    
    parser.add_argument('--ratio_store', default="005", type=str, help='save the pruned model in subfolder')
    parser.add_argument('--gpu', type=str, default='0', help='GPU using, i.e. \'0,1,2\'')     
    parser.add_argument('--parallel', default=False, action='store_true', help='paralleled computing')
    args = parser.parse_args()
    return args

def batch_train(model, device, data_loader, optimizer, epoch, loss_function):
    model.train()
    sum_loss = 0
    total_num = len(data_loader.dataset)
    print("n_sample: {}, n_batch: {}".format(total_num, len(data_loader)))
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        out = model(data)
        if not isinstance(out, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            out, out_kd = out
        loss = loss_function(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), total_num,
                       100. * (batch_idx + 1) / len(data_loader), loss.item()))
    avg_loss = sum_loss / len(data_loader)
    print('epoch:{},loss:{}'.format(epoch, avg_loss))
    return avg_loss

@torch.no_grad()
def val(model, device, data_loader, loss_function):
    model.eval()
    sum_loss = 0
    correct = 0
    total_num = len(data_loader.dataset)
    print("n_sample: {}, n_batch: {}".format(total_num, len(data_loader)))
    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            out = model(data)
            loss = loss_function(out, target)
            _, pred = torch.max(out.data, 1)
            correct += torch.sum(pred == target)
            sum_loss += loss.data.item()
        correct = correct.data.item()
        val_acc = correct / total_num
        avgloss = sum_loss / len(data_loader)
        
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            avgloss, correct, total_num, 100 * val_acc))
        return avgloss, val_acc

@torch.no_grad()
def Evaluate(model, device, data_loader, loss_function, n_labels):
    model.eval()
    sum_loss = 0
    total_correct = 0
    total_num = len(data_loader.dataset)
    print(total_num, len(data_loader))
    with torch.no_grad():
        correct = torch.zeros([n_labels], device = device)
        num = torch.zeros([n_labels], device = device)
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data).to(device), Variable(target).to(device)
            out = model(data)
            loss = loss_function(out, target)
            _, pred = torch.max(out.data, 1)
            total_correct += torch.sum(pred == target)
            label, count = target[pred == target].unique(return_counts = True)
            if label.shape[0]>0:
                correct[label] += count
            label, count = target.unique(return_counts = True)
            num[label] += count
            sum_loss += loss.data.item()
        total_correct = total_correct.data.item()
        acc = correct / num
        total_acc = total_correct / total_num
        avgloss = sum_loss / len(data_loader)

        df = pd.DataFrame(index=range(n_labels))
        df.index.name = "Label"
        df["Total_numbers"] = num.cpu()
        df["correct"] = correct.cpu()
        df["accuracy"] = acc.cpu()

        return avgloss, df, (total_acc, total_correct, total_num)
    
def main():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['PYHTONHASHSEED'] = str(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch_cache = '/your_path/torch/hub'
    if not os.path.isdir(torch_cache):
        os.makedirs(torch_cache, exist_ok=True)
    if not os.path.isdir(args.model_checkpoint):
        os.makedirs(args.model_checkpoint, exist_ok=True)
    torch.hub.set_dir(torch_cache)

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
        imp = RandomImportance()
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

    if args.dataset == 'tiny-imagenet':
        from methods.data_loader import tiny_imagenet
        print("Loading tiny imagenet...")
        train_loader, val_loader = tiny_imagenet(args)
        n_labels = 200
    elif args.dataset == 'imagenet-1k':
        from methods.data_loader import imagenet_1k, imagenet_transform
        print("Loading imagenet-1k...")
        path = args.data_path + "/imagenet/ILSVRC/prepared"
        dataset_train = datasets.ImageFolder(os.path.join(path, 'train'), transform=imagenet_transform)
        dataset_val = datasets.ImageFolder(os.path.join(path, 'val'), transform=imagenet_transform)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size = args.BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size = args.BATCH_SIZE, shuffle=False)
        n_labels = 1000
    elif args.dataset == 'cifar100':
        from methods.data_loader import cifar100
        print("Loading cifar100...")
        train_loader, val_loader = cifar100(args)
        n_labels = 100
    elif args.dataset == 'caltech101':
        from methods.data_loader import caltech101
        print("Loading caltech101...")
        train_loader, val_loader = caltech101(args)
        n_labels = 102

    n_sample = len(train_loader.dataset)

    model_list = ['regnetx_160','resnet101','resnext101'
                  
                  'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
                  'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
                  'deit_base_distilled_patch16_224', 'deit_base_patch16_384', 'deit_base_distilled_patch16_384', 
                  'SwinTransformerV2Cr', 'SwinTransformerV2',
                  ]
    model_list = ['regnetx_160','resnet101','resnext101',
                  'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
                  'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
                  'deit_base_distilled_patch16_224', 'deit_base_patch16_384', 'deit_base_distilled_patch16_384', 
                  'SwinTransformerV2Cr', 'SwinTransformerV2',
                  ]
    if args.model not in model_list:
        raise ValueError("Only support the models including: ".format(model_list))
    elif args.model=='regnetx_160':
        from timm.models import regnetx_160
        model = regnetx_160(pretrained=True)
        model.reset_classifier(num_classes = n_labels)
        target_model = 'cnn'
    elif args.model=='resnet101':
        from torchvision.models import resnet101
        model = resnet101(pretrained=True)
        num_features = model.fc.in_features
        model.fc = Linear(num_features, n_labels)
        target_model = 'cnn'
    elif args.model=='resnext101':
        from torchvision.models import resnext101_32x8d
        model = resnext101_32x8d(pretrained=True)
        num_features = model.fc.in_features
        model.fc = Linear(num_features, n_labels)
        target_model = 'cnn'
    elif args.model=='deit_tiny_patch16_224':
        from timm.models import deit_tiny_patch16_224
        model = deit_tiny_patch16_224(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=='deit_small_patch16_224':
        from timm.models import deit_small_patch16_224
        model = deit_small_patch16_224(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=='deit_base_patch16_224':
        from timm.models import deit_base_patch16_224
        model = deit_base_patch16_224(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=='deit_tiny_distilled_patch16_224':
        from timm.models import deit_tiny_distilled_patch16_224
        model = deit_tiny_distilled_patch16_224(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=='deit_small_distilled_patch16_224':
        from timm.models import deit_small_distilled_patch16_224
        model = deit_small_distilled_patch16_224(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=='deit_base_distilled_patch16_224':
        from timm.models import deit_base_distilled_patch16_224
        model = deit_base_distilled_patch16_224(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=='deit_base_patch16_384':
        from timm.models import deit_base_patch16_384
        model = deit_base_patch16_384(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=='deit_base_distilled_patch16_384':
        from timm.models import deit_base_distilled_patch16_384
        model = deit_base_distilled_patch16_384(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=="SwinTransformerV2Cr":
        from timm.models import SwinTransformerV2Cr
        model = SwinTransformerV2Cr(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'
    elif args.model=="SwinTransformerV2":
        from timm.models import SwinTransformerV2
        model = SwinTransformerV2(pretrained=True)
        model.reset_classifier(n_labels)
        target_model = 'vit'

    model.to(DEVICE)

    input_size = [3, 224, 224]
    example_inputs = torch.randn(1, *input_size).to(DEVICE)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    loss_function = CrossEntropyLoss()

    for p in model.parameters():
        p.requires_grad_(True)
    #########################################
    # Ignore classification modules
    # Since requires_grad are set to be True, all parameters in the ignored layers will be finetuned.
    #########################################
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, Linear) and m.out_features == n_labels:
            ignored_layers.append(m)

    print("{} to be finetuned, importance are evaluated by {}.".format(args.model, 
                                    args.finetuning_type + ("_bottleneck" if args.bottleneck else "") +\
                                    ("_global" if args.global_finetuning else "") +\
                                    (("_" + args.label_aggregator) if args.label_aggregator != "None" else "") +\
                                    ("_freeze_attn" if args.freeze_attn else "")
                                    ))

    #########################################
    # Build network finetuner
    #########################################
    if args.finetuning_channel == "in":
        print("Finetuning in channels/features.")
    unwrapped_parameters = None

    if args.dependency:
        if target_model == 'cnn':
            channel_groups = {}
            finetuner = MetaFinetuner(
                model, 
                example_inputs = example_inputs,
                global_finetuning = args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
                importance = imp, # importance criterion for parameter selection
                finetuning_ratio = (args.finetuning_ratio/2), 
                # Dependency will increase the fine-tuning ratio by at least 1 times of original ratio
                ignored_layers = ignored_layers,
                round_to = 2,
                unwrapped_parameters = unwrapped_parameters,
                out_channel_groups = channel_groups,
                )
        elif target_model == 'vit':
            num_heads = {}
            for m in model.modules():
                if isinstance(m, timm.models.vision_transformer.Attention):
                    if args.freeze_attn:
                        ignored_layers.append(m)
                    else:
                        num_heads[m.qkv] = m.num_heads 
                if args.bottleneck and isinstance(m, timm.models.vision_transformer.Mlp): 
                    ignored_layers.append(m.fc2) # only prune the internal layers of FFN & Attention    
            finetuner = MetaFinetuner(
                model, 
                example_inputs, 
                global_finetuning=args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                finetuning_ratio=(args.finetuning_ratio/2), 
                # Dependency will increase the fine-tuning ratio by at least 1 times of original ratio
                ignored_layers=ignored_layers,
                num_heads=num_heads, # number of heads in self attention
                finetune_num_heads=args.finetune_num_heads, # reduce num_heads by pruning entire heads (default: False)
                finetune_head_dims=not args.finetune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
                head_finetuning_ratio=0.5, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
                round_to=2
            )
    else:
        if target_model == 'cnn':
            channel_groups = {}
            finetuner = Finetuner(
                model, 
                global_finetuning = args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
                importance = imp, # importance criterion for parameter selection
                finetuning_ratio = args.finetuning_ratio, # target pruning ratio
                ignored_layers = ignored_layers,
                round_to = 2,
                finetuning_channel = args.finetuning_channel,
                #unwrapped_parameters = unwrapped_parameters, #finetuning dim for pos_embed, cls_token, dis_token, etc in transformers
                channel_groups = channel_groups,
                quantization = False,
                )
        elif target_model == 'vit':
            num_heads = {}
            for m in model.modules():
                if isinstance(m, timm.models.vision_transformer.Attention):
                    if args.freeze_attn:
                        ignored_layers.append(m)
                    else:
                        num_heads[m.qkv] = m.num_heads 
                if args.bottleneck and isinstance(m, timm.models.vision_transformer.Mlp): 
                    ignored_layers.append(m.fc2) # only prune the internal layers of FFN & Attention    
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
                quantization = False,
            )

    print(model)

    if isinstance(imp, (wandaImp, taylorImp, hessianImp)):
        model.zero_grad()
        if isinstance(imp, hessianImp):
            imp.zero_grad()
        elif isinstance(imp, wandaImp):
            imp.register_wanda_hook(model, n_sample)
        print("Accumulating gradients for finetuning...")
        for k, (imgs, lbls) in enumerate(train_loader):
            if k>=args.taylor_batchs: break
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)
            output = model(imgs)
            if isinstance(imp, hessianImp):
                loss = F.cross_entropy(output, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model)
            elif isinstance(imp, taylorImp):
                loss = F.cross_entropy(output, lbls)
                loss.backward()
            elif isinstance(imp, wandaImp):
                imp.remove_hooks()

    finetuner.run()
    if target_model == 'vit':
        for name, parameters in model.named_parameters():
            if "finetuned" not in name and "attn" in name:
                parameters.requires_grad_(False)  
    print("Completed setting the finetuned parameters.")
    
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
    checkpoint_folder = os.path.join(os.path.join(checkpoint_folder, args.model), args.dataset)  
    checkpoint_folder = os.path.join(os.path.join(checkpoint_folder, "ratio" + args.ratio_store), 
                                    ("inchannel_" if args.finetuning_channel == "in" else "") +\
                                    "by_" + args.finetuning_type + ("_bottleneck" if args.bottleneck else "") +\
                                    ("_global" if args.global_finetuning else "") +\
                                    (("_" + args.label_aggregator) if args.label_aggregator != "None" else "")  +\
                                    ("_freeze_attn" if args.freeze_attn else ""))
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    print("Testing accuracy of the pretrained foundation model {}".format(args.model))
    loss_ori, acc_ori = val(model, DEVICE, val_loader, loss_function)
    Best_ACC = acc_ori
    checkpoint = {'model': model.state_dict(), 'best_accuracy': Best_ACC}      

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.EPOCHS, eta_min=1e-9)
    
    checkpoint_file = os.path.join(checkpoint_folder, args.model + '_epoch_{}.pth'.format(args.START_EPOCH-1))
    if args.START_EPOCH==1:
        torch.save(checkpoint, os.path.join(checkpoint_folder, args.model + '_best.pth') )
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
        Best_ACC = checkpoint['best_accuracy']

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=checkpoint_folder)

    for epoch in range(args.START_EPOCH, args.EPOCHS + 1):
        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats()

        train_loss = batch_train(model, DEVICE, train_loader, optimizer, epoch, loss_function)
        scheduler.step()
        # Check the maximum memory allocated
        print("Max memory allocated in training: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
        
        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats()

        val_loss, val_acc = val(model, DEVICE, val_loader, loss_function)
        # Check the maximum memory allocated
        print("Max memory allocated in validation: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        training_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        accuracy_history.append(val_acc)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'training_loss': training_loss_history,
            'val_loss': val_loss_history,
            'accuracy': accuracy_history,
            'best_accuracy': Best_ACC,
            }
        if val_acc > Best_ACC:
            checkpoint['best_accuracy'] = val_acc
            torch.save(checkpoint, os.path.join(checkpoint_folder, args.model + '_best.pth') )
            Best_ACC = val_acc
        print('Best accuracy: {:.2f}%\n'.format( 100 * Best_ACC))
        if epoch % args.store_frequency == 0:
            torch.save(checkpoint, os.path.join(checkpoint_folder, args.model + '_epoch_{}.pth'.format(epoch)) )
    torch.save(checkpoint, os.path.join(checkpoint_folder, args.model + '_final.pth') )
    writer.close()

    #=============================================================================================
    for name, layer in model.named_modules():
        if isinstance(layer, myLinear): 
            path = name.split('.')
            module = model
            if len(path)>1:
                for p in path[:-1]:
                    module = getattr(module, p)
            setattr(module, path[-1], layer.merge_and_unload())  

    print("Testing accuracy of the finel finetuned model...")
    loss_Finetuned, acc_Finetuned = val(model, DEVICE, val_loader, loss_function)
    print('Best accuracy: {:.2f}%\n'.format( 100 * Best_ACC))

    print("----------------------------------------")
    print("Summary:")
    Finetuned_macs, Finetuned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("Base MACs: %.2f G, Finetuned MACs: %.2f G"%(base_macs/1e9, Finetuned_macs/1e9))
    print("Base Params: %.2f M, Finetuned Params: %.2f M"%(base_params/1e6, Finetuned_params/1e6))
    print("Base Loss: %.4f, Finetuned Loss: %.4f"%(loss_ori, loss_Finetuned))
    print("Base Accuracy: %.4f, Finetuned Accuracy: %.4f (Best: %.4f)"%(acc_ori, acc_Finetuned, Best_ACC))


    output_path = "./output_evaluation"
    if args.dependency:
        output_path = os.path.join(output_path, "dependency")
    figure_path = os.path.join(output_path, "figures")
    table_path = os.path.join(output_path, "tables")
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path, exist_ok=True)
    if not os.path.isdir(table_path):
        os.makedirs(table_path, exist_ok=True)
    avgloss, df, (total_acc, total_correct, total_num) = Evaluate(model, DEVICE, val_loader, loss_function, n_labels)
    print("--------EVALUATING MODEL--------")
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, total_correct, total_num, 100 * total_acc))
    print('Correction for each class:')
    print(df.sort_values(by=["correct"])["correct"])
    print('Total number of each class:')
    print(df.sort_values(by=["Total_numbers"])["Total_numbers"])
    print('Accuracy for each class:')
    print(df.sort_values(by=["accuracy"])["accuracy"])
    
    output_file = os.path.join(table_path, args.model + "_{}_".format(args.dataset) + ("inchannel_" if args.finetuning_channel == "in" else "") +\
                                    "by_" + args.finetuning_type + ("_bottleneck" if args.bottleneck else "") +\
                                    ("_global" if args.global_finetuning else "") +\
                                    (("_" + args.label_aggregator) if args.label_aggregator != "None" else "")  +\
                                    ("_freeze_attn" if args.freeze_attn else "") + "_{}epochs.csv".format(args.EPOCHS))
    df.to_csv(output_file,)

if __name__=='__main__':
    main()