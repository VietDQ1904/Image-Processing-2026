import sys
from pathlib import Path
import os
import subprocess

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("Installing GNN dependencies...")
    packages = [
        "torch-scatter", 
        "torch-sparse", 
        "torch-cluster", 
        "torch-spline-conv", 
        "torch-geometric", 
        "ogb",
        "logger" # Only if 'logger.py' isn't in your uploaded folder
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
    print("Installation complete.")

from pathlib import Path
current_file = Path(os.path.abspath(__file__))

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    project_root = "/kaggle/working"
else:
    project_root = str(current_file.parents[3])

#sys.path.insert(0, project_root)

# Respect OMP thread count from runner (runner sets MAX_OMP_THREADS).
# Only set defaults if runner has not already configured them.
_n_threads = str(max(1, (os.cpu_count() or 2) // 2))
os.environ.setdefault("OMP_NUM_THREADS",        _n_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS",   _n_threads)
os.environ.setdefault("MKL_NUM_THREADS",        _n_threads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _n_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS",    _n_threads)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"]     = "FALSE"

import argparse
from shutil import copy
import time
import torch

# Use all threads allowed by env (set by runner or default above)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
print(f'[GNN] Using {torch.get_num_threads()} CPU threads for torch operations.', flush=True)

# --- BẮT ĐẦU ĐOẠN CODE VÁ LỖI PYTORCH 2.6 ---
_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _legacy_load
# --- KẾT THÚC ĐOẠN CODE VÁ LỖI ---

import torch.nn.functional as F
from torch.utils.data import DataLoader
import pdb

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

from torch.utils.tensorboard import SummaryWriter

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.transforms import ToSparseTensor
from sklearn.metrics import roc_auc_score
from logger import Logger

from torch.utils.tensorboard import SummaryWriter

class GATConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers, dropout):
        super(GATConv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
                   GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=1)
                   )
        for i in range(num_layers - 2):
             self.convs.append(
                     GATv2Conv(hidden_channels, hidden_channels, heads=1))
        self.convs.append(
                   GATv2Conv(hidden_channels, out_channels, heads=1))
        self.dropout = dropout
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=True,improved=True))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=True,improved=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size, splitting_strategy):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    if splitting_strategy == 'spatial':
        neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)

    # ---------------------------------------------------------------
    # Compute full-graph node embeddings ONCE per epoch.
    # Previously this was called inside the batch loop (very slow on
    # large graphs). Now we compute h once, accumulate losses over all
    # batches, then do a single backward pass.
    # ---------------------------------------------------------------
    print('  [train] Computing node embeddings (GCN forward)...', flush=True)
    t0 = time.time()
    optimizer.zero_grad()
    h = model(data.x, data.adj_t)
    print(f'  [train] Embeddings done in {time.time()-t0:.1f}s. Running edge batches...', flush=True)

    batches = list(DataLoader(range(pos_train_edge.size(0)), batch_size,
                              shuffle=True, num_workers=0))
    n_batches = len(batches)
    total_loss = total_examples = 0
    accumulated_loss = None

    for i, perm in enumerate(batches):
        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        if splitting_strategy == 'random':
            edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                                 device=h.device)
            neg_out = predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        elif splitting_strategy == 'spatial':
            edge = neg_train_edge[perm].t()
            neg_out = predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        else:
            raise ValueError("Splitting Strategy not defined!")

        loss = pos_loss + neg_loss
        accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        if (i + 1) % max(1, n_batches // 5) == 0 or (i + 1) == n_batches:
            print(f'  [train] Batch {i+1}/{n_batches}  loss={loss.item():.4f}', flush=True)

    # Single backward through full computation graph
    print('  [train] Backprop...', flush=True)
    accumulated_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
    optimizer.step()
    print(f'  [train] Epoch done. avg_loss={total_loss/total_examples:.4f}', flush=True)

    return total_loss / total_examples


# from SEAL
def evaluate_mrr(evaluator, pos_train_pred, neg_train_pred,pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)

    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_train_pred,
    })['mrr_list'].mean().item()
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr,valid_mrr, test_mrr)
    
    return results

# from SEAL
def evaluate_auc(train_pred,train_true,val_pred, val_true, test_pred, test_true):
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (train_auc,valid_auc, test_auc)

    return results

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,eval_metric):
    model.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, num_workers=0):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size, num_workers=0):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size, num_workers=0):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size, num_workers=0):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size, num_workers=0):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size, num_workers=0):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    if eval_metric == 'mrr':
        neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
        neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
        neg_valid_pred = neg_valid_pred.view(pos_valid_pred.shape[0], -1)
        result = evaluate_mrr(evaluator, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    elif eval_metric == 'auc':
        train_pred = torch.cat([pos_train_pred, neg_train_pred])
        train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                              torch.zeros(neg_train_pred.size(0), dtype=int)])
        val_pred = torch.cat([pos_valid_pred, neg_valid_pred])
        val_true = torch.cat([torch.ones(pos_valid_pred.size(0), dtype=int), 
                              torch.zeros(neg_valid_pred.size(0), dtype=int)])
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                              torch.zeros(neg_test_pred.size(0), dtype=int)])
        result = evaluate_auc(train_pred,train_true,val_pred, val_true, test_pred, test_true)

    else:
        result = None
        raise ValueError('Evaluation Metric {eval_metric} not implemented for this dataset.')
        
    
    return result


def main():
    parser = argparse.ArgumentParser(description='OGBL (GNN) Algorithm.')
    
    # Log settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--log_dir',type=str, default= "gnn_log")
    parser.add_argument('--n_par_combs',type=int, default = 1) 
    parser.add_argument('--curr_param_idx', type=int, default = 1)
    parser.add_argument('--save_state_dict', action='store_true')
    parser.add_argument('--save_appendix', type=str, default='', 
                        help="an appendix to the save directory")

    # Model settings
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_gat', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.000001) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--eval_metric', type=str, default='auc')
    parser.add_argument('--dataset', type=str,default='ogbl-link_vessap_roi3_spatial_no_edge_attr')
    parser.add_argument('--splitting_strategy',type=str,default='spatial')
    parser.add_argument('--use_edge_weight',action='store_true')

    # Load pretrained model
    parser.add_argument('--load_state_dict',action='store_true')
    parser.add_argument('--test_only',action='store_true')

    args = parser.parse_args()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f'Running on: {args.dataset}')
    print(f'Utilizing evaluation metric: {args.eval_metric}')

    if args.use_edge_weight:

        dataset = PygLinkPropPredDataset(name=args.dataset,
                                        root=str(Path(os.path.abspath(__file__)).parents[1])+'/dataset')
        data = dataset[0]
        data.edge_weight = torch.abs(data.edge_attr[:,2]) # 2 is radius
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)

    else:
        # Load without ToSparseTensor transform to avoid CSR tensor deadlock on Windows.
        # Manually convert edge_index -> adj_t after loading.
        print('Loading dataset...', flush=True)
        dataset = PygLinkPropPredDataset(name=args.dataset,
                                     root=str(Path(os.path.abspath(__file__)).parents[1])+'/dataset')
        data = dataset[0]
        print('Converting to sparse tensor...', flush=True)
        data = T.ToSparseTensor()(data)
        print('Dataset loaded.', flush=True)
     
    data.x = data.x.to(torch.float)

    # normalize x,y,z coordinates  
    print('Normalizing node features...', flush=True)
    data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
    data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
    data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)

    if args.use_node_embedding:
        embedding_name = str(Path(os.path.abspath(__file__)).parents[1])+'/OGB_Node2Vec/node2vec_'+ args.dataset +'.pt'
        data.x = torch.cat([data.x, torch.load(embedding_name)], dim=-1)

    print('Moving data to device...', flush=True)
    data = data.to(device)
    print('Loading edge splits...', flush=True)
    split_edge = dataset.get_edge_split()
    print('Setup complete. Starting training...', flush=True)

    # from Muhan Zhang's OGB SEAL repository

    if args.save_appendix == '':
        if args.use_sage:
            args.save_appendix = '_gnn_sage_'+ ('node_emb_' if args.use_node_embedding else '') + time.strftime("%Y%m%d%H%M%S")
        else:
            args.save_appendix = '_gnn_gcn_' + ('node_emb_' if args.use_node_embedding else '') + time.strftime("%Y%m%d%H%M%S")


    args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
    print('Results will be saved in ' + args.res_dir)

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir) 
    
    # Backup python files.
    # copy('gnn.py', args.res_dir)
    log_file = os.path.join(args.res_dir, 'log.txt')

    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    with open(log_file, 'a') as f:
        f.write('\n' + cmd_input)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    if args.use_gat:
        model = GATConv(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)

    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

        # Pre-compute GCN normalization.
#        adj_t = data.adj_t.set_diag()
#        deg = adj_t.sum(dim=1).to(torch.float)
#        deg_inv_sqrt = deg.pow(-0.5)
#        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
#        data.adj_t = adj_t

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name=args.dataset)
    logger = Logger(args.runs, args)   

    for run in range(args.runs):
        
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        if args.load_state_dict:

            print("Loading State Dictonaries")

            dict_path = str(Path(os.path.abspath(__file__)).parents[0])+'/neurips_state_dict_final_gnn_'

            if args.use_sage:

                dict_path += 'sage_'

            if args.use_node_embedding:

                dict_path += 'embedding_'

            model_path = dict_path + 'model_checkpoint.pth'
            optim_path = dict_path + 'optimizer_checkpoint.pth'
            predictor_path = dict_path + 'predictor_checkpoint.pth'

            model.load_state_dict(
                torch.load(os.path.join(os.getcwd(),model_path), map_location=torch.device('cpu'))
            )
            predictor.load_state_dict(
                torch.load(os.path.join(os.getcwd(),predictor_path), map_location=torch.device('cpu'))
            )
            optimizer.load_state_dict(
                torch.load(os.path.join(os.getcwd(),optim_path), map_location=torch.device('cpu'))
            )

        if args.test_only:
            results = test(model, predictor, data, split_edge, evaluator,
                                args.batch_size, args.eval_metric)

            for key, result in results.items():
                train_res, valid_res, test_res = result
                
                print(key)
                log_text = (   
                    f'Train: {100 * train_res:.2f}%, ' +
                    f'Valid: {100 * valid_res:.2f}%, ' +
                    f'Test: {100 * test_res:.2f}%')

            print(log_text)
            exit()

        # init tensorboard writer
        writer = SummaryWriter(os.path.join(args.log_dir,f'{args.curr_param_idx}_of_{args.n_par_combs}'))

        best_val = 0.0
        best_epoch = 0
        for epoch in range(1, 1 + args.epochs):
            t_epoch = time.time()
            print(f'\n[Epoch {epoch}/{args.epochs}] Run {run+1}/{args.runs} - bat dau...', flush=True)
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.splitting_strategy)
            print(f'[Epoch {epoch}/{args.epochs}] Train loss={loss:.4f}  ({time.time()-t_epoch:.1f}s)', flush=True)

            writer.add_scalar('loss', loss, epoch)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator,
                               args.batch_size, args.eval_metric)
                for key, result in results.items():
                    train_res, valid_res, test_res = result
                    
                    append = 'neurips_state_dict_final_gnn_'

                    if args.use_sage:
                        append += 'sage_'
                    if args.use_node_embedding:
                        append += 'embedding_'

                    if valid_res > best_val:
                        
                        print("save best validation!")
                        best_val = valid_res
                        best_epoch = epoch
                        if args.save_state_dict:
                            predictor_name = append + 'predictor_checkpoint.pth'
                            model_name = append + 'model_checkpoint.pth'
                            optimizer_name = append + 'optimizer_checkpoint.pth'
                            torch.save(predictor.state_dict(), predictor_name)
                            torch.save(model.state_dict(), model_name)
                            torch.save(optimizer.state_dict(), optimizer_name)

                    logger.add_result(run, result)

                    if epoch % args.log_steps == 0:
                        for key, result in results.items():
                            train_res, valid_res, test_res = result
                            writer.add_scalars('auc',
                                               {'train': train_res,
                                                'valid': valid_res,
                                                'test': test_res}, epoch)
                            print(key)
                            log_text = (f'Run: {run + 1:02d}, ' +
                                f'Epoch: {epoch:02d}, ' +
                                f'Loss: {loss:.4f}, ' +
                                f'Train: {100 * train_res:.2f}%, ' +
                                f'Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')

                            print(log_text)
                            with open(log_file, 'a') as f:
                                print(log_text, file=f)


        print('GraphSAGE' if args.use_sage else 'GCN')
        logger.print_statistics(run)
        with open(log_file, 'a') as f:
            print('GraphSAGE' if args.use_sage else 'GCN', file=f)
            logger.print_statistics(run,f=f)

        # save hyperparameters for tensorboard logging
        writer.add_hparams(
            vars(args),
            {'hparam/loss':loss,
                'hparam/auc':best_val,
                'hparam/best_epoch':best_epoch})
        writer.flush()
        writer.close()

    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics()
    with open(log_file, 'a') as f:
        print('GraphSAGE' if args.use_sage else 'GCN', file=f)
        logger.print_statistics(f=f)

if __name__ == "__main__":
    main()
