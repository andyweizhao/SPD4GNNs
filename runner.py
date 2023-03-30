import warnings
import json
import time
import torch
import os 
import random
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score
from config import load_arguments
from data import DatasetLoader
from models import EuclideanGNNModel, SPDGNNModel  
from pytorchtools import EarlyStopping             

class Runner(object):

    def __init__(self, model, optimizer, data, args):
        self.model = model
        self.optimizer = optimizer        
        
        self.data = data
        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask
        self.args = args
        
        self.loss_function = torch.nn.CrossEntropyLoss()

    def train_epoch(self):
        model.train()
        
        optimizer.zero_grad()
                        
        pred_y = model(self.data.x, self.data.edge_index)
        loss = self.loss_function(pred_y[self.train_mask], self.data.y[self.train_mask])
        
        loss.backward()
        optimizer.step()
                        
        tr_loss = loss.item()

        return tr_loss

    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():

            pred_y = model(self.data.x, self.data.edge_index)
            
            loss = self.loss_function(pred_y[mask], self.data.y[mask])            
            acc = accuracy_score(pred_y[mask].argmax(dim=1).cpu(), self.data.y[mask].cpu())   
                        
            return loss, acc

    def run(self):
        
        checkpoint_path = f'save/model.pt'
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=checkpoint_path)    

        for epoch in range(args.epoch): 
            start = time.perf_counter()
            train_loss = self.train_epoch()   
            exec_time = time.perf_counter() - start            

            if epoch % self.args.val_every == 0:
        
                val_loss, val_acc = self.evaluate(self.val_mask)
                
                print(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs | valid loss: {val_loss:.4f} | valid acc: {val_acc * 100:.2f}')
            
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    break
        model.load_state_dict(torch.load(checkpoint_path))

        model.eval()

        test_loss, test_acc = self.evaluate(self.test_mask)

        print(f"Final Results | Accuracy: {test_acc * 100:.2f}")

def setup_envs(seed=-1):
    warnings.filterwarnings("ignore")

    if seed == -1: seed = random.randint(1, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.set_default_tensor_type(torch.DoubleTensor)

    args = load_arguments()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args 

def build_dataset(args):

    data = DatasetLoader(args, os.path.join('data', args.dataset))

    data.to(args.device)    

    args.num_node_features = data.num_node_features
    args.num_classes = data.y.max().item() + 1
        
    if len(data.train_mask.shape) > 1:  # When dataset contains cross-validation splits, we only use the first one.
        data.train_mask = data.train_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    #A = A + Id
    data.edge_index, _ = add_remaining_self_loops(data.edge_index) 

    return data 

def load_hyperparameters(args):
    with open(f'json/{args.dataset}.json',) as f:
        parameters = json.load(f)[args.model]
        
        args.learningrate = parameters.get('learningrate', None)
        args.dropout = parameters.get('dropout', None)
        args.weight_decay = parameters.get('weight_decay', None)
        args.nonlinear = parameters.get('nonlinear', None)
        args.hidden_dims = parameters.get('hidden_dims', None)

    return args  

def GNNFactory(type_str, args):
    classes = {
            "spd": SPDGNNModel,
            "euclidean": EuclideanGNNModel
    }
    return classes[type_str](args).to(args.device)
    
   
if __name__ == "__main__":
    
    args = setup_envs(seed=42)    
    args = load_hyperparameters(args)
    
    dataset = build_dataset(args)
    model = GNNFactory(args.manifold, args)  

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay, amsgrad=False)

    runner = Runner(model, optimizer, dataset, args)
    runner.run()
