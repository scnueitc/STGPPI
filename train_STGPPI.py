import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from model.ppi_model_edge_attr  import *
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import getopt
import argparse


# ============ Get data_path / save_dir via command-line arguments ============
# One-line run example:
#   python train_pdbbind.py --data_path DATA/all.pt --save_dir ./save_model_pdbbind_power_transformer
# When no args are passed, the defaults below are used (same behavior as before).
parser = argparse.ArgumentParser(description="PDBbind transformer affinity training")
parser.add_argument("--data_path", type=str, default="DATA/all.pt",
                    help="Path to dataset (.pt), default DATA/all.pt")
parser.add_argument("--save_dir", type=str, default="./save_model_pdbbind_power_transformer",
                    help="Model save directory, default ./save_model_pdbbind_power_transformer")
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = args.data_path
data_list = torch.load(data_path, weights_only=False)

# Create save directory
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)


if __name__ == "__main__":
    # Fixed hyperparameters
    fixed_batch = 200
    fixed_hiddendim = 32
    fixed_aa_embed = 16
    init_lr = 1e-3       # Initial learning rate
    epochs = 1000
    fold_loss_record = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=50)
    fold_metrics = []
    # Fixed aa_embed_dim=32
    model = PPIAffinityModelEdgeAttr(
            cont_feat_dim=3,
            aa_embed_dim=fixed_aa_embed,
            hidden_dim=fixed_hiddendim,
            conv_type='transformer'
        ).to(DEVICE)

    for i in range(2):
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(data_list)):
            init_lr = 1e-3       # Initial learning rate
            optimizer = getopt.get_optimizer(model, lr=1e-3, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6)
            print(f"\n========== Fold {fold_idx+1}/5 ==========")
            train_data = [data_list[i] for i in train_idx]
            val_data   = [data_list[i] for i in val_idx]

            train_loader = DataLoader(train_data, batch_size=fixed_batch, shuffle=True)
            val_loader   = DataLoader(val_data, batch_size=1, shuffle=False)
            all_loader   = DataLoader(data_list , batch_size=1, shuffle=False)
            criterion = nn.MSELoss()
            epoch_loss_list = []
            print(f"start training Fold{fold_idx+1}, epoch={epochs}")
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    #loss1 = weight_mse.weighted_mse_loss(out,batch.y,batch.xs1)
                    #loss = criterion(out.squeeze(-1), batch.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                scheduler.step(avg_loss) # Adjust lr based on validation loss
                epoch_loss_list.append(avg_loss)
                if epoch % 1 == 0:
                    print(f"Epoch[{epoch:4d}] TrainLoss:{avg_loss:.6f}| LR:{optimizer.param_groups[0]['lr']:.6f}")
                if epoch % 100 == 0:
                    # Save model
                    save_path = os.path.join(save_dir, f"fold_{fold_idx+1}_{epoch}_best_ppbbind_model.pth")
                    torch.save({
                           'model_state_dict': model.state_dict(),
                           'param':{"batch_size":fixed_batch,"hidden_dim":fixed_hiddendim,"aa_embed":fixed_aa_embed,"init_lr":init_lr},
                           'fold': fold_idx+1
                    }, save_path)
                    print(f"Fold{fold_idx+1} model saved at: {save_path}")
                if epoch % 10 == 0:
                   # Validate each epoch to update the learning rate
                    model.eval()
                    val_loss_sum = 0.0
                    with torch.no_grad():
                       for batch in val_loader:
                           batch = batch.to(DEVICE)
                           pred = model(batch)
                           #val_loss_sum += criterion(pred.squeeze(-1), batch.y).item()
                           val_loss_sum += criterion(pred, batch.y).item()

                    val_avg_loss = val_loss_sum / len(val_loader)
                    print(f"Epoch[{epoch:4d}] TrainLoss:{avg_loss:.4f} | ValLoss:{val_avg_loss:.4f} | LR:{optimizer.param_groups[0]['lr']:.6f}")

            fold_loss_record.append(epoch_loss_list)
            # Validation set evaluation
            model.eval()
            pred_list, true_list = [], []
            with torch.no_grad():
                for batch in all_loader:
                    batch = batch.to(DEVICE)
                    pred = model(batch)
                    pred_list.append(pred)
                    true_list.append(batch.y)

            pred_arr = torch.cat(pred_list).cpu().numpy().ravel()
            true_arr = torch.cat(true_list).cpu().numpy().ravel()

            pearson, _ = pearsonr(true_arr, pred_arr)
            spearman, _ = spearmanr(true_arr, pred_arr)
            rmse = np.sqrt(mean_squared_error(true_arr, pred_arr))
            fold_metrics.append([pearson, spearman, rmse])
            print(f"Fold{fold_idx+1} Result | Pearson:{pearson:.4f} | Spearman:{spearman:.4f} | RMSE:{rmse:.4f}")
            with open('transformeroutput.txt', 'w', encoding='utf-8') as f:
                print(f"Fold{fold_idx+1} Result | Pearson:{pearson:.4f} | Spearman:{spearman:.4f} | RMSE:{rmse:.4f}",file=f)


    # Output 5-fold average
    print("fold_metrics:",fold_metrics)
    fold_metrics = np.array(fold_metrics)
    avg_pearson = fold_metrics[:,0].mean()
    avg_spearman = fold_metrics[:,1].mean()
    avg_rmse = fold_metrics[:,2].mean()

    print("\n===== 5-fold cross-validation average results =====")
    print(f"Avg Pearson:  {avg_pearson:.4f}")
    print(f"Avg Spearman: {avg_spearman:.4f}")
    print(f"Avg RMSE:     {avg_rmse:.4f}")
