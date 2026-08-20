import argparse
import torch
from ppi_model_edge_attr import *
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import numpy as np


# run example：
#  python your_script_name.py --data_path DATA/all.pt --checkpoint ./STGPPI.pth

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate PPI affinity prediction model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (e.g., DATA/all.pt)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (e.g., ./fold_3_700_best_ppbbind_model.pth)")
    return parser.parse_args()


if __name__ == "__main__":
    # 1. Parse command-line arguments
    args = parse_args()
    data_path = args.data_path
    checkpoint_path = args.checkpoint

    # 2. Load dataset and model
    print(f"Loading dataset from: {data_path}")
    data_list = torch.load(data_path, weights_only=False)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    fixed_batch = 1
    fixed_hiddendim = 32
    fixed_aa_embed = 16

    model = PPIAffinityModelEdgeAttr(
        cont_feat_dim=3,
        aa_embed_dim=fixed_aa_embed,
        hidden_dim=fixed_hiddendim,
        conv_type='transformer'
    ).to(DEVICE)

    print(f"Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    all_loader = DataLoader(data_list, batch_size=fixed_batch, shuffle=True)

    # 3. Model prediction
    model.eval()
    pred_list, true_list = [], []

    print("Starting prediction...")
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            pred_list.append(pred)
            true_list.append(batch.y)

    # 4. Calculate evaluation metrics
    pred_arr = torch.cat(pred_list).cpu().numpy().ravel()
    true_arr = torch.cat(true_list).cpu().numpy().ravel()

    pearson, _ = pearsonr(true_arr, pred_arr)
    spearman, _ = spearmanr(true_arr, pred_arr)
    rmse = np.sqrt(mean_squared_error(true_arr, pred_arr))

    print(f"Pearson: {pearson:.4f} | Spearman: {spearman:.4f} | RMSE: {rmse:.4f}")
    print("Evaluation finished!")