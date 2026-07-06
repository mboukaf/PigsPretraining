import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.io
import scipy.signal
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =============================================================================
# 1. DATA LOADING FUNCTION
# =============================================================================
def import_data(file_path):
    """
    Safely loads a .mat file (whether it's older scipy.io format or newer h5py format)
    and converts it into a pandas DataFrame.
    """
    data_dict = {}
    try:
        # Try loading with scipy.io (MATLAB v7.2 and below)
        mat = scipy.io.loadmat(file_path)
        # The data is stored inside a MATLAB struct (e.g. mat['Pig1'])
        struct_key = [k for k in mat.keys() if not k.startswith('__')][0]
        struct = mat[struct_key]
        # Extract each physiological signal from the struct
        for field_name in struct.dtype.names:
            data_dict[field_name] = struct[0, 0][field_name].flatten()
    except NotImplementedError:
        # If it fails, it's likely MATLAB v7.3, so use h5py
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data_dict[key] = np.array(f[key]).flatten()
                
    return pd.DataFrame(data_dict)

# =============================================================================
# 2. PYTORCH DATASET (WITH WINDOWING & NORMALIZATION)
# =============================================================================
class WindowedBPDataset(Dataset):
    def __init__(self, data_dir, pig_ids=range(1, 9), T=None, window_size=500, split_type=None, split="train", augment=False):
        """
        Initializes the dataset, extracting 'window_size' blocks of data.
        window_size=500 at 250Hz means 2-second continuous windows.
        split_type: 'time' (80% train, 20% test per pig)
        split: 'train' or 'test'
        augment: 'True' randomly scales amplitude and adds noise
        """
        self.augment = augment
        all_data = []
        self.valid_indices = []
        self.window_size = window_size
        current_idx = 0
        
        print(f"Loading Pig data for {split.upper()} set (split_type={split_type})...")
        for pid in pig_ids:
            file_path = os.path.join(data_dir, f"Pig{pid}.mat")
            df = import_data(file_path)
            
            if T is not None:
                df = df.iloc[:int(T * 60 * 250)]
                
            # ---> DYNAMICALLY ALIGN FLOW TO PRESSURE <---
            p_chunk = df['aorticPressure'].values[:20000]
            f_chunk = df['aorticFlow'].values[:20000]
            
            dp_dt = np.gradient(p_chunk)
            corr = scipy.signal.correlate(dp_dt, f_chunk, mode='full')
            lags = scipy.signal.correlation_lags(len(dp_dt), len(f_chunk), mode='full')
            best_lag = lags[np.argmax(corr)]
            
            # Shift flow by the calculated optimal lag
            df['aorticFlow'] = df['aorticFlow'].shift(best_lag)
            df = df.dropna().reset_index(drop=True)
            
            # ---> TRAIN / TEST SPLIT LOGIC <---
            if split_type == "time":
                split_idx = int(len(df) * 0.8) # 80% Train, 20% Test
                if split == "train":
                    df = df.iloc[:split_idx].reset_index(drop=True)
                elif split == "test":
                    df = df.iloc[split_idx:].reset_index(drop=True)
                
            num_samples = len(df)
            
            # Record valid starting indices for this specific pig to prevent 
            # windows from crossing over from one pig to the next!
            self.valid_indices.extend(range(current_idx, current_idx + num_samples - self.window_size + 1))
            current_idx += num_samples
            
            df['pig_id'] = pid
            all_data.append(df)
            print(f"  -> Loaded Pig {pid} ({num_samples} samples, Flow shifted by {best_lag} timesteps)")
            
        self.full_df = pd.concat(all_data, ignore_index=True)
        print(f"Data loading complete! Total raw windows: {len(self.valid_indices)}")
        
        # --- QUALITY CONTROL FILTERING ---
        print("Running Quality Control to remove corrupted sensor windows...")
        clean_indices = []
        raw_p = self.full_df['aorticPressure'].values
        raw_f = self.full_df['aorticFlow'].values
        
        for start in self.valid_indices:
            end = start + self.window_size
            p_w = raw_p[start:end]
            f_w = raw_f[start:end]
            
            # QC Rules (Biologically viable ranges & No Flatlines)
            if np.max(p_w) < 50 or np.min(p_w) < 0: continue # Pressure must be real
            if np.max(f_w) < 1: continue                     # Flow must exist
            if np.std(p_w) < 5: continue                     # No flatlining
                
            clean_indices.append(start)
            
        print(f"QC Complete: Removed {len(self.valid_indices) - len(clean_indices)} corrupted windows.")
        self.valid_indices = clean_indices
        
        # X: Aortic Pressure (Input)
        X_data = self.full_df['aorticPressure'].values.reshape(-1, 1)
        self.x_mean = X_data.mean()
        self.x_std = X_data.std()
        X_data = (X_data - self.x_mean) / self.x_std
        self.X = torch.tensor(X_data, dtype=torch.float32)
        
        # Y: Aortic Flow (Target for Fine-Tuning)
        Y_data = self.full_df['aorticFlow'].values.reshape(-1, 1)
        self.y_mean = Y_data.mean()
        self.y_std = Y_data.std()
        Y_data = (Y_data - self.y_mean) / self.y_std
        self.Y = torch.tensor(Y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.window_size
        
        # Slice the 2-second windows (Shape: [500, 1])
        window_x = self.X[start:end].clone()
        window_y = self.Y[start:end]
        
        # Physiological Data Augmentation (Only if augment=True)
        if self.augment:
            # 1. Amplitude Scaling (Simulate different pulse pressures)
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
            window_x = window_x * scale
            
            # 2. Gaussian Noise (Simulate sensor static)
            noise = torch.randn_like(window_x) * 0.05
            window_x = window_x + noise
        
        return window_x, window_y

# =============================================================================
# 3. TRANSFORMER ARCHITECTURE
# =============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=10, in_channels=1, d_model=128, window_size=500):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.num_patches = window_size // patch_size
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

class ResNetPatchEmbedding(nn.Module):
    def __init__(self, patch_size=10, in_channels=1, d_model=128, window_size=500):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = window_size // patch_size
        
        # Convolutional Stem (Extracts local high-frequency features)
        self.conv1 = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.gelu = nn.GELU()
        
        # Residual Block
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        
    def forward(self, x):
        # x is [Batch, Seq_Len, Channels] = [B, 500, 1]
        x = x.transpose(1, 2) # [B, 1, 500]
        
        # Stem
        x = self.conv1(x) # [B, d_model, 100]
        x = self.bn1(x)
        x = self.gelu(x)
        
        # Residual
        res = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x + res) # Add residual connection
        
        x = x.transpose(1, 2) # [B, 100, d_model]
        return x

class BPTransformer(nn.Module):
    def __init__(self, patch_size=10, in_channels=1, d_model=128, window_size=500, 
                 n_layers=4, n_heads=4, mask_ratio=0.15, use_resnet=False):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.num_patches = window_size // patch_size
        
        if use_resnet:
            self.patch_embed = ResNetPatchEmbedding(patch_size, in_channels, d_model, window_size)
        else:
            self.patch_embed = PatchEmbedding(patch_size, in_channels, d_model, window_size)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # ---> FIXED SINE/COSINE POSITIONAL ENCODING <---
        # Instead of learning time over 100 epochs, we hardcode a perfect mathematical clock!
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model), requires_grad=False)
        pos = torch.arange(self.num_patches, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model))
        self.pos_embed[0, :, 0::2] = torch.sin(pos * div_term)
        self.pos_embed[0, :, 1::2] = torch.cos(pos * div_term)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                   dim_feedforward=d_model*4, 
                                                   batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, patch_size)
        self.flow_head = nn.Linear(d_model, patch_size) # New Regression Head

    def random_masking(self, x):
        batch_size, seq_len, _ = x.shape
        noise = torch.rand(batch_size, seq_len, device=x.device) 
        ids_shuffle = torch.argsort(noise, dim=1) 
        len_mask = int(seq_len * self.mask_ratio)
        mask_indices = ids_shuffle[:, :len_mask]
        
        mask_bool = torch.ones(batch_size, seq_len, device=x.device)
        mask_bool.scatter_(1, mask_indices, 0)
        mask_bool_expanded = mask_bool.unsqueeze(-1)
        
        x_masked = x * mask_bool_expanded + self.mask_token * (1 - mask_bool_expanded)
        return x_masked, mask_bool

    def forward(self, x):
        # 1. Embed the continuous signal into discrete patches
        x_embed = self.patch_embed(x) 
        
        # 2. Masking
        if self.training:
            x_masked, mask_bool = self.random_masking(x_embed)
        else:
            x_masked = x_embed
            mask_bool = torch.ones(x.shape[0], self.num_patches, device=x.device)
            
        # 3. ---> Add timestamps AFTER masking! <---
        x_masked = x_masked + self.pos_embed
            
        # 4. Pass through the Transformer
        transformer_out = self.transformer(x_masked) 
        predictions = self.head(transformer_out) 
        
        return predictions, mask_bool

    def forward_finetune(self, x):
        # 1. Embed continuous signal
        x_embed = self.patch_embed(x)
        # 2. Add timestamps (No Masking!)
        x_encoded = x_embed + self.pos_embed
        # 3. Pass through Transformer
        transformer_out = self.transformer(x_encoded)
        # 4. Predict Flow
        flow_predictions = self.flow_head(transformer_out)
        return flow_predictions

# =============================================================================
# 4. PRETRAINING LOOP
# =============================================================================
def pretrain_transformer(model, dataloader, val_dataloader=None, epochs=5, lr=1e-4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    loss_history = [] 
    
    print(f"Starting Pretraining on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            batch_size = x.size(0)
            
            optimizer.zero_grad()
            predictions, mask_bool = model(x) 
            
            # 1. Standard MSE on hidden patches
            x_target = x.view(batch_size, model.num_patches, model.head.out_features)
            loss_all = (predictions - x_target) ** 2
            mask_hidden = (1 - mask_bool).unsqueeze(-1) 
            loss_mse = (loss_all * mask_hidden).sum() / (mask_hidden.sum() * model.head.out_features)
            
            # 2. Shape Loss (First Derivative) on the continuous reconstructed wave
            full_recon = x_target * mask_bool.unsqueeze(-1) + predictions * mask_hidden
            full_recon_flat = full_recon.view(batch_size, -1)
            x_flat = x.view(batch_size, -1)
            
            pred_diff = torch.diff(full_recon_flat, dim=1)
            target_diff = torch.diff(x_flat, dim=1)
            
            # Multiply shape loss by 10 to heavily penalize straight lines!
            loss_shape = ((pred_diff - target_diff) ** 2).mean()
            loss = loss_mse + 10.0 * loss_shape
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Print every 50 batches so you can track progress
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss) 
        
        # --- VALIDATION PASS ---
        val_loss_str = ""
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for v_x, _ in val_dataloader:
                    v_x = v_x.to(device)
                    v_batch_size = v_x.size(0)
                    v_preds, v_mask = model(v_x)
                    
                    v_target = v_x.view(v_batch_size, model.num_patches, model.head.out_features)
                    v_loss_all = (v_preds - v_target) ** 2
                    v_mask_hidden = (1 - v_mask).unsqueeze(-1)
                    v_loss_mse = (v_loss_all * v_mask_hidden).sum() / (v_mask_hidden.sum() * model.head.out_features)
                    
                    v_full_recon = v_target * v_mask.unsqueeze(-1) + v_preds * v_mask_hidden
                    v_full_recon_flat = v_full_recon.view(v_batch_size, -1)
                    v_x_flat = v_x.view(v_batch_size, -1)
                    
                    v_pred_diff = torch.diff(v_full_recon_flat, dim=1)
                    v_target_diff = torch.diff(v_x_flat, dim=1)
                    v_loss_shape = ((v_pred_diff - v_target_diff) ** 2).mean()
                    
                    val_loss += (v_loss_mse + 10.0 * v_loss_shape).item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_loss_str = f" | Val Loss: {avg_val_loss:.4f}"
            model.train()
            
        print(f"=== Epoch {epoch+1} Completed | Train Loss: {avg_loss:.4f}{val_loss_str} ===")
        
    print("Pretraining Complete!")
    return model, loss_history

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
def visualize_reconstruction(model, dataset, index=150):
    device = next(model.parameters()).device
    model.eval() 
    
    x_real, _ = dataset[index] 
    x_real = x_real.unsqueeze(0).to(device) 
    
    with torch.no_grad():
        x_embed = model.patch_embed(x_real)
        
        model.train() 
        x_masked, mask_bool = model.random_masking(x_embed)
        model.eval() 
        
        x_masked = x_masked + model.pos_embed
        transformer_out = model.transformer(x_masked)
        predictions = model.head(transformer_out) 
        
    predicted_wave = predictions.cpu().view(-1).numpy()
    real_wave = x_real.cpu().view(-1).numpy()
    mask_flat = mask_bool.cpu().view(-1).numpy() 
    
    mask_stretched = np.repeat(mask_flat, model.patch_embed.patch_size)
    
    # ---> UN-NORMALIZE FOR PLOTTING <---
    predicted_wave = (predicted_wave * dataset.x_std) + dataset.x_mean
    real_wave = (real_wave * dataset.x_std) + dataset.x_mean
    
    plt.figure(figsize=(14, 6))
    plt.plot(real_wave, color='gray', alpha=0.3, label='Original Real Wave', linewidth=4)
    
    visible_wave = real_wave.copy()
    visible_wave[mask_stretched == 0] = np.nan 
    plt.plot(visible_wave, color='black', label='What the Model Saw (Visible Input)', linewidth=2)
    
    predicted_hidden = predicted_wave.copy()
    predicted_hidden[mask_stretched == 1] = np.nan 
    plt.plot(predicted_hidden, color='red', linestyle='dashed', label="Model's Guess for Hidden Parts", linewidth=2.5)
    
    plt.title(f"Visualizing Transformer Reconstruction (Window #{index})")
    plt.xlabel("Timesteps")
    plt.ylabel("Aortic Pressure (mmHg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. FINE-TUNING LOOP
# =============================================================================
def finetune_transformer(model, dataloader, val_dataloader=None, epochs=5, lr=5e-5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    # Only train the Flow Head! Freeze the Transformer Encoder for now.
    # Actually, we can do full fine-tuning. We'll fine-tune everything.
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    loss_history = []
    
    print(f"Starting Fine-Tuning on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            optimizer.zero_grad()
            predictions = model.forward_finetune(x)
            
            y_target = y.view(batch_size, model.num_patches, model.flow_head.out_features)
            
            # Simple MSE across the whole wave (No masking!)
            loss_mse = ((predictions - y_target) ** 2).mean()
            
            # We REMOVE Shape Loss for Flow because the true flow has artificial 
            # vertical jumps (discontinuities) which cause the derivative loss to explode 
            # and create jagged noise in the predictions!
            loss = loss_mse
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # --- VALIDATION PASS ---
        val_loss_str = ""
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for v_x, v_y in val_dataloader:
                    v_x, v_y = v_x.to(device), v_y.to(device)
                    v_batch_size = v_x.size(0)
                    v_preds = model.forward_finetune(v_x)
                    v_target = v_y.view(v_batch_size, model.num_patches, model.flow_head.out_features)
                    
                    val_loss += ((v_preds - v_target) ** 2).mean().item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_loss_str = f" | Val Loss: {avg_val_loss:.4f}"
            model.train()
            
        print(f"=== Epoch {epoch+1} Completed | Train Loss: {avg_loss:.4f}{val_loss_str} ===")
        
    print("Fine-Tuning Complete!")
    return model, loss_history

# =============================================================================
# 7. FINE-TUNING VISUALIZATION
# =============================================================================
def visualize_finetuning(model, dataset, index=150):
    device = next(model.parameters()).device
    model.eval()
    
    x_real, y_real = dataset[index]
    x_real = x_real.unsqueeze(0).to(device)
    y_real = y_real.unsqueeze(0).to(device)
    
    with torch.no_grad():
        flow_predictions = model.forward_finetune(x_real)
        
    predicted_flow = flow_predictions.cpu().view(-1).numpy()
    true_flow = y_real.cpu().view(-1).numpy()
    input_pressure = x_real.cpu().view(-1).numpy()
    
    # UN-NORMALIZE
    predicted_flow = (predicted_flow * dataset.y_std) + dataset.y_mean
    true_flow = (true_flow * dataset.y_std) + dataset.y_mean
    input_pressure = (input_pressure * dataset.x_std) + dataset.x_mean
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1.plot(input_pressure, color='black', linewidth=2)
    ax1.set_title(f"Input: Aortic Pressure (Window #{index})")
    ax1.set_ylabel("Pressure (mmHg)")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(true_flow, color='gray', alpha=0.5, label='True Aortic Flow', linewidth=4)
    ax2.plot(predicted_flow, color='blue', linestyle='dashed', label='Predicted Aortic Flow', linewidth=2.5)
    ax2.set_title("Output: True vs Predicted Aortic Flow")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Flow")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# EXECUTION (Run this script to test it end-to-end)
# =============================================================================
if __name__ == "__main__":
    # Automatically resolve paths based on the location of this script
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_directory = os.path.join(base_dir, "data", "osfstorage-archive")
    
    print("1. Creating Datasets...")
    train_dataset = WindowedBPDataset(data_dir=data_directory, T=5, split="train") 
    test_dataset = WindowedBPDataset(data_dir=data_directory, T=5, split="test")
    
    print("\n2. Initializing DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("\n3. Initializing Model...")
    model = BPTransformer(patch_size=5, window_size=500, n_layers=4, n_heads=4, mask_ratio=0.15)
    
    model_pretrain_path = os.path.join(base_dir, "models", "pretrained_bp_model.pth")
    model_finetune_path = os.path.join(base_dir, "models", "finetuned_flow_model.pth")
    
    print("\n4. Starting Phase 1: Pretraining (5 epochs)...")
    if os.path.exists(model_pretrain_path):
        print(" -> Found existing pretrained model! Loading weights...")
        model.load_state_dict(torch.load(model_pretrain_path, map_location=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')))
        trained_model = model
    else:
        trained_model, pretrain_history = pretrain_transformer(model, train_loader, val_dataloader=test_loader, epochs=5)
        torch.save(trained_model.state_dict(), model_pretrain_path)
        print(" -> Pretrained model saved to disk!")
        
    print("\n5. Visualizing Phase 1 Reconstruction...")
    visualize_reconstruction(trained_model, test_dataset, index=250)
    
    print("\n6. Starting Phase 2: Fine-Tuning (Pressure -> Flow, 5 epochs)...")
    if os.path.exists(model_finetune_path):
        print(" -> Found existing fine-tuned model! Loading weights...")
        trained_model.load_state_dict(torch.load(model_finetune_path, map_location=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')))
        finetuned_model = trained_model
    else:
        finetuned_model, ft_history = finetune_transformer(trained_model, train_loader, val_dataloader=test_loader, epochs=5)
        torch.save(finetuned_model.state_dict(), model_finetune_path)
        print(" -> Fine-Tuned model saved to disk!")
    
    print("\n7. Visualizing Phase 2 Flow Prediction...")
    visualize_finetuning(finetuned_model, test_dataset, index=250)
