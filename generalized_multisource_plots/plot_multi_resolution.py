import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

# Publication-quality figure styling: large fonts so the in-figure text
# (titles, axis labels, ticks, legend) stays readable once the figure is
# scaled down to column width in the paper.
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 16,
    'axes.titlepad': 12,
    'axes.labelsize': 19,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
})
import pandas as pd


import re
import numpy as np

def parse_all_methods_da(full_log):
    """
    Parses the log file for Epochs, Test MSE, Test Loss, Cost, and Weights.
    Targets the multi-resolution log format: TestMSE 0.00... | W [w0, w1, w2]
    """
    results = {}
    # Split by the header: >>> Training Method Name <<<
    sections = re.split(r'>>> Training (.*?) <<<', full_log)
    
    for i in range(1, len(sections), 2):
        method_name = sections[i].strip()
        log_content = sections[i+1]
        
        epochs, mse_vals, loss_vals, cost_vals, all_weights = [], [], [], [], []
        
        lines = log_content.strip().split('\n')
        for line in lines:
            # Check for lines containing "Ep" and "MSE" or "Loss"
            if re.search(r'^Ep\s+\d+', line):
                try:
                    # 1. Extract epoch number
                    epoch = int(re.search(r'Ep\s+(\d+)', line).group(1))
                    
                    # 2. Extract Test MSE
                    # Handles "TestMSE 0.0..." or "MSE (Tr/Val/Ts): .../.../VAL"
                    mse_match = re.search(r'TestMSE\s+([\d\.e+-]+)', line)
                    if not mse_match:
                        mse_match = re.search(r'MSE.*?/.*?/([\d\.e+-]+)', line)
                    test_mse = float(mse_match.group(1)) if mse_match else None

                    # 3. Extract Test Loss
                    loss_match = re.search(r'TestLoss\s+([\d\.e+-]+)', line)
                    if not loss_match:
                        loss_match = re.search(r'Loss.*?/.*?/([\d\.e+-]+)', line)
                    test_loss = float(loss_match.group(1)) if loss_match else test_mse
                    
                    # 4. Extract Cost
                    cost_match = re.search(r'Cost\s+([\d\.e+-]+)', line)
                    cost = float(cost_match.group(1)) if cost_match else 0.0
                    
                    # 5. Extract Weights
                    weights = []
                    # Format: W [0.34, 0.32, 0.32]
                    w_list_match = re.search(r'W\s+\[(.*?)\]', line)
                    # Format: Weights: [S0:0.312 | S1:0.263 ...]
                    w_pair_match = re.findall(r'S\d+:([\d\.e+-]+)', line)
                    
                    if w_list_match:
                        weights = [float(x.strip()) for x in w_list_match.group(1).split(',')]
                    elif w_pair_match:
                        weights = [float(x) for x in w_pair_match]
                    
                    if test_mse is not None:
                        epochs.append(epoch)
                        mse_vals.append(test_mse)
                        loss_vals.append(test_loss)
                        cost_vals.append(cost)
                        # Pad weights to ensure at least 3 sources (Res 32, 64, 128)
                        padded_weights = (weights + [0.0]*3)[:3]
                        all_weights.append(padded_weights)
                        
                except Exception:
                    continue 
        
        if epochs:
            results[method_name] = {
                'epochs': np.array(epochs),
                'mse': np.array(mse_vals),
                'loss': np.array(loss_vals),
                'cost': np.array(cost_vals),
                'weights': np.array(all_weights).T
            }
            
    return results

def plot_convergence(results, data_key, label_name, title, filename):
    """General plotting function for convergence metrics (MSE/Loss)."""
    plt.figure(figsize=(7, 5))
    for name, data in results.items():
        plt.plot(data['epochs'], data[data_key], label=name, linewidth=2.5)

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(label_name)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(filename.rsplit('.', 1)[0] + '.pdf')  # vector copy for the paper
    plt.close()

def plot_weights_evolution_grid(results, title, save_path):
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 1, figsize=(9, 1.6*n_methods), sharex=True)
    if n_methods == 1: axes = [axes]

    source_labels = ["Res 32", "Res 64", "Res 128"]

    for ax, (name, data) in zip(axes, results.items()):
        ax.stackplot(data['epochs'], data['weights'], labels=source_labels, alpha=0.8)
        ax.set_title(f"Weight Evolution: {name}")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.margins(x=0)  # no left/right padding; data fills the full width
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.3), ncol=4, frameon=False)
    plt.tight_layout(h_pad=0.6)
    plt.savefig(save_path)
    plt.savefig(save_path.rsplit('.', 1)[0] + '.pdf')  # vector copy for the paper
    plt.close()

def main():
    # Create plot directory
    save_dir = './plots/'
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Read the log file
        with open('very_new/1d/results/multi_resolution/3.0.txt', 'r') as f:
            raw_data = f.read()
        
        print("📖 Parsing log file...")
        results = parse_all_methods_da(raw_data)
        # Enforce a consistent method ordering so the 1D and 2D figures (and
        # their legends) match regardless of the order methods appear in the log.
        ORDER = ["Softmax", "Lagrangian", "ImpLagrangian", "AugLag (Routing)", "ADMM (Routing)"]
        results = {**{k: results[k] for k in ORDER if k in results},
                   **{k: v for k, v in results.items() if k not in ORDER}}
        #print(results)
        if results:
            print(f"✅ Successfully parsed methods: {list(results.keys())}")
            for name, data in results.items():
                print(f"   {name}: {len(data['epochs'])} epochs, "
                      f"MSE range: {data['mse'].min():.4f} to {data['mse'].max():.4f}")
            
            # Plot all comparisons
            plot_convergence(results, 'mse', "MSE", "1D Multi-Resolution Optimization", save_dir + "/mse.png")
            plot_convergence(results, 'loss', "Loss", "1D Multi-Resolution Optimization", save_dir + "/loss.png")
            plot_weights_evolution_grid(results, "1D Multi-Resolution Evolution", save_dir + "/weights.png")

        else:
            print("❌ No data found. Check regex patterns.")
            
    except FileNotFoundError:
        print("Error: File '1d_da.txt' not found in current directory!")
        print("Please make sure the file exists or update the filename in the script.")
    except Exception as e:
        print(f"Error during parsing or plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()