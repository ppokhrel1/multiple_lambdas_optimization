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
    results = {}
    # Split by the header: >>> Training Method (Params) | Budget: X <<<
    sections = re.split(r'>>> Training (.*?) <<<', full_log)
    
    for i in range(1, len(sections), 2):
        # Clean method name (e.g., "Lagrangian (Top-10 1D) | Budget: 1.5")
        method_name = sections[i].split('|')[0].strip()
        
        log_content = sections[i+1]
        epochs, mse_vals, loss_vals, cost_vals, all_weights = [], [], [], [], []
        
        lines = log_content.strip().split('\n')
        for line in lines:
            if re.search(r'Ep\s+\d+', line):
                try:
                    # 1. Extract epoch number
                    epoch_match = re.search(r'Ep\s+(\d+)', line)
                    epoch = int(epoch_match.group(1))
                    
                    # 2. Extract Test MSE (Ts value inside brackets)
                    mse_match = re.search(r'MSE\s*\[.*?Ts:([\d\.e+-]+)\]', line)
                    mse_val = float(mse_match.group(1)) if mse_match else None

                    # 3. Extract Test Loss (Ts value inside brackets)
                    loss_match = re.search(r'LOSS\s*\[.*?Ts:([\d\.e+-]+)\]', line)
                    loss_val = float(loss_match.group(1)) if loss_match else mse_val

                    # 4. Extract Cost (Current/Budget)
                    cost_match = re.search(r'Cost:\s*([\d\.e+-]+)/', line)
                    cost = float(cost_match.group(1)) if cost_match else 0.0
                    
                    # 5. Extract Weights from "Top Usage" (S0:val, S1:val...)
                    # We assume up to 10 sources are shown in Top Usage
                    weights = [0.0] * 10 
                    weight_pairs = re.findall(r'S(\d+):([\d\.e+-]+)', line)
                    if weight_pairs:
                        for s_idx, s_val in weight_pairs:
                            idx = int(s_idx)
                            # In your log, indices like S59 exist, but let's 
                            # capture the first 10 encountered or map to a fixed size
                            if len(all_weights) == 0: # Initialize dynamic size if needed
                                pass 
                        
                        # Simplified for plotting: capture the values of the first 4 sources mentioned
                        # or specifically S0, S1, S2, S3 if they exist.
                        current_weights = [0.0] * 4
                        for s_idx, s_val in weight_pairs:
                            idx = int(s_idx)
                            if idx < 4:
                                current_weights[idx] = float(s_val)
                        weights = current_weights

                    if mse_val is not None:
                        epochs.append(epoch)
                        mse_vals.append(mse_val)
                        loss_vals.append(loss_val)
                        cost_vals.append(cost)
                        all_weights.append(weights)
                        
                except Exception as e:
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

    source_labels = [f"Source {i}" for i in range(4)]

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
        with open('very_new/1d/results/large_scale/3.0.txt', 'r') as f:
            raw_data = f.read()
        
        print("📖 Parsing log file...")
        results = parse_all_methods_da(raw_data)
        # Drop the redundant configuration suffix from method names; it's the
        # same for every method and only clutters the titles/legend.
        results = {re.sub(r'\s*\(Top-10[^)]*\)$', '', k): v for k, v in results.items()}
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
            plot_convergence(results, 'mse', "MSE", "1D Large Scale Multi-Source Optimization", save_dir + "/mse.png")
            plot_convergence(results, 'loss', "Loss", "1D Large Scale Multi-Source Optimization", save_dir + "/loss.png")
            plot_weights_evolution_grid(results, "1D Large Scale Multi-Source Optimization Weights Evolution", save_dir + "/weights.png")

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