import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from perceptron import Perceptron

def calculate_margin(X, weights, bias):
    weight_arr = np.array(weights)
    normalized_weights = np.linalg.norm(weight_arr)

    # for learning_rate == 0.0
    if normalized_weights == 0:
        return 0.0

    distances = []
    for x in X:
        x_arr = np.array(x)

        distance = (np.dot(weight_arr,x_arr) + bias)/normalized_weights
        distances.append(abs(distance))

    return min(distances)

    
def plot_overlay_hyperplanes(X, Y, gate_name, LR_results):
    X = np.array(X)
    Y = np.array(Y)
    
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label="Class 0", color='blue', s=80, zorder=5)
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], label="Class 1", color='orange', s=80, zorder=5)
    
    learning_rates = sorted(list(LR_results.keys()))
    norm = mcolors.Normalize(vmin=min(learning_rates), vmax=max(learning_rates))
    cmap = cm.viridis
    
    x1_vals = np.linspace(-1.5, 2.5, 100)
    
    for learning_rate in learning_rates:
        res = LR_results[learning_rate]
        
        w1 = res['weights'][0]
        w2 = res['weights'][1]
        b = res['bias']
        
        
        color = cmap(norm(learning_rate))
        
        if w2 != 0:
            x2_vals = (-w1 * x1_vals - b) / w2
            ax.plot(x1_vals, x2_vals, color=color, linewidth=2, alpha=0.8)
        elif w1 != 0:
            x1_fixed = -b / w1
            ax.axvline(x1_fixed, color=color, linewidth=2, alpha=0.8)
            
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'Learning Rate ($\alpha$)')
    
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(f'Overlay of Final Hyperplanes for {gate_name} Gate Across Learning Rates')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    folder_name = f"Outputs/{gate_name}"
    file_path = os.path.join(folder_name, f"hyper_plane_overlay.png")
    plt.savefig(file_path)
    
    plt.show()
    



def main():
    X = [[0,0],[0,1],[1,0],[1,1]]

    gates = {
        "AND" : [0,0,0,1],
        "OR" : [0,1,1,1],
        "XOR" : [0,1,1,0]
    }
    
   # learning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    learning_rates = [0.9,1.0]


    for gate_name in ["AND", "OR"]:
        Y = gates[gate_name]

       

        LR_results = {}
        
        for learning_rate in learning_rates:

            percept = Perceptron(gate_name, learning_rate, max_epoch = 200)
            perceptron_results = percept.fit(X, Y)

            margin_value = calculate_margin(X,
                                            perceptron_results['weights'],
                                            perceptron_results['bias'])
            perceptron_results['margin'] = margin_value
            
            LR_results[learning_rate] = perceptron_results

        print("=" * 80)
        print(gate_name+' Gate summery')
        print("=" * 80)
        header = f"|{'LR':<2} | {'Converged':<2} | {'Epochs':<1} | {'Updates':<1} | {'Weights':<14} | {'Bias':<6} | {'Margin':<2}|"
        print(header)
        print("=" * 80)

        for lr, res in LR_results.items():
            converged = str(res['converged'])
            epochs = res['epochs']
            updates = res['updates']
            w_str = f"[{res['weights'][0]:.2f}, {res['weights'][1]:.2f}]"
            bias = res['bias']
            margin_value = res['margin']
            
            row = f"| {lr:<2.1f} | {converged:<7} | {epochs:<6} | {updates:<7} | {w_str:<14} | {bias:<6.2f} | {margin_value:<6.2f}|"
            print(row)

        plot_overlay_hyperplanes(X, Y, gate_name, LR_results)
        break;
            

if __name__ == "__main__":
    main()
    

    
