import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from perceptron import Perceptron


def main():
    X = [[0,0],[0,1],[1,0],[1,1]]

    gates = {
        "AND" : [0,0,0,1],
        "OR" : [0,1,1,1],
        "XOR" : [0,1,1,0]
    }
    
   # learning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    learning_rates = [0.9]


    for gate_name in ["AND", "OR"]:
        Y = gates[gate_name]

        LR_results = {}
        
        for learning_rate in learning_rates:

            percept = Perceptron(learning_rate = learning_rate, max_epoch = 200)
            perceptron_results = percept.fit(X, Y)

            LR_results[learning_rate] = perceptron_results
        
        print("=" * 80)
        header = f"|{'LR':<2} | {'Converged':<2} | {'Epochs':<1} | {'Updates':<1} | {'Weights':<14} | {'Bias':<6} |"
        print(header)
        print("=" * 80)

        for lr, res in LR_results.items():
            converged = str(res['converged'])
            epochs = res['epochs']
            updates = res['updates']
            w_str = f"[{res['weights'][0]:.2f}, {res['weights'][1]:.2f}]"
            bias = res['bias']
            
            row = f"| {lr:<2.1f} | {converged:<7} | {epochs:<6} | {updates:<7} | {w_str:<14} | {bias:<6.2f} |"
            print(row)
        break;
            

if __name__ == "__main__":
    main()
    

    
