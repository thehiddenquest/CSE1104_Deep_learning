import copy
import os
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, gate_name = "Unknown", learning_rate = 1, initial_weights = [1.0, -1.0], initial_bias = 0.5, max_epoch = 10):
        self.gate_name = gate_name
        self.learning_rate = learning_rate
        self.weights = list(initial_weights)
        self.bias = initial_bias
        self.weight_update_counter = 0
        self.max_epoch = max_epoch
        self.boundaries = []
        
    def __activation_function(self, val):
        return 1 if val >= 0 else 0
    
    def _ploting(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        valid_boundaries = [b for b in self.boundaries]
        n = len(valid_boundaries)
        if n == 0:
            return

        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        for i, boundary in enumerate(valid_boundaries):
            ax = axes[i]
            w1 = boundary["w1"]
            w2 = boundary["w2"]
            b = boundary["b"]

            ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label="Class 0", color='blue', alpha=0.7)
            ax.scatter(X[Y == 1, 0], X[Y == 1, 1], label="Class 1", color='orange', alpha=0.7)

            if w2 != 0:
                x1 = np.linspace(-2, 2, 100)
                x2 = (-w1 * x1 - b) / w2
                ax.plot(x1, x2, color='r', label='Decision Boundary')
            elif w1 != 0:
                x1_val = -b / w1
                ax.axvline(x1_val, color='r', label='Decision Boundary')

            ax.quiver(0, 0, w1, w2, angles='xy', scale_units='xy', scale=1, color='green')
            ax.plot([], [], color='green', marker='>', markersize=8, label='W vector')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            
            if boundary.get('x') is None:
                ax.set_title(f'Final (Epoch {boundary.get("epoch")})', fontsize=10)
            else:
                ax.set_title(f'Update {i+1} (Epoch {boundary.get("epoch")})', fontsize=10)

            ax.grid(True)

        for j in range(n, len(axes)):
            fig.delaxes(axes[j])
        
        handles, labels = axes[0].get_legend_handles_labels()

        
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 1.0))

        plt.subplots_adjust(hspace=0.9)

        folder_name = f"Outputs/{self.gate_name}"
        os.makedirs(folder_name, exist_ok = True)
        file_path = os.path.join(folder_name, f"lr_{self.learning_rate}.png")
        plt.savefig(file_path)
        plt.show()
        
    def fit(self, input_arr=None, output_arr=None):
        if not (input_arr or output_arr):
            print("Please enter valid input")
            return None
        
        X = input_arr
        Y = output_arr
        
        update_vector = []
        update_vector.append((tuple(self.weights), self.bias))
        cycle_detected = False
        cycle_period = 0
        
        for epoch in range(1, self.max_epoch + 1):
            print('='*80)
            print(f'Epoch : {epoch}, learning_rate : {self.learning_rate}')
            print('='*80)
            header = f"{'X1':<2} | {'X2':<2} | {'Ac':<1} | {'Pred':<1} | {'Err':<2} | {'W_old':<14} | {'b_old':<6} | {'W_new':<14} | {'b_new':<6}"
            print(header)
            print("-" * 80)
            weight_update = False

            for x, y in zip(input_arr, output_arr):
                z = sum(xi * w for xi, w in zip(x, self.weights)) + self.bias
                y1 = self.__activation_function(z)
                error_term = y - y1

                old_weights = copy.deepcopy(self.weights)
                old_bias = copy.deepcopy(self.bias)

                if error_term != 0:
                    weight_update = True
                    self.weight_update_counter += 1
                    self.weights = [w + self.learning_rate * error_term * xi for w, xi in zip(self.weights, x)]
                    self.bias += self.learning_rate * error_term

                    current_update = (tuple(self.weights), self.bias)
                    
                    if current_update in update_vector and not cycle_detected:
                        cycle_start = update_vector.index(current_update)
                        cycle_period = len(update_vector) - cycle_start
                        print(f"      -> [Cycle Detected] Period: {cycle_period} updates")
                        cycle_detected = True
                        
                        
                    update_vector.append(current_update)
                    self.boundaries.append({
                        'w1': self.weights[0],
                        'w2': self.weights[1],
                        'b': self.bias,
                        'epoch': epoch,
                        'x': x
                    })

                
                old_w_str = f"[{', '.join(f'{w:.2f}' for w in old_weights)}]"
                new_w_str = f"[{', '.join(f'{w:.2f}' for w in self.weights)}]"
                row = f"{x[0]:<2} | {x[1]:<2} | {y:<2} | {y1:<4} | {(y-y1):<3} | {old_w_str} {'':<3} | {old_bias:<6.2f} | {new_w_str} {'':<3} | {self.bias:<6.2f}"
                print(row)
                print('.'*80)

            if not weight_update:
                break
        
        if len(self.boundaries) > 0 and self.learning_rate > 0:
            self._ploting(X, Y)
        
        return {
            'converged': not weight_update,
            'weights': self.weights,
            'bias': self.bias,
            'epochs': epoch,
            'updates': self.weight_update_counter,
            'boundaries': self.boundaries,
            'cycle_detected': cycle_detected,
            'cycle_period': cycle_period,
            'update_history': update_vector
        }
if __name__ == '__main__':
    p = Perceptron("TEST AND")
    X = [[0,0],[0,1],[1,0],[1,1]]
    and_output = [0,0,0,1]
    results = p.fit(X,and_output)
    if results:
        print(results)
