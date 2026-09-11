import matplotlib.pyplot as plt
import numpy as np

def plot_activation_function(acfunc, dacfunc, x, name, discontinuity = None, kink = None):
    fig,(ax1,ax2) = plt.subplots(2, 1, figsize=(8,6))
    fig.subplots_adjust(hspace = 0.3)

    # Activation function
    ax1.plot(x, acfunc, color='b', linestyle='-', label=f'{name} function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True)
    ax1.legend()

    
    # Derivative
    ax2.plot(x, dacfunc, color='r', linestyle='-', label=f'{name} derivative')
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.grid(True)
    ax2.legend()

    if discontinuity:
        for x_val, y_eval, y_ival in discontinuity:
            
            ax1.plot(x_val, y_eval, 'wo', markeredgecolor = 'b')
            ax1.plot(x_val, y_ival, 'bo', markeredgecolor = 'k')

            ax2.plot(x_val, y_eval, 'wo', markeredgecolor = 'r')
    
    plt.show()
    
if __name__ == '__main__':
    x = np.linspace(-8, 8, 2000)

    # Heavy-side step function
    f1 = [ 1 if i>=0 else 0 for i in x ]

    df1 = np.zeros_like(x)
    
    plot_activation_function(f1, df1, x, "heavyside-step", [(0,0,1)])

    # Piecewise-linear function

    f2 = [0 if i < -1 else (1/2)*(i+1) if -1<=i<=1 else 1 for i in x]

    df2 = [(1/2) if -1<i<1 else 0 for i in x] 

    plot_activation_function(f2, df2, x, "Piecewise-Linear")

    # Sigmoid function

    f3 = [(1/(1+np.exp(-i))) for i in x]
    df3 = [(1/(1+np.exp(-i)))*(1-(1/(1+np.exp(-i)))) for i in x]

    plot_activation_function(f3, df3, x, "Sigmoid")

    # Hyperbolic tangent

    f4 = [(np.exp(i)-np.exp(-i))/(np.exp(i)+np.exp(-i)) for i in x]
    df4 = [1-((np.exp(i)-np.exp(-i))/(np.exp(i)+np.exp(-i)))**2 for i in x]

    plot_activation_function(f4, df4, x, "Tanh")
    
    # Relu

    f5 = [max(i,0) for i in x]
    df5 =[ 1 if i > 0 else 0  for i in x]

    plot_activation_function(f5, df5, x, "Relu")
