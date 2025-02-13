import numpy as np
import matplotlib.pyplot as plt
import pickle

class LinearRegression:

    def __init__(self, degree, iterations, learning_rate, points, lamb = 0):
        self.degree = degree
        self.coefficients = np.ones(self.degree + 1)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.points = points
        self.independent = None
        self.dependent = None
        self.lamb = lamb

    def make_vector(self, val):
        x = np.array([val**i for i in range(self.degree + 1)])
        return x[::-1]

    def gradient_descent(self, x_matrix, errors):
        gradients = np.dot(x_matrix.T, errors) * 2 / self.points
        return self.coefficients + self.learning_rate * gradients

    def fit(self, x, y, regularisation=0):
        self.independent = x
        self.dependent = y


        # To make the plots for the animations, LLMs were used as I did not know how to use subplots in plt.
        # Create the figure and axes objects once, outside the loop
        # fig, ax = plt.subplots()

        for i in range(self.iterations):
            x_matrix = np.array([self.make_vector(xi) for xi in x])
            y_pred = np.dot(x_matrix, self.coefficients)
            error = y - y_pred
            if regularisation == 0:
                self.coefficients = self.gradient_descent(x_matrix, error)
            elif regularisation == 1:
                self.coefficients = self.l1(x_matrix, error)
            elif regularisation == 2:
                self.coefficients = self.l2(x_matrix, error)

            # mse = self.mse(x, y)
            # std = self.std(x, y)
            # var = self.variance(x, y)

            # Clear the previous plot
            # ax.clear()

            # # Set the title and labels
            # ax.set_title(f"Plot for degree = {self.degree} and iteration {i}")
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            

            # # Plot the data and regression line
            # ax.scatter(x, y, color='deepskyblue', s=7, label="datapoints")
            # line, = ax.plot(x_sorted, y_pred_sorted, color='red', label="regression line", zorder = 2)

            # for xi, yi, y_predi in zip(x, y, y_pred):
            #     ax.plot([xi, xi], [yi, y_predi], color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)

            # ax.legend()

            # ax.text(0.98, 0.95, f"Variance: {var: .3f}\nMSE: {mse: .3f}\n Standard Deviation: {std: .3f}", 
            # transform=ax.transAxes, ha='right', va='top',
            # bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

            # # Save the figure
            # if i < 10:
            #     plt.savefig(f"../../assignments/1/figures/k{self.degree}/iteration0{i}.png")
            # else:
            #     plt.savefig(f"../../assignments/1/figures/k{self.degree}/iteration{i}.png")

        # # Add the legend after the loop
        # ax.legend()

        # # Save the final plot with the legend
        # plt.savefig(f"../../assignments/1/figures/k{self.degree}/final_plot.png")

        # # Close the figure to free up memory
        # plt.close(fig)

        return np.dot(x_matrix, self.coefficients)

        
        

    def predict(self, x):
        x_vector = self.make_vector(x)
        y_pred = np.sum(np.multiply(x_vector, self.coefficients))
        return y_pred
    
    def mse(self, x, y, regularisation = 0):
        x_matrix = np.array([self.make_vector(xi) for xi in x])
        y_pred = np.dot(x_matrix, self.coefficients)
        error = y - y_pred
        mse = np.mean(error**2)
        if regularisation == 1:
            mse = mse + self.lamb*np.sum(np.abs(self.coefficients))/self.points
        if regularisation == 2:
            mse = mse + self.lamb*np.sum(self.coefficients**2)/self.points
        return mse

    def variance(self, x, y):
        x_matrix = np.array([self.make_vector(xi) for xi in x])
        y_pred = np.dot(x_matrix, self.coefficients)
        error = y - y_pred
        return np.var(error)
    
    def std(self, x, y):
        var = self.variance(x, y)
        return np.sqrt(var)

    def l2(self, x_matrix, errors):
        grad = self.gradient_descent(x_matrix, errors)
        return grad + self.coefficients*self.lamb*2/self.points

    def l1(self, x_matrix, errors):
        grad = self.gradient_descent(x_matrix, errors)
        return grad + self.lamb*2*np.sign(self.coefficients)/self.points
    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)


