import numpy as np

class MLP:
    def __init__(self, learning_rate, optimizer, n_neurons, batch_size, epochs, task_type, cost_calc):
        self.lr = learning_rate
        self.neurons = [0] + n_neurons
        self.hidden = len(n_neurons)
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.cost_calc = cost_calc

        self.biases = None
        self.weights = None
        
        self.activ = None
        self.activ_derivative = None
        self.training_cost = []

        self.task_type = task_type

    def sigmoid(self, zval):
        zval = np.clip(zval, -500, 500)
        return 1.0 / (1.0 + np.exp(-zval))
    
    def relu(self, zval):
        return np.maximum(0, zval)
        
    def tanh(self, zval):
        return np.tanh(zval)
    
    def linear(self, zval):
        return zval
    
    def softmax(self, zval):
        exp_z = np.exp(zval - np.max(zval, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def sigmoid_derivative(self, zval):
        sig = self.sigmoid(zval)
        return sig * (1 - sig)

    def relu_derivative(self, zval):
        return np.where(zval > 0, 1.0, 0.0)

    def tanh_derivative(self, zval):
        return 1.0 - np.tanh(zval)**2
    
    def linear_derivative(self, zval):
        return np.ones_like(zval)

    def activation(self, func):
        if func == "sigmoid":
            self.activ = self.sigmoid
            self.activ_derivative = self.sigmoid_derivative
        elif func == "relu":
            self.activ = self.relu
            self.activ_derivative = self.relu_derivative
        elif func == "tanh":
            self.activ = self.tanh
            self.activ_derivative = self.tanh_derivative
        elif func == "linear":
            self.activ = self.linear
            self.activ_derivative = self.linear_derivative
        else:
            raise ValueError("Activation function must be 'sigmoid', 'relu', or 'tanh'!")

    def forward_prop(self, X):
        activations = [X]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(self.activ(z))
        
        # Use softmax for classification in the output layer
        if self.task_type == "classification":
            activations[-1] = self.softmax(zs[-1])
        
        return activations, zs

    def back_prop(self, X, y, activations, zs):
        m = X.shape[1]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Compute error for output layer
        if self.cost_calc == "bce":
            delta = activations[-1] - y  # Cross-entropy loss derivative
        elif self.cost_calc == "mse":
            delta = (activations[-1] - y) * self.activ_derivative(zs[-1])  # MSE loss derivative
        
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        # Propagate error backward
        for l in range(2, len(self.neurons)):
            delta = np.dot(self.weights[-l+1].T, delta) * self.activ_derivative(zs[-l])
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            
        return nabla_b, nabla_w

    def update_weights(self, nabla_b, nabla_w, batch_size):
        self.weights = [w - (self.lr / batch_size) * nw 
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.lr / batch_size) * nb 
                      for b, nb in zip(self.biases, nabla_b)]

    def fit(self, train_X, train_y, activ, val_X=None, val_y=None, patience=10, tolerance=1e-4):
        self.activation(activ)
        m = train_X.shape[1]
        self.neurons[0] = train_X.shape[0]

        if train_y.ndim > 1:
            self.neurons.append(train_y.shape[0])
        else:
            self.neurons.append(1)

        self.biases = [np.random.randn(n, 1) * 0.1 for n in self.neurons[1:]]
        self.weights = [np.random.randn(n2, n1) * np.sqrt(1. / n1) for n1, n2 in zip(self.neurons[:-1], self.neurons[1:])]
        self.training_cost = []
        if self.optimizer == "mgd":
            n_batches = m // self.batch_size
        elif self.optimizer == "sgd":
            n_batches = 1
        elif self.optimizer == "bgd":
            n_batches = self.batch_size
        else:
            raise ValueError("Optimizer must be one of the defined types.")
        best_val_cost = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            train_X_shuffled = train_X[:, indices]

            if self.task_type == "classification":
                train_y_shuffled = train_y[:, indices]
            else:
                train_y_shuffled = train_y[:, indices] if train_y.ndim > 1 else train_y[indices]

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_X = train_X_shuffled[:, start_idx:end_idx]
                
                if self.task_type == "classification":
                    batch_y = train_y_shuffled[:, start_idx:end_idx]
                else:
                    batch_y = train_y_shuffled[:, start_idx:end_idx] if train_y.ndim > 1 else train_y_shuffled[start_idx:end_idx]

                activations, zs = self.forward_prop(batch_X)
                nabla_b, nabla_w = self.back_prop(batch_X, batch_y, activations, zs)
                self.update_weights(nabla_b, nabla_w, self.batch_size)

            final_activations, _ = self.forward_prop(train_X)
            cost = self.compute_cost(final_activations[-1], train_y)
            self.training_cost.append(cost)

            # Validation Cost Calculation - early stopping (implemented with the help of LLM services)
            if val_X is not None and val_y is not None:
                val_activations, _ = self.forward_prop(val_X)
                val_cost = self.compute_cost(val_activations[-1], val_y)
                print(f"Epoch {epoch+1}, Training Cost: {cost:.6f}, Validation Cost: {val_cost:.6f}")
                
                # Early Stopping Logic
                if val_cost < best_val_cost - tolerance:
                    best_val_cost = val_cost
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping on epoch {epoch+1}")
                    break
            # else:
            #     print(f"Epoch {epoch+1}, Training Cost: {cost:.6f}")


    def predict(self, X):
        activations, _ = self.forward_prop(X)
        return activations[-1]

    def compute_cost(self, output_activations, y):
        if self.cost_calc == "bce":
            return -np.mean(np.sum(y * np.log(output_activations + 1e-9), axis=0))
        elif self.cost_calc == "mse":
            return np.mean(np.square(output_activations - y)) / 2
    
    def gradient_checking(self, X, y, epsilon=1e-7): # implemented with some help from LLM services
        activations, zs = self.forward_prop(X)
        nabla_b, nabla_w = self.back_prop(X, y, activations, zs)

        numerical_w_grads = []
        for l in range(len(self.weights)):
            grad = np.zeros_like(self.weights[l])
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    # Positive perturbation
                    self.weights[l][i, j] += epsilon
                    cost_pos = self.compute_cost(self.forward_prop(X)[0][-1], y)
                    self.weights[l][i, j] -= 2 * epsilon
                    cost_neg = self.compute_cost(self.forward_prop(X)[0][-1], y)
                    self.weights[l][i, j] += epsilon
                    grad[i, j] = (cost_pos - cost_neg) / (2 * epsilon)
            numerical_w_grads.append(grad)

        # Compare numerical and analytical gradients
        for l, (num_grad, anal_grad) in enumerate(zip(numerical_w_grads, nabla_w)):
            diff = np.linalg.norm(num_grad - anal_grad) / (np.linalg.norm(num_grad) + np.linalg.norm(anal_grad))
            if diff > 1e-5:
                print(f"Gradient Check Failed at layer {l}. Difference: {diff:.6f}")
            else:
                print(f"Gradient Check Passed at layer {l}. Difference: {diff:.6f}")

