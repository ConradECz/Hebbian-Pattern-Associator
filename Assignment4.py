import numpy as np

class PatternAssociator:
    def __init__(self, input_size, output_size, learning_rate, iterations):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.zeros((output_size, input_size))  # Initialize weights to zero

    def train_network(self, input_patterns, output_patterns):
        for _ in range(self.iterations):
            for inputs, outputs in zip(input_patterns, output_patterns):
                self.weights += self.learning_rate * np.outer(outputs, inputs)

    def predict(self, inputs):
        return np.sign(np.dot(self.weights, inputs))

    def show_weights(self):
        print("\nFinal weight matrix:")
        print(self.weights)

def run_experiment(part=1):    
    if part == 1:
        input_data = np.array([
            [1, -1, -1],  
            [-1, 1, -1],  
            [-1, -1, 1],  
            [1, 1, 1]      
        ])

        output_data = np.array([
            [-1],  
            [-1],  
            [1],   
            [1]    
        ])
    else:
        input_data = np.array([
            [1, 1, 0],  
            [1, -1, 1],  
            [0, 1, 2]    
        ])

        output_data = np.array([
            [2],  
            [1],  
            [3]    
        ])

    # Network parameters
    input_size = input_data.shape[1]
    output_size = output_data.shape[1]
    learning_rate = 0.1  
    iterations = 10  

    network = PatternAssociator(input_size, output_size, learning_rate, iterations)

    # Train the network with the selected dataset
    network.train_network(input_data, output_data)

    # Display learned weight matrix
    network.show_weights()

    # Define test cases
    if part == 1:
        test_cases = np.array([
            [0, 0, 0],  
            [-2, -2, 0],  
            [0, 0, 2],  
            [3, -1, -3]  
        ])
    else:
        test_cases = np.array([
            [1, 0, 1],  
            [2, -1, 1],  
            [0, 1, 3]  
        ])

    print("\nEvaluating the network with test cases:")
    for idx, test_case in enumerate(test_cases, 1):
        result = network.predict(test_case)
        print(f"Test {idx}: Input {test_case} → Output {result}")

if __name__ == "__main__":
    # Choose which part to run (1 or 2)
    selected_part = int(input("Enter 1 for Part 1, or 2 for Part 2: "))
    run_experiment(selected_part)