"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Define the input and output sizes
INPUT_SIZE = 2
OUTPUT_SIZE = 1

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Create the first layer with the specified input and hidden layer sizes
        self.layer1 = Linear(INPUT_SIZE, hidden_layers)

        # Create the second layer with the specified hidden layer size
        self.layer2 = Linear(hidden_layers, hidden_layers)

        # Create the third layer with the specified hidden and output layer sizes
        self.layer3 = Linear(hidden_layers, OUTPUT_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (N, D) where N is the number of samples and D is the number of features.

        Returns:
            Tensor: Output tensor of shape (N, 1) representing the predicted probabilities.
        """
        # Forward pass through the first layer
        middle = self.layer1.forward(x).relu()

        # Forward pass through the second layer
        end = self.layer2.forward(middle).relu()

        # Forward pass through the third layer and return the result
        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        # Initialize weights as a parameter with shape (in_size, out_size)
        self.weights = self.add_parameter("weights", RParam(in_size, out_size).value)

        # Initialize bias as a parameter with shape (out_size,)
        self.bias = self.add_parameter("bias", RParam(out_size).value)

    def forward(self, inputs: Tensor) -> minitorch.Tensor:
        """Forward pass through the linear layer.

        Args:
            inputs (Tensor): Input tensor of shape (N, D) where N is the number of samples and D is the number of features.

        Returns:
            Tensor: Output tensor of shape (N, out_size) representing the transformed inputs.
        """
        # Reshape inputs to match weight dimensions (adds a dimension to the end)
        reshaped_inputs = inputs.view(*inputs.shape, 1)

        # Perform element-wise multiplication to get the weighted inputs
        weighted_inputs = reshaped_inputs * self.weights.value

        # Sum the weighted inputs along the input dimension
        weighted_sum = weighted_inputs.sum(1)

        # Reshape the result to match the expected output shape
        output = weighted_sum.view(inputs.shape[0], self.weights.value.shape[1])

        # Add bias to get the final output
        final_output = output + self.bias.value

        # Return the final output
        return final_output


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
