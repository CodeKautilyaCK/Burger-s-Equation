
class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving 1-D PDEs like Burgers' equation.

    Inputs:  x, t (space and time)
    Output:  u(x,t) (dependent variable, e.g., velocity)
    """

    def __init__(self, layers):
        """
        Initialize a fully-connected feedforward neural network.

        Args:
            layers (list): Number of neurons per layer including input and output.
                           Example: [2, 64, 64, 64, 1] 
                           2 inputs -> 3 hidden layers with 64 neurons -> 1 output
        """
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        """
        Forward pass through the network.

        - Uses tanh activations for hidden layers for smooth derivatives.
        - Output layer is linear for regression of PDE solution.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2] (x and t)

        Returns:
            torch.Tensor: Predicted solution u(x,t)
        """
        for layer in self.net[:-1]:
            x = torch.tanh(layer(x))
        return self.net[-1](x)
