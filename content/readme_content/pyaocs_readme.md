### README

## Satellite Digital Twin

This repository contains a Satellite Digital Twin to test control strategies for satellite systems. The digital twin simulates the dynamics of a satellite in a given environment, allowing the testing of control algorithms and strategies.

### Installation

You can install the package using pip:

```bash
pip install pyaocs
```

### Usage

To use the code, you can create a control strategy by creating a class that defines the `compute` method. Here is an example of a simple test control strategy:

```python
import numpy as np

from pyaocs import parameters as param
from pyaocs.simulation import digital_twin

class SimpleTestControl():

    def __init__(self):
        self.firing_time = 1 #s

        self.dt = 1 / param.sample_rate

        self.total_steps = self.firing_time / self.dt

        self.step = 0

    def compute(self, obs):

        F1 = 0
        F2 = 0
        γ1 = 0
        γ2 = 0
        δ1 = 0
        δ2 = 0

        if (self.step + 1) <= self.total_steps:
            F1 = 1

        action = np.array([F1, F2, γ1, γ2, δ1, δ2])

        self.step += 1

        return action
    
    def plot(self):
        pass
    

if __name__ == "__main__":

    strategy = SimpleTestControl()

    digital_twin.run(strategy, 
                     render=True, 
                     real_time=False, 
                     use_disturbances=False,
                     noise=False)
```

This code creates a simple test control strategy and runs the simulation using the digital twin. You can modify the control strategy and parameters as needed for your testing.

### Folder Structure

- **pyaocs/control/**: Contains control strategies for the digital twin.
- **pyaocs/parameters.py**: Contains parameters for the satellite system.
- **pyaocs/simulation/**: Contains the core simulation code for the digital twin.
- **pyaocs/urdf_models/**: Contains URDF model files for visualization.
- **pyaocs/utils.py**: Contains utility functions for the digital twin.
- **tests/example.py**: Contains a sample test file for the digital twin.

### Author

- Pedro [@pr1u20](mailto:pr1u20@soton.ac.uk)

### Bug Reporting

If you encounter any issues or have suggestions for improvement, please visit the [GitHub repository](https://github.com/pr1u20/AOCS-digital-twin) and create an issue.

### License

This project is licensed under the MIT License.