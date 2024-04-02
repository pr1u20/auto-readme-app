### Content of README.md:

```python
Satellite Digital Twin

```

### Content of pyaocs/control/example.py:

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

### Content of pyaocs/parameters.py:

```python
start_position = [0, 0, 1] # define the cube start position
start_velocity = [0, 0, 0]
start_orientation = [0.0, 0.0, 0.0, 1.0] # define start orientation [0.0, 0.0, 0.4214104, 0.90687 ]
start_angular_velocity = [0, 0, 0] # in terms of euler angles

target_position = [2,2,1] # Target position where we want the satellite to move.
target_orientation = [0,0,0]  # [ψ, θ, φ]


thruster_positions = [[-0.15,0,0], 
                      [0.15,0,0]]

F = 0.26 # force applied by each thruster in Newtons. F/m ratio 1/20
L = abs(thruster_positions[0][0]) # m
m = 10 # kg
Ix = m * L**2  # 0.05 kg m^2
Iy = m * L**2 # 0.05 kg m^2
Iz = 0.15 # 0.1 kg m^2

I = Ix, Iy, Iz

fps = 240 # Hz. # Frames per second of the simulation. Mantain to 240 to mantain accuracy.
sample_rate = 24 # Hz. Rate at which the control algorithms reads position information.
step_rate = int(fps / sample_rate)
dt = 1 / fps
thrust_vector_range = 16 # degrees
discrete_angle = thrust_vector_range

simulation_time = 50 # s
actuation_speed = 240 # degree/s. Assuming linear, but could be some complex function dependent on current angles.
thrust_delay = 0.04 #3e-4 #30e-3 # s
thrust_speed = F / thrust_delay #N / s  How fast the thrust can be increased to desired value. We will assume it to be linear increase from 0 to desired force (1N) in the thrust delay time (30 ms)
discrete_thurster = 24 # the levels of thrust we can get 

# Orbit Fab uses rotation_error of 0.01 deg, rotation_drift of 0.1 deg/s and translation_error of 0.01 m.
translation_error = 0.05    # m
translation_drift = 0.005   # m/s
rotation_error = 0.1       # degree
rotation_drift = 0.01       # degree/s
max_v = 0.3 # m/s
```

### Content of pyaocs/simulation/actuator.py:

```python
import numpy as np

from pyaocs import parameters as param


class ActuatorSimulation():
    """Class used to change the 6 parameters of the satellite F1, F2, γ1, γ2, δ1, δ2. Details of the variables and order are specified in the Guidelines 55 Requirements document.
    Would not be neccessary if we assumed instaneous changes, but there are delays, limitations in actuation speed, etc.
    For now we will assume linear changes, in the future we should have more complex estimations.
    This is external from the Satellite Env class as this will be an input from a different work package, where they could specify the exact relationship between torque and angle change, the actual change in thrust"""

    def __init__(self, dt = param.dt):

        self.initialize()

        self.actuation_speed = param.actuation_speed # Rate of change of γ1, γ2, δ1, δ2
        self.thrust_speed = param.thrust_speed # Rate of change of F1, F2

        self.dt = dt

    def initialize(self):

        """In case Reinforcement Learning is used, the initialize function needs to be run after every episode. To reset variables.
        Any variable that changes through time, should be defined here.
        """

        # The six system parameters that can be changed to control satellite.
        self.F1 = 0
        self.F2 = 0
        self.γ1 = 0
        self.γ2 = 0
        self.δ1 = 0
        self.δ2 = 0

    def compute(self, targets, wait_actuator = False):

        """Computes the next position of the satellite parameters given the target value

        :param targets: 6 target values for the state variables of the satellite.
        :param type: np.ndarray
        :return: (state, finished).
            -   state (np.ndarray): the updated six parameters of the satellite.
            -   finished (np.ndarray): whether each of the six variables has reached there target values.
                Consists of a numpy array with 6 bool variables. 
        :rtype: float
        """

        target_F1 = targets[0]
        target_F2 = targets[1]
        target_γ1 = targets[2]
        target_γ2 = targets[3]
        target_δ1 = targets[4]
        target_δ2 = targets[5]

        # Calculate change for each parameter
        self.F1, F1_finshed = self.actuation_change(self.F1, target_F1, self.thrust_speed)
        self.F2, F2_finshed = self.actuation_change(self.F2, target_F2, self.thrust_speed)
        self.γ1, γ1_finshed = self.actuation_change(self.γ1, target_γ1, self.actuation_speed)
        self.γ2, γ2_finshed = self.actuation_change(self.γ2, target_γ2, self.actuation_speed)
        self.δ1, δ1_finshed = self.actuation_change(self.δ1, target_δ1, self.actuation_speed)
        self.δ2, δ2_finshed = self.actuation_change(self.δ2, target_δ2, self.actuation_speed)

        state = np.array([self.F1, self.F2, self.γ1, self.γ2, self.δ1, self.δ2])
        finished = np.array([F1_finshed, F2_finshed, γ1_finshed, γ2_finshed, δ1_finshed, δ2_finshed])

        return state, finished

    def actuation_change(self, value, target_value, rate_of_change):
        
        """Common function used to vary all the parameters. In the future thrust will not changes as the thruster angles, so there will be different functions for each.

        :param value: current values of the system
        :type value: float
        :param target_value: target values for the system
        :type target_value: float
        :param rate_of_change: rate of change of the variable
        :type rate_of_change: float
        :return: (value, finished)
            -   state (np.ndarray): the updated value.
            -   finished (np.ndarray): whether the value is equal to target values. 
        :rtype: float
        """

        #Whether value == target_value, so objective target has been reached
        finished = False

        if value == target_value:
            finished = True
            pass

        elif value < target_value:

            value = value + rate_of_change*self.dt

            if value > target_value:
                value = target_value
                finished = True

        elif value > target_value:

            value = value - rate_of_change*self.dt

            if value < target_value:
                value = target_value
                finished = True

        return value, finished
```

### Content of pyaocs/simulation/digital_twin.py:

```python
import time
import numpy as np
import traceback
from pyaocs import parameters as param
from pyaocs.utils import wait_for_actuator
from pyaocs.simulation.environment import SatelliteEnv
from pyaocs.simulation.actuator import ActuatorSimulation
from pyaocs.simulation.noise import Noise


def run(strategy, render = True, real_time = True, bullet = False, use_disturbances = False, plot = True, trajectory = None, noise = False, wait_actuator = True):

    env = SatelliteEnv(render = render, bullet = bullet, real_time=real_time, use_disturbances=use_disturbances, trajectory=trajectory)
    obs, info = env.reset()

    act = ActuatorSimulation()

    if noise:
        QTM_noise = Noise()

    step_rate = param.step_rate # Steps after which control function is run.

    done = False
    try:
        while not done:
            # Choose an action
            if env._current_step % step_rate == 0:

                t1 = time.time()

                # AOCS control block
                action = strategy.compute(obs) # Gives target values

                time.sleep(0.000001)
                t2 = time.time()

                fps = 1 / (t2 - t1)
                print(f"FPS: {fps:.2f}", end='\r', flush=True)

            # Actuator control block
            state, finished = act.compute(action) # Gives variables values after step change.

            # If actuator has not reached target, do not fire thruster
            if wait_actuator:
                action, state = wait_for_actuator(state, action)

            # Perform the chosen action in the environment
            obs, reward, done, _, info = env.step(state)

            # Add noise to the observation
            if noise:
                obs = QTM_noise.add_noise(obs)

        print(f"FPS: {fps:.2f}")
        env.close()

    except KeyboardInterrupt:
        traceback.print_exc()
        print("An error occurred. Exiting simulation.")

    if plot:
        env.plot_results()
        strategy.plot()

    return env, obs, strategy
```

### Content of pyaocs/simulation/disturbances.py:

```python
import numpy as np
import random
import matplotlib.pyplot as plt

from pyaocs import parameters as param


class ESA_table():
    def __init__():
        pass
    def calculate_force(self):
        pass
    def calculate_torque(self):
        pass

class RandomDistubances():

    def __init__(self):
        pass

    def calculate_force(self):

        max_F = 0.05 # N

        x_min = - max_F
        x_max = max_F
        y_min = - max_F
        y_max = max_F
        z_min = 0
        z_max = 0

        Fx = random.uniform(x_min, x_max)
        Fy = random.uniform(y_min, y_max)
        Fz = random.uniform(z_min, z_max)

        Fd = np.array([Fx, Fy, Fz])

        return Fd

    def calculate_torque(self):

        max_T = 0.05 # Nm

        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        z_min = - max_T
        z_max = max_T

        Tx = random.uniform(x_min, x_max)
        Ty = random.uniform(y_min, y_max)
        Tz = random.uniform(z_min, z_max)

        Td = np.array([Tx, Ty, Tz])

        return Td

class SinusodialDistubances():

    def __init__(self):
        pass

    def calculate_force(self, t):

        period = 20

        max_F = 0.05 # N

        A_x = max_F
        A_y = max_F
        A_z = 0

        period_steps = period / param.dt
        Fx = A_x*np.sin(((2*np.pi) / period_steps) * t)
        Fy = A_y*np.sin(((2*np.pi) / period_steps) * t)
        Fz = A_z*np.sin(((2*np.pi) / period_steps) * t)

        Fd = np.array([Fx, Fy, Fz])

        return Fd

    def calculate_torque(self, t):

        period = 20

        max_T = 0.05 # N

        A_x = 0
        A_y = 0
        A_z = max_T

        period_steps = period / param.dt
        Tx = A_x*np.sin(((2*np.pi) / period_steps) * t)
        Ty = A_y*np.sin(((2*np.pi) / period_steps) * t)
        Tz = A_z*np.sin(((2*np.pi) / period_steps) * t)

        Td = np.array([Tx, Ty, Tz])

        return Td
    

class RandomSineNoise():

    def __init__(self, dt):

        self.dt = dt

        self.t = np.linspace(0, int(param.simulation_time / self.dt)-1, int(param.simulation_time / self.dt))

        self.load_signals()

    def create_noise_signal(self, t):

        num_sines = 10
        period = np.array([np.random.uniform(1, 20) for _ in range(num_sines)])
        period_steps = period / param.dt
        amplitude = period / 20 # amplitude between 0 and 1
        phase = np.array([np.random.uniform(0, 2*np.pi) for _ in range(num_sines)])  # random phase between 0 and 2*pi

        noisy_signal = 0

        # Add sine waves with different frequencies and amplitudes
        for i in range(num_sines):

            
            sine_wave = amplitude[i] * np.sin(((2*np.pi) / period_steps[i]) * t + phase[i])
            noisy_signal += sine_wave

        noisy_signal /= np.max(noisy_signal)

        return noisy_signal
    
    def load_signals(self):

        self.Fx = self.create_noise_signal(self.t)
        self.Fy = self.create_noise_signal(self.t)
        self.Fz = self.create_noise_signal(self.t)
        self.Tx = self.create_noise_signal(self.t)
        self.Ty = self.create_noise_signal(self.t)
        self.Tz = self.create_noise_signal(self.t)


    def calculate_force(self, t):

        t = int(t)

        max_F = 0.01 # N

        Fx = self.Fx[t] * max_F
        Fy = self.Fy[t] * max_F
        Fz = self.Fz[t] * 0

        Fd = np.array([Fx, Fy, Fz])

        return Fd

    def calculate_torque(self, t):

        t = int(t)

        max_T = 0.01 # N

        Tx = 0
        Ty = 0
        Tz = self.Tz[t] * max_T

        Td = np.array([Tx, Ty, Tz])

        return Td
    

class SpaceEnvironment():
    def __init__():
        pass
    def calculate_force(self):
        pass
    def calculate_torque(self):
        pass

class Disturbances():
    def __init__(self, dt):

        self.noise = RandomSineNoise(dt)

    def calculate_disturbance_force(self, p, q, t):

        Fd = self.noise.calculate_force(t)
        # Constant Force (uncomment next line)
        #Fd = np.array([0.1,0,0])

        return Fd
    
    def calculate_disturbance_torque(self,p, q, t):

        Td = self.noise.calculate_torque(t)

        # Constant Torque (uncomment next line)
        #Td = np.array([0,0,0])

        return Td

if __name__ == "__main__":

    ts = np.linspace(0, int(param.simulation_time / param.dt)-1, int(param.simulation_time / param.dt))

    noise = RandomSineNoise(param.dt)
    disturbances_F = []

    for t in ts:

        Fd = noise.calculate_force(t)
        disturbances_F.append(Fd)

    disturbances_F = np.array(disturbances_F)

    plt.plot(ts*param.dt, disturbances_F[:, 0])
    plt.plot(ts*param.dt, disturbances_F[:, 1])
    plt.show()




```

### Content of pyaocs/simulation/environment.py:

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from pyaocs.utils import Quaternion_to_Euler321
import os
from importlib.resources import files

from pyaocs.simulation.non_linear_equations import NonLinearPropagator
from pyaocs import parameters as param

class SatelliteEnv():
    
    def __init__(self, render = True, bullet = False, real_time = True, use_disturbances = False, trajectory = None):

        self.render_GUI = render
        self.pybullet = bullet # Wether to use or not Pybullet
        self.realTime = real_time # If True simulation runs in real time. Change to False if training reinforcement learning.
        self.use_disturbances = use_disturbances
        self.trajectory = trajectory

        if self.pybullet or self.render_GUI:
            import pybullet as pb
            import pybullet_data
            self.pb = pb
            self.pybullet_data = pybullet_data

        if self.render_GUI:

            username = os.environ.get('USERNAME')
            if username == "Nesto":
                physicsClient = self.pb.connect(self.pb.GUI, options= ' --opengl2')# p.GUI or p.DIRECT for non-graphical version

            else:
                physicsClient = self.pb.connect(self.pb.GUI)# p.GUI or p.DIRECT for non-graphical version

           # self.pb.configureDebugVisualizer(self.pb.COV_ENABLE_GUI,0)

            if self.trajectory is not None:
                for point in self.trajectory["target_positions"].T:
                    self.pb.addUserDebugLine(point, point+0.01, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)

                # to run faster later when we plot one point at a time, as we only care about position.
                self.trajectory = self.trajectory["target_positions"].T

        elif not self.render_GUI and self.pybullet:
            physicsClient = self.pb.connect(self.pb.DIRECT)# p.GUI or p.DIRECT for non-graphical version

        if self.pybullet or self.render_GUI:
            self.pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
            self.pb.setGravity(0,0,0)
            planeId = self.pb.loadURDF("plane.urdf") # creates the square plane in simulation.
            urdf_path = files('pyaocs').joinpath('urdf_models/cylinder.urdf')
            self.boxId = self.pb.loadURDF(str(urdf_path)) # load the object and set the pos and orientation

        self.initialize()

        # Change dynamics of system and add correct moments of inertia
        I = param.I
        self.mass = param.m
        print(f"Mass of satellite: {self.mass} kg")

        if self.pybullet:
            self.pb.changeDynamics(self.boxId, -1, mass = self.mass, linearDamping = 0, angularDamping = 0, localInertiaDiagonal = I)
            #self.mass, _, self.CoG, *_ = self.pb.getDynamicsInfo(self.boxId, -1)

        self.F = param.F # force applied by each thruster in Newtons
        self.thrust_vector_range = param.thrust_vector_range # Range that the thruster can thrust vector in degrees.
        self.fps = param.fps # Frames per second of the simulation. Mantain to 240 to mantain accuracy.
        self.simulation_time = param.simulation_time # max duration of the simulation before reset
        self._end_step = (self.fps * self.simulation_time) -1 # last timestep of the simulation.
        self.dt = 1 / self.fps
        if self.pybullet:
            self.pb.setTimeStep(self.dt) # If timestep is set to different than 240Hz, it affects the solver and other changes need to be made (check pybullet Documentation)

        self.thruster_positions = np.array(param.thruster_positions)

        if not self.pybullet:
            self.nl = NonLinearPropagator(dt = self.dt, y0=self.y0, use_disturbances=self.use_disturbances)

        # For reinforcement Leanring
        #self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,))
    
    def initialize(self):
        """Function to be runned everytime the class is initialized and the environment is reset. All the variables are reinitialized and the simulation starts from 0.
        """
        self.control = False # Wether to send control command or not (True or False)
        self._done = False  #If simulation has ended --> done = True
        self._current_step = 0 #Current step equals 0 at the beginning
        self.reward = 0 # For RL purposes, starts with 0.
        self.firing_time = 0 # Time the thrusters are on
        self.start_position = param.start_position # define the cube start position
        self.start_velocity = param.start_velocity
        self.start_orientation = param.start_orientation # define start orientation
        self.start_angular_velocity = param.start_angular_velocity # in terms of euler angles

        self.y0 = np.concatenate((self.start_position, self.start_velocity, self.start_orientation, self.start_angular_velocity))
        
        self.target_position = param.target_position # Target position where we want the satellite to move.
        self.target_orientation = param.target_orientation  # [φ,θ,ψ]
        self.previous_position = self.start_position # Previous position equals starting position at the beginning.
        self.current_position = self.start_position # Current position equals starting position at the beginning.
        self.current_orientation = self.start_orientation # Current orientation equals starting orientation at the beginning.
        self.current_velocity = self.start_velocity  # Current velocity equals starting velocity at the beginning.
        
        if self.pybullet:
            self.pb.resetBasePositionAndOrientation(self.boxId, self.current_position, self.current_orientation) # Set the position of the satellite.

        self.actual_positions = []
        self.target_positions = []
        self.actual_orientations = []
        self.target_orientations = []

        # Store 6 system parameters.
        self.F1s = []
        self.F2s = []
        self.γ1s = []
        self.γ2s = []
        self.δ1s = []
        self.δ2s = []

        self.parameters = [0,0,0,0,0,0]

        if self.render_GUI:
            # Draw the line that should be followed by satellite
            #self.pb.addUserDebugLine(self.current_position, self.target_position, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
            pass
    
    def reset_fps(self, fps):
        """Change the timestep of the simulation.

        :param fps: New timestep
        :type fps: float
        """
        self.fps = fps
        self.dt = 1 / self.fps
        self._end_step = (self.fps * self.simulation_time) -1 # last timestep of the simulation.
        if self.pybullet:
            self.pb.setTimeStep(self.dt)
        if not self.pybullet:
            self.nl.dt = self.dt
        
    def reset(self, seed = None):
        """Reset the environment and return the initial observation

        :param seed: Some variable that we are not using, but is required for RL, defaults to None
        :type seed: _type_, optional
        :return: (observation, info) 
        :rtype: _type_
        """
        # Reset the environment and return the initial observation
        #p.resetSimulation()
        self.initialize()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        """Get current state of the simulation, we choose to return the current position, current orientation and target position, but we can return whatever we want.

        :return: obs
        :rtype: array with the desired info.
        """

        # Data type of Spaces.Box is float32.
        """
        obs = np.concatenate((self.current_position,            # obs[0:3]
                              self.current_orientation,         # obs[3:7]
                              self.target_position,             # obs[7:10]
                              self.target_orientation,          # obs[10:13]
                              self.current_velocity,            # obs[13:16]
                              self.parameters)                  # obs[16:22]
                              ).astype(np.float32)
        """
        obs = {"current_position": np.array(self.current_position),
               "current_orientation": np.array(self.current_orientation),
               "target_position": np.array(self.target_position),
               "target_orientation": np.array(self.target_orientation),
               "current_velocity": np.array(self.current_velocity),
               "parameters": np.array(self.parameters)}

        return obs
 
    def _get_info(self):
        """Function to return information about the system.

        :return: You can choose what to return.
        :rtype: dict
        """
        distance_from_target = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.current_position))**2))
        
        return {'Current Position (m)': self.current_position, "Target Position (m)": self.target_position, "Distance from target": distance_from_target, "Rewards": self.reward}
    
    def step(self, action):
        """Function to be run for every timestep of the simulation. Every time the function is run, the simulation runs one timestep of length dt, defined in __init__. Most important function in the class. 

        :param action: Variable that the control algorithm gives to the simulation to set the actions performed by the system. It's composed of 6 variables normalized between -1 and 1: 
            - F1: a value of 1 means thruster 1 should be activated, -1 means it should be off. In between depends on the scheme decided.
            - F2: a value of 1 means thruster 2 should be activated, -1 means it should be off. In between depends on the scheme decided.
            - γ1: angle y in degrees of thruster 1, should be multiplied by the range of motion of the compliant mechanism.
            - γ2: angle y in degrees of thruster 2
            - δ1: angle z in degrees of thruster 1
            - δ2: angle z in degrees of thruster 2
        :type action: _type_
        :return: _description_
        :rtype: _type_
        """

        F1, F2 = action[:2]

        γ1, γ2, δ1, δ2 = action[2:] * np.pi / 180

        # If the output signal from the PID is very small, smaller than 0.01 (for example), the thruster is not activated. This reduces fuel usage, but decreases accuracy.
        if F1 > 0.01:
            F1 = self.F
            #F = round(F1, 5)
            self.apply_force(force = F1, γ = γ1, δ = δ1, thruster_number = 1)
        else:
            self.F1 = 0

        if F2 > 0.01:
            F2 = self.F
            #F = round(F2, 5)
            self.apply_force(force = F2, γ = γ2, δ = δ2, thruster_number = 2)
        else:
            F2 = 0

        self.parameters = np.array([F1, F2, γ1, γ2, δ1, δ2])

        if self.pybullet:
            self.pb.stepSimulation()
            self.current_position, self.current_orientation = self.pb.getBasePositionAndOrientation(self.boxId)
            self.current_velocity = self.pb.getBaseVelocity(self.boxId)[0]

        else:
            p, v, q, w = self.nl.stepSimulation(self.parameters)
            self.current_position = p.copy()
            self.current_orientation = q.copy()
            self.current_velocity = v.copy()

        if self.render_GUI and self.pybullet == False:
            self.pb.resetBasePositionAndOrientation(self.boxId, p, q) # Set the position of the satellite.
            self.pb.resetBaseVelocity(self.boxId, v)

        if self.realTime:
            time.sleep(1./self.fps)

        # Save the data from the simulation. Positions, target positions, orientations
        self.parameters[2:] = self.parameters[2:] * 180 / np.pi
        self.data_recorder(self.parameters)
        
        reward = self.reward_calculation()
        self.reward += reward

        self.previous_position = self.current_position
            
        if self._current_step == self._end_step:
            self._done = True
            observation = self._get_obs()
            #print(observation)
            info = self._get_info()
            #print(info)
        
        else:   
            observation = self._get_obs()
            info = self._get_info()

        self._current_step += 1
        
        return observation, reward, self._done, self.control, info
    
    def get_force_vector(self, force, γ, δ, thruster_number = 1):
        """Given the thruster angles and force, find the force vector.

        :param force: Force applied by the thruster
        :type force: float
        :param γ: the angle in radians that the thruster is vectored about the y-axis.
        :type γ: float
        :param δ: the angle in radians that the thruster is vectored about the z-axis. 
        :type δ: float
        :param thruster_number: The number of the thruster you want to activate, defaults to 1
        :type thruster_number: int, optional
        """

        opposite_y = force * np.sin(δ)
        opposite_z = force * np.sin(γ)
        opposite_total = np.sqrt(opposite_y**2 + opposite_z**2)
        resultant_x = np.sqrt(force**2 - opposite_total**2)

        force_magnitude_check = np.sqrt(resultant_x**2 + opposite_y**2 + opposite_z**2)

        assert round(force_magnitude_check, 5) == force, f"Component forces {force_magnitude_check} do not equate to initial force {force}."

        # In the x_direction the resultant_x should have different sign. Unless you don't want one of the thrusters to activate.
        # In the y_direction the opposite_y should have opposite signs for rotation and equal sign for translation.
        # In the z_direction the opposite_z should have opposite signs for rotation and equal sign for translation.

        if thruster_number == 1:
            self.force_vector = np.array([resultant_x, opposite_y, -opposite_z])

        elif thruster_number == 2:
            # Thruster is placed on opposite direction to thruster 1, so the sign of resultant_x is reversed
            self.force_vector = np.array([-resultant_x, -opposite_y, -opposite_z])

        self.thruster_position = self.thruster_positions[thruster_number - 1] # Position of thruster with respect to satellite origin.
    
    def apply_force(self, force: float, γ: float, δ: float, thruster_number: int = 0) -> None:
        """Function to apply force by the thruster in a given timestep

        :param force: Force applied by the thruster
        :type force: float
        :param γ: the angle in radians that the thruster is vectored about the y-axis.
        :type γ: float
        :param δ: the angle in radians that the thruster is vectored about the z-axis. 
        :type δ: float
        :param thruster_number: The number of the thruster you want to activate, defaults to 1
        :type thruster_number: int, optional
        """

        self.get_force_vector(force, γ, δ, thruster_number)

        if self.pybullet:
            self.pb.applyExternalForce(self.boxId, -1, self.force_vector, self.thruster_position, self.pb.LINK_FRAME) # in Newton # WORLD_FRAME p.LINK_FRAME
            #p.applyExternalForce(self.boxId, -1, (-resultant_x, -opposite_y, -opposite_z), [-0.5,0,0], p.LINK_FRAME)

        if self.render_GUI:
            joint_index = thruster_number - 1 # Index of the joint you want to control
            target_angle = δ # Target angle in radians (for example, 90 degrees)
            self.pb.resetJointState(bodyUniqueId=self.boxId, jointIndex=joint_index, targetValue=target_angle)


        self.draw_thrust_trajectory()

        self.firing_time += self.dt

    def reward_calculation(self):
        """Defines how the rewards for RL are calculated. Many methods can be used.

        :return: reward
        :rtype: int
        """

        distance_from_target_prev = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.previous_position))**2))
        distance_from_target = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.current_position))**2))

        #reward =  50000*(distance_from_target_prev - distance_from_target) / distance_from_target
        
        if distance_from_target <= distance_from_target_prev:
            reward = 1 / distance_from_target
        else:
            reward = 0
        

        return reward

    def draw_thrust_trajectory(self):
        """Draw the red lines representing the thrust.
        """
        if self.render_GUI:
            line_lenght = 0.2
            # Remember self.force_vector is defined in get_force_vector
            self.pb.addUserDebugLine(self.thruster_position, self.thruster_position - line_lenght*self.force_vector, lineColorRGB=[1.00,0.25,0.10], lineWidth=5.0, lifeTime=self.dt, parentObjectUniqueId = self.boxId)

        # show all the trajectory points at their correpsonding time
        if self.render_GUI and self.trajectory is not None and self._current_step % param.step_rate == 0:

            point = self.trajectory[int(self._current_step / param.step_rate)]
            self.pb.addUserDebugLine(point, point+0.02, lineColorRGB=[0, 1, 0], lineWidth=4.0, lifeTime=0)

    
    def data_recorder(self, inputs):
        """Saves important data every timestep for visualization after the simulation finishes, like the x, y, z position coordinates.
        """
        self.actual_positions.append(self.current_position)
        self.target_positions.append(self.target_position)

        #rotation = Rotation.from_quat(self.current_orientation)
        #euler_angles = rotation.as_euler('zyx', degrees=True)
        euler_angles = Quaternion_to_Euler321(self.current_orientation)

        self.actual_orientations.append(euler_angles)
        self.target_orientations.append(self.target_orientation)

        F1, F2, γ1, γ2, δ1, δ2 = inputs

        self.F1s.append(F1)
        self.F2s.append(F2)
        self.γ1s.append(γ1)
        self.γ2s.append(γ2)
        self.δ1s.append(δ1)
        self.δ2s.append(δ2)

    def render(self, mode = 'human'):
        """This is required for the RL library. Not used if not.

        :param mode: _description_, defaults to 'human'
        :type mode: str, optional
        """
        self.render_GUI = True
        return
        
    def plot_training(self):
        """If RL used, this function plots important things that we would like to see, like episode rewards.
        """
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Rewards")
        
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_balances)
        plt.title("Balances (£)")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_trades)
        plt.title("Number of trades")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_success)
        plt.title("Successful trades (%)")
        
        plt.show()
        
    
    def plot_results(self, name = "performance_report"):
        """Plots the data recorded.

        :param name: _description_, defaults to "performance_report"
        :type name: str, optional
        """

        self.t = np.linspace(0, (self._current_step + 1) / self.fps, self._current_step)

        self.plot_pose()
        self.plot_parameters()
        if self.pybullet == False:
            self.plot_disturbances()

        print(f"Total firing time: {round(self.firing_time, 3)} s")

        return
    
    def plot_pose(self):

        self.actual_positions = np.array(self.actual_positions)
        self.target_positions = np.array(self.target_positions)
        self.actual_orientations = np.array(self.actual_orientations)
        self.target_orientations = np.array(self.target_orientations)

        plt.subplot(3, 2, 1)
        plt.plot(self.t, self.actual_positions[:, 0],  label='Actual')
        #plt.plot(self.t, self.target_positions[:, 0], label='Target')
        plt.title("X positions")
        plt.xlabel("t (s)")
        plt.ylabel("x (m)")

        plt.subplot(3, 2, 2)
        plt.plot(self.t, self.actual_positions[:, 1],  label='Actual')
       # plt.plot(self.t, self.target_positions[:, 1], label='Target')
        plt.title("Y positions")
        plt.xlabel("t (s)")
        plt.ylabel("y (m)")

        plt.subplot(3, 2, 3)
        plt.plot(self.t, self.actual_positions[:, 2],  label='Actual')
        #plt.plot(self.t, self.target_positions[:, 2], label='Target')
        plt.title("Z positions")
        plt.xlabel("t (s)")
        plt.ylabel("z (m)")

        plt.subplot(3, 2, 4)
        plt.plot(self.t, self.actual_orientations[:,2],  label='Actual')
        #plt.plot(self.t, self.target_orientations[:,2], label='Target')
        plt.title("φ positions")
        plt.xlabel("t (s)")
        plt.ylabel("φ (°)")

        plt.subplot(3, 2, 5)
        plt.plot(self.t, self.actual_orientations[:,1],  label='Actual')
        #plt.plot(self.t, self.target_orientations[:,1], label='Target')
        plt.title("θ positions")
        plt.xlabel("t (s)")
        plt.ylabel("θ (°)")

        plt.subplot(3, 2, 6)
        plt.plot(self.t, self.actual_orientations[:,0],  label='Actual')
        #plt.plot(self.t, self.target_orientations[:,0], label='Target')
        plt.title("ψ positions")
        plt.xlabel("t (s)")
        plt.ylabel("ψ (°)")

        plt.tight_layout()
        plt.show()
    
    def plot_parameters(self):
        """Plot the 6 system parameters F1, F2, γ1, γ2, δ1, δ2 over time"""
    
        self.F1s = np.array(self.F1s)
        self.F2s = np.array(self.F2s)
        self.γ1s = np.array(self.γ1s)
        self.γ2s = np.array(self.γ2s)
        self.δ1s = np.array(self.δ1s)
        self.δ2s = np.array(self.δ2s)

        plt.subplot(3, 2, 1)
        plt.plot(self.t, self.F1s,  label='F1')
        plt.title("F1 thrust")
        plt.xlabel("t (s)")
        plt.ylabel("F (N)")
        plt.grid()

        plt.subplot(3, 2, 2)
        plt.plot(self.t, self.F2s, label='F2')
        plt.title("F2 thrust")
        plt.xlabel("t (s)")
        plt.ylabel("F (N)")
        plt.grid()

        plt.subplot(3, 2, 3)
        plt.plot(self.t, self.γ1s,  label='γ1')
        plt.title("γ1 angle")
        plt.xlabel("t (s)")
        plt.ylabel("γ1 (°)")
        plt.grid()

        plt.subplot(3, 2, 4)
        plt.plot(self.t, self.γ2s, label='γ2')
        plt.title("γ2 angle")
        plt.xlabel("t (s)")
        plt.ylabel("γ2 (°)")
        plt.grid()

        plt.subplot(3, 2, 5)
        plt.plot(self.t, self.δ1s,  label='δ1')
        plt.title("δ1 angle")
        plt.xlabel("t (s)")
        plt.ylabel("δ1 (°)")
        plt.grid()

        plt.subplot(3, 2, 6)
        plt.plot(self.t, self.δ2s, label='δ2')
        plt.title("δ2 angle")
        plt.xlabel("t (s)")
        plt.ylabel("δ2 (°)")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_disturbances(self):

        self.nl.disturbances_F = np.array(self.nl.disturbances_F)
        self.nl.disturbances_T = np.array(self.nl.disturbances_T)

        plt.subplot(3, 2, 1)
        plt.plot(self.t, self.nl.disturbances_F[:, 0])
        plt.title("x force disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Fx (N)")

        plt.subplot(3, 2, 2)
        plt.plot(self.t, self.nl.disturbances_F[:, 1])
        plt.title("y force disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Fy (N)")

        plt.subplot(3, 2, 3)
        plt.plot(self.t, self.nl.disturbances_F[:, 2])
        plt.title("z force disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Fz (N)")

        plt.subplot(3, 2, 4)
        plt.plot(self.t, self.nl.disturbances_T[:,0])
        plt.title("φ torque disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Tφ (Nm)")

        plt.subplot(3, 2, 5)
        plt.plot(self.t, self.nl.disturbances_T[:,1])
        plt.title("θ torque disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Tθ (Nm)")

        plt.subplot(3, 2, 6)
        plt.plot(self.t, self.nl.disturbances_T[:,2])
        plt.title("ψ torque disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Tψ (Nm)")

        plt.tight_layout()
        plt.show()

    def close(self):
        """Run to close the simulation window
        """
        if self.pybullet or self.render_GUI:
            self.pb.disconnect()
```

### Content of pyaocs/simulation/noise.py:

```python
import numpy as np
import matplotlib.pyplot as plt
from pyaocs import parameters as param
from pyaocs.utils import Quaternion_to_Euler321, euler_to_quaternion, normalize_quaternion, ExponentialMovingAverageFilter

class Noise():
    def __init__(self, sample = "100hz"):

        duration = param.simulation_time   # Duration in seconds

        self.sample = sample

        if self.sample == "24hz":
            sample_rate = 24
        else:
            sample_rate = param.fps  # Sampling rate in Hz

        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        self.step = 0

        if self.sample == "24hz":
            self.position_noise = self.generate_signal(1, 0.00005)
            self.orientation_noise = self.generate_signal(1, 0.00005)
        else:
            self.position_noise = self.generate_signal(1, 0.00001)
            self.orientation_noise = self.generate_signal(1, 0.00001)


    def add_noise(self, obs):
        """
        Add noise to the IMU sensor data.
        """
        p = obs["current_position"]
        q = obs["current_orientation"]

        p = p + self.position_noise[self.step]

        q = q + self.orientation_noise[self.step]
        q = normalize_quaternion(q)

        obs["current_position"] = p
        obs["current_orientation"] = q

        self.step += 1

        return obs
    
    def generate_signal(self, frequency, amplitude):
        """
        Generate a sine wave signal with the given parameters.
        """
        # Generate a sine wave
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * self.t)

        if self.sample:
            coefficient = (amplitude * 10)
        else:
            coefficient = (amplitude * 50)

        # Add Gaussian noise
        noise = coefficient*np.random.normal(0, 0.2, sine_wave.shape)   # Noise with mean=0 and std dev=0.2
        noisy_signal = sine_wave + noise

        return noisy_signal




if __name__ == "__main__":

    noise = Noise("24hz")
    ema = ExponentialMovingAverageFilter(alpha = 0.1)

    obs = {"current_position": np.array([0, 0, 0]),
           "current_orientation": euler_to_quaternion(np.array([0, 0, 0]))}
    
    positions = []
    orientations = []
    filtered_positions = []

    num_steps = 200

    for i in range(num_steps):
        obs = noise.add_noise(obs)
        filtered = ema.filter(obs["current_position"])
        filtered_positions.append(filtered)
        positions.append(obs["current_position"])
        orientations.append(Quaternion_to_Euler321(obs["current_orientation"]))

    # Make to subfigures for position and orientation
    positions = np.array(positions).T
    orientations = np.array(orientations).T
    filtered_positions = np.array(filtered_positions).T

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(noise.t[:num_steps], positions[0], label="X")
    axs[0].plot(noise.t[:num_steps], filtered_positions[0], label="Filtered X")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(noise.t[:num_steps], orientations[0], label="Yaw")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Orientation")
    axs[1].legend()

    plt.show()    

```

### Content of pyaocs/simulation/non_linear_equations.py:

```python
import numpy as np
import time

from pyaocs.utils import normalize_quaternion, transform_vector
from pyaocs import parameters as param
from pyaocs.simulation.disturbances import Disturbances

def quaternion_derivative(q, w):
     """
     Calculate the derivative of a quaternion with respect to angular velocity.

     Args:
          q: Quaternion (numpy array) [x, y, z, w]
          w: Angular velocity (numpy array) [wx, wy, wz]

     Returns:
          dq_dt: Derivative of the quaternion with respect to angular velocity (numpy array) [dq/dt]
     """
     wx, wy, wz = w
     
     A = np.array([[0, wz, -wy, wx],
                    [-wz, 0, wx, wy],
                    [wy, -wx, 0, wz],
                    [-wx, -wy, -wz, 0]])

     dq_dt = 0.5 * np.dot(A, q)

     return dq_dt

def angular_velocity_derivative(I, w, Tc, Td):
     """Function to calculate the change in angular velocity given
     I, w, Tc and Td

     :param I: moment of inertia
     :type I: np.ndarray
     :param w: angular velocity
     :type w: np.ndarray
     :param Tc: control torque.
     :type Tc: np.ndarray
     :param Td: disturbance torque.
     :type Td: np.ndarray
     :return: change in angular velocity
     :rtype: np.ndarray
     """

     Ix, Iy, Iz = I
     wx, wy, wz = w

     H = np.array([(Iy - Iz)*wy*wz,
                    (Iz - Ix)*wx*wz,
                    (Ix - Iy)*wx*wy])
     
     dw_dt = np.divide((H + Tc + Td), I)

     return dw_dt

def force_calculation(inputs):

     F1, F2, γ1, γ2, δ1, δ2 = inputs

     θ_r1 = np.sqrt(γ1**2 + δ1**2)
     θ_r2 = np.sqrt(γ2**2 + δ2**2)

     # Find acceleration in every direction
     F = np.array([(F1*np.cos(θ_r1) - F2*np.cos(θ_r2)),
                   (F1*np.sin(δ1) - F2*np.sin(δ2)),
                   (- F1*np.sin(γ1) - F2*np.sin(γ2))
                   ])

     return F

class NonLinearPropagator():

     def __init__(self, dt, y0, use_disturbances = False):

          self.use_disturbances = use_disturbances

          self.dt = dt

          if use_disturbances:
               self.disturbance = Disturbances(self.dt)

          self.m = param.m # 1kg
          self.L = param.L # m
          self.I = param.I

          x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz = y0

          self.p = np.array([x, y, z])
          self.v = np.array([vx, vy, vz])
          self.q = np.array([q1, q2, q3, q4])
          self.w = np.array([wx, wy, wz])

          self.disturbances_F = []
          self.disturbances_T = []

          self.step = 0

     def rotation_derivative(self, inputs):

          F1, F2, γ1, γ2, δ1, δ2 = inputs

          self.q = normalize_quaternion(self.q)

          dq_dt = quaternion_derivative(self.q, self.w)

          Tx = 0
          Ty = -F1*self.L*np.sin(γ1) + F2*self.L*np.sin(γ2)
          Tz = -F1*self.L*np.sin(δ1) - F2*self.L*np.sin(δ2)

          Tc = np.array([Tx, Ty, Tz])

          if self.use_disturbances:
               Td = self.disturbance.calculate_disturbance_torque(self.p, self.q, self.step)
          else:
               Td = np.array([0, 0, 0])
          
          self.disturbances_T.append(Td)

          dw_dt = angular_velocity_derivative(self.I, self.w, Tc, Td)

          return dq_dt, dw_dt
     
     def position_derivative(self, inputs):
           
          Fc = force_calculation(inputs)

          if self.use_disturbances:
               Fd = self.disturbance.calculate_disturbance_force(self.p, self.q, self.step)
          else:
               Fd = np.array([0, 0, 0])

          self.disturbances_F.append(Fd)

          # Find acceleration in every direction
          F = Fc + Fd
          accel = F / self.m

          #Convert from body fixed to world inertial reference frame
          accel = transform_vector(accel, self.q)

          return accel

     def stepSimulation(self, inputs):

          # Perform rotattion propagation
          dq_dt, dw_dt = self.rotation_derivative(inputs)

          self.q += self.dt*dq_dt
          self.w += self.dt*dw_dt

          #Perform translation propagation 
          accel = self.position_derivative(inputs)

          #previous_v = self.v
          self.v += self.dt*accel
          #assuming that velocity chnages linearly, which is probably better aproximation than instantaneous change.
          #avg_v = (previous_v + self.v) / 2
          self.p += self.dt*self.v

          self.step += 1

          return self.p, self.v, self.q, self.w


def test_quaternion_derivative():

     # Example usage
     q = np.array([1.0, 0.0, 0.0, 0.0])  # Example quaternion [x, y, z, w]
     w = np.array([0.1, 0.2, 0.3])     # Example angular velocity [wx, wy, wz]

     dq_dt = quaternion_derivative(q, w)
     print("Derivative of the quaternion:", dq_dt)


if __name__ == "__main__":

     t = np.linspace(0, 10, 100)

     p = np.array(param.start_position)  # Example position [x, y, z]
     v = np.array(param.start_velocity)     # Example velocity [vx, vy, vz]
     q = np.array(param.start_orientation)  # Example quaternion [x, y, z, w]
     w = np.array(param.start_angular_velocity)     # Example angular velocity [wx, wy, wz]

     y0 = np.concatenate((p, v, q, w))
     dt = param.dt

     nl = NonLinearPropagator(dt = dt, y0=y0)

     F1 = 0
     F2 = 1
     γ1 = 0
     γ2 = 0
     δ1 = 0
     δ2 = param.thrust_vector_range * np.pi / 180

     inputs = F1, F2, γ1, γ2, δ1, δ2

     for _ in range(int(param.simulation_time / dt)):

          p, v, q, w = nl.stepSimulation(inputs)

          #time.sleep(dt)
```

### Content of pyaocs/urdf_models/cylinder.urdf:

```python
<?xml version="1.0"?>
<robot name="myfirst">
    <link name="base_link">
        <visual>
            <geometry>
                <cylinder length="0.3" radius="0.15"/>
            </geometry>
            <material name="black">
                <color rgba="0.1 0.1 0.1 0.6"/>
            </material>
        </visual>
    </link>

    <link name="thruster_1">
        <visual>
            <geometry>
                <cylinder length="0.1" radius="0.01"/>
            </geometry>
            <material name="green">
                <color rgba="0 1 0.1 0.6"/>
            </material>
            <origin rpy="0 1.57075 0" xyz="0 0 0"/>
        </visual>
    </link>

    <joint name="thruster_1_joint" type="revolute">
    	<parent link="base_link"/>
    	<child link="thruster_1"/>
    	<origin xyz="-0.15 0 0" rpy="0 0 0"/>
       	<limit lower="-0.5" upper="0.5" effort="10" velocity="3"/>
        <axis xyz="0 0 1"/>
	</joint>

    <link name="thruster_2">
        <visual>
            <geometry>
                <cylinder length="0.1" radius="0.01"/>
            </geometry>
            <origin rpy="0 1.57075 0" xyz="0 0 0"/>
        </visual>
    </link>

    <joint name="thruster_2_joint" type="revolute">
    	<parent link="base_link"/>
    	<child link="thruster_2"/>
    	<origin xyz="0.15 0 0" rpy="0 0 0"/>
       	<limit lower="-0.5" upper="0.5" effort="10" velocity="3"/>
        <axis xyz="0 0 1"/>
	</joint>
	
</robot>

```

### Content of pyaocs/utils.py:

```python
import numpy as np
import math

def wait_for_actuator(state, action):
    """Wait for the actuator to reach the desired state before firing."""

    if not np.array_equal((state[-4:]).astype(float), action[-4:].astype(float)):
        action[:2] = 0
        state[:2] = 0

        #print(f"[INFO] Waiting for actuator to reach target state: {action[-4:]}")
        
    return action, state

def Quaternion_to_Euler321(q):
    """Convert quaternion to euler in order 321 (ZYX) and in the fixed body reference frame."""

    x,y,z,w =q

    φ = math.atan2(2*((w*x)+(y*z)),1-2*((x**2)+y**2))

    θ = -(math.pi/2) +2*math.atan2(math.sqrt(1+2*((w*y)-(x*z))),math.sqrt(1-2*((w*y)-(x*z))))

    ψ = math.atan2(2*((w*z)+(x*y)),1-2*((y**2)+z**2))

    φ_deg = φ *180/math.pi
    θ_deg = θ *180/math.pi
    ψ_deg = ψ *180/math.pi

    return [ψ_deg, θ_deg, φ_deg]

def normalize_quaternion(q):
    """
    Normalize a quaternion to have unit length.

    Args:
        q: Quaternion as a numpy array [w, x, y, z].

    Returns:
        normalized_q: Normalized quaternion as a numpy array [w, x, y, z].
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        return q  # Avoid division by zero
    normalized_q = q / norm

    return normalized_q

def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion to rotation matrix.

    :param quaternion: q (x, y, z, w)
    :type quaternion: list
    :return: R, rotation matrix
    :rtype: np.ndarray
    """

    x, y, z, w = quaternion  # Assuming quaternion as [a, b, c, d]

    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

    return R

def transform_vector(vector_world, orientation, inverse = False, euler = False):

    """
    If inverse == False, it can be used to transform a vector by the given orientation.

    If inverse == True it could be used for the follwoing:
    Convert vector from world reference frame to satellite reference frame. 
    vector_world can refer to the vector connecting the current position of the 
    satelite and the target position. The output would be that same vector in
    the satellite reference frame.

    :param vector_world: vector to transform.
    :type vector_world: list
    :param orientation: orientation of the satellite q (x, y, z, w).
    :type orientation: list
    :return: transformed vector
    :rtype: list
    """

    if euler == True:
        orientation = euler_to_quaternion(orientation)

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(orientation)

    # Convert the vector to a column vector
    vector_world = np.array(vector_world).reshape(-1, 1)

    # Apply inverse rotation to the vector
    if inverse == True:
        vector_satellite = np.dot(np.linalg.inv(rotation_matrix), vector_world)
    else:
        vector_satellite = np.dot(rotation_matrix, vector_world)

    return vector_satellite.flatten()  # Return as a flattened array

def euler_to_quaternion(euler):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """

    roll, pitch, yaw = euler


    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

class ExponentialMovingAverageFilter:
    def __init__(self, alpha=0.5):
        """
        Initializes the filter.
        
        :param alpha: Smoothing factor of the filter, 0 < alpha <= 1. 
                      A higher alpha discounts older observations faster.
        """
        self.alpha = alpha
        self.estimated_value = None

    def filter(self, new_value):
        """
        Applies the Exponential Moving Average filter to the new value.
        
        :param new_value: The new data point to be filtered.
        :return: The filtered value.
        """
        if self.estimated_value is None:
            # Initialize the estimated value with the first data point
            self.estimated_value = new_value
        else:
            # Apply the EMA formula
            self.estimated_value = self.alpha * new_value + (1 - self.alpha) * self.estimated_value
        
        return self.estimated_value
```

### Content of pyproject.toml:

```python
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pyaocs"]

[project]
name = "pyaocs"
version = "0.0.5"
authors = [
    { name="Pedro", email="pr1u20@soton.ac.uk" },
]
description = "Satellite Digital Twin to test control strategies."
dependencies = [
    "numpy",
    "pybullet",
    "matplotlib"
]
readme = "README.md"
license = { file="LICENSE" }
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[tool.setuptools.package_data]
pyaocs = ["urdf_models/*.urdf"]

[project.urls]
"Homepage" = "https://github.com/pr1u20/AOCS-digital-twin"
"Bug Tracker" = "https://github.com/pr1u20/AOCS-digital-twin/issues"
```

### Content of requirements.txt:

```python
numpy
matplotlib
pybullet
ipython
build
twine
```

### Content of tests/example.py:

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

