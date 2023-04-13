import numpy as np
import matplotlib.pyplot as plt

class PID:
    def __init__(self, Kp, Ki, Kd, Ts):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.integral = 0
        self.prev_error = 0

    def update(self, error):
        self.integral += error*self.Ts
        derivative = (error - self.prev_error)/self.Ts
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = error
        return output

def gradient_descent_controller(Kp, Ki, Kd, Ts, setpoint, plant_output):
    # Initialize parameters
    p = [Kp, Ki, Kd]
    dp = [1, 1, 1]
    tolerance = 0.01

    # Initialize error and best error
    error = 0
    best_error = np.inf

    # Initialize counter and iteration limit
    counter = 0
    max_iterations = 1000

    # Initialize PID controller
    controller = PID(p[0], p[1], p[2], Ts)

    # Main loop
    while np.sum(dp) > tolerance and counter < max_iterations:
        # Simulate system with current parameters
        plant_output = np.zeros(len(setpoint))
        for i in range(len(setpoint)):
            error = setpoint[i] - plant_output[i]
            control_input = controller.update(error)
            plant_output[i] = np.sin(control_input)

        # Calculate error
        error = np.sum((setpoint - plant_output)**2)

        # Update best error and parameters
        if error < best_error:
            best_error = error
            dp = 1.1*dp
        else:
            p += dp
            dp = 0.9*dp

        # Reset PID controller with updated parameters
        controller = PID(p[0], p[1], p[2], Ts)

        # Update counter
        counter += 1

    # Return best parameters
    return p[0], p[1], p[2]

def moving_average_filter(data, window_size):
    weights = np.repeat(1.0, window_size)/window_size
    filtered_data = np.convolve(data, weights, 'valid')
    return filtered_data

def simulate_system(Kp, Ki, Kd, Ts, duration, noise_amplitude=0):
    t = np.arange(0, duration, Ts)
    setpoint = np.sin(t)
    plant_output = np.zeros(len(t))
    controller = PID(Kp, Ki, Kd, Ts)

    for i in range(len(t)):
        error = setpoint[i] - plant_output[i]
        control_input = controller.update(error)
        plant_output[i] = np.sin(control_input) + noise_amplitude*np.random.randn()

    return setpoint, plant_output

# Parameters
Kp = 1.0
Ki = 0.1
Kd = 0.01
Ts = 0.01
duration = 10

# Simulate system without noise
setpoint, plant_output = simulate_system(Kp, Ki, Kd, Ts, duration, noise_amplitude=0)

# Tune PID controller using gradient descent
Kp, Ki, Kd = gradient_descent_controller(Kp, Ki, Kd, Ts, plant_output, setpoint, max_iterations=1000)

# Simulate system with tuned PID controller
setpoint, plant_output = simulate_system(Kp, Ki, Kd, Ts, duration, noise_amplitude=0)

# Filter the plant output using moving average filter
window_size = 10
filtered_output = moving_average_filter(plant_output, window_size)

# Plot results
plt.plot(setpoint, label='Setpoint')
plt.plot(plant_output, label='Plant Output')
plt.plot(filtered_output, label='Filtered Output')
plt.legend()
plt.show()
