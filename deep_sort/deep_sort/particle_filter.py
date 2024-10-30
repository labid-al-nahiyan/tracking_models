import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.zeros((num_particles, state_dim))  # Particle states
        self.weights = np.ones(num_particles) / num_particles  # Particle weights
    
    def initialize(self, init_state, init_std):
        """ Initialize particles around the initial state with Gaussian noise """
        self.particles = np.random.normal(init_state, init_std, size=(self.num_particles, self.state_dim))
    
    def predict(self, motion_std):
        """ Predict the next state for each particle with added Gaussian noise """
        noise = np.random.normal(0, motion_std, size=(self.num_particles, self.state_dim))
        self.particles += noise
    
    def update(self, measurement, meas_std):
        """ Update particle weights based on measurement similarity """
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-0.5 * (distances ** 2) / meas_std ** 2)
        self.weights /= np.sum(self.weights)  # Normalize weights
    
    def resample(self):
        """ Resample particles based on weights to focus on high-probability particles """
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    def estimate(self):
        """ Estimate the state based on weighted average of particles """
        return np.average(self.particles, weights=self.weights, axis=0)
