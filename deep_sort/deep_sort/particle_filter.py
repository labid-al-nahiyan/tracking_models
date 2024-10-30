import numpy as np

class ParticleFilter:
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.particles = None  # This will hold particle states
        self.weights = None  # This will hold particle weights

    def initiate(self, state):
        """Initialize particles around the initial state."""
        self.particles = np.random.normal(state, scale=0.5, size=(self.num_particles, len(state)))
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Return the initial mean and covariance
        mean = np.mean(self.particles, axis=0)
        covariance = np.cov(self.particles, rowvar=False)
        return mean, covariance

    def predict(self):
        """Predict the next state for each particle."""
        # Implement the prediction step
        # For example, applying a motion model
        noise = np.random.normal(0, 0.1, self.particles.shape)
        self.particles += noise

    def update(self, measurement):
        """Update the particle weights based on the measurement."""
        # Implement the measurement update
        # For example, calculating weights based on likelihood
        for i in range(self.num_particles):
            self.weights[i] = self._measurement_likelihood(self.particles[i], measurement)

        # Normalize weights
        self.weights /= np.sum(self.weights)

    def _measurement_likelihood(self, particle, measurement):
        """Calculate the likelihood of a measurement given a particle state."""
        # Implement a likelihood function (e.g., Gaussian)
        return np.exp(-0.5 * np.sum((particle - measurement) ** 2))

    def resample(self):
        """Resample particles based on weights."""
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)  # Reset weights after resampling

    def update(self, measurement):
        """Update weights based on the measurement and resample particles."""
        # Calculate weights based on likelihood (e.g., Gaussian likelihood)
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-0.5 * (distances ** 2))  # Example: Gaussian likelihood
        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)  # Normalize weights

        # Resampling step
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)  # Reset weights

    def get_estimate(self):
        """Return the weighted mean and covariance of the particles."""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        covariance = np.cov(self.particles, rowvar=False)
        return mean, covariance