import numpy as np
import scipy.linalg


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
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        
        return mean, covariance 


    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha