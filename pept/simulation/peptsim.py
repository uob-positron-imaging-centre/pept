#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : PEPTsimulator.py
# License           : License: GNU v3.0
# Author            : Andrei Leonard Nicusan <aln705@student.bham.ac.uk>
# Date              : 01.07.2019


import  numpy                   as          np
import  matplotlib.pyplot       as          plt
from    mpl_toolkits.mplot3d    import      Axes3D

from    tqdm                    import      tqdm




class Shape:
    '''Class of shape functions which return a random [x, y, z] point inside a
    given shape (sphere, cylinder, etc) centred around the origin.
    '''

    def __init__(self, x=0.5, y=0.5, z=0.5):
        '''x, y, z are coordinate ranges for the three
        major axes. For example:
        For a sphere, x is radius; y and z are unused
        For a cylinder, x is radius, y is height; z is unused
        For a parallelipiped, x is width, y is depth, z is height

        '''
        self.x = x
        self.y = y
        self.z = z


    def rotateX3D(vec, angleX):
        '''Rotate vec around the X axis by angleX radians.
        '''
        rot = np.array([ [1, 0, 0], [0, np.cos(angleX), -np.sin(angleX)], [0, np.sin(angleX), np.cos(angleX)] ])

        return (rot @ vec)



    def sphere(self):
        '''Simulate as spherical coordinates.
        '''
        r = np.random.uniform(0, self.x)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return [x, y, z]


    def cylinder(self, angleX=0, angleY=0, angleZ=0):
        '''Simulate as cylindrical coordinates.

        Cylinder is horizontal on the x axis
        => self.x is max radius
        => self.y is max height
        '''
        r = np.random.uniform(0, self.x)

        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-self.y, self.y)




class Noise:
    '''Add noise to simulated PEPT data.
    '''

    def __init__(self, trajectory, xMax, yMax, zMax):
        '''Trajectory row: [time, X, Y, Z]
        '''
        self.trajectory = trajectory
        self.xMax = xMax
        self.yMax = yMax
        self.zMax = zMax


    # Better to use addNoiseToPEPT. Add noise to PEPT screen projection, rather than to the
    # actual trajectory
    def create_trajectory_noise(self, numberOfPoints):
        '''Generate random X, Y, Z coordinates for noise points
        '''
        xNoise = np.random.uniform(0, self.xMax, numberOfPoints)
        yNoise = np.random.uniform(0, self.yMax, numberOfPoints)
        zNoise = np.random.uniform(0, self.zMax, numberOfPoints)

        # Generate times for noise points between the start time and end time of given trajectory
        timeNoise = np.random.uniform(self.trajectory[0][0], self.trajectory[-1][0], numberOfPoints)
        timeNoise = np.sort(timeNoise)

        self.noise = np.stack((timeNoise, xNoise, yNoise, zNoise), axis=1)
        return self.noise


    def get_trajectory_with_noise(self, noiseRatio):
        '''Get trajectory with noise.
        '''

        self.numberOfNoisePoints = int(len(self.trajectory) * noiseRatio)
        self.createTrajectoryNoise(self.numberOfNoisePoints)

        # Find indices at which the noise should be inserted to maintain time order of trajectory
        insertionIndices = np.searchsorted(self.trajectory[:, 0], self.noise[:, 0], side='left')
        self.trajectoryWithNoise = np.insert(self.trajectory, insertionIndices, self.noise, axis=0)

        return self.trajectoryWithNoise


    def add_noise_to_pept(self, pept_data, number_of_noise_points):
        '''PEPTdata row: [time, X1, Y1, X2, Y2]
        Generate random (X, Y) pairs coordinates for noise points in PEPTdata
        '''

        x1Noise = np.random.uniform(0, self.xMax, number_of_noise_points)
        y1Noise = np.random.uniform(0, self.yMax, number_of_noise_points)

        x2Noise = np.random.uniform(0, self.xMax, number_of_noise_points)
        y2Noise = np.random.uniform(0, self.yMax, number_of_noise_points)

        # Generate times for noise lines between the start time and end time of given trajectory
        time_noise = np.random.uniform(self.trajectory[0][0], self.trajectory[-1][0], number_of_noise_points)
        time_noise = np.sort(time_noise)

        self.pept_noise = np.stack((time_noise, x1Noise, y1Noise, x2Noise, y2Noise), axis=1)
        insertion_indices = np.searchsorted(pept_data[:, 0], self.pept_noise[:, 0], side='left')
        self.pept_data_with_noise = np.insert(pept_data, insertion_indices, self.pept_noise, axis=0)

        return self.pept_data_with_noise


    def add_spread_to_pept(self, pept_data, max_spread, depth):
        '''PEPTdata row: [time, X1, Y1, X2, Y2]
        '''

        pept_data_copy = np.copy(pept_data)

        # Add positional spread to X1, Y1, X2, Y2
        spread = np.random.uniform(-max_spread, max_spread, (len(pept_data), 4))
        pept_data_copy[:, 1:5] = pept_data_copy[:, 1:5] + spread

        # Add angular spread to X1, Y1, X2, Y2
        # x1_angular_spread = x1 - (x2 - x1) / sep * depth * alpha
        alpha = np.random.uniform(0, 1, (len(pept_data), 4))

        self.pept_data_with_spread = np.copy(pept_data_copy)
        self.pept_data_with_spread[:, 1] = pept_data_copy[:, 1] - (pept_data_copy[:, 3] - pept_data_copy[:, 1]) / self.zMax * alpha[:, 0] * depth
        self.pept_data_with_spread[:, 3] = pept_data_copy[:, 3] + (pept_data_copy[:, 3] - pept_data_copy[:, 1]) / self.zMax * alpha[:, 1] * depth

        self.pept_data_with_spread[:, 2] = pept_data_copy[:, 2] - (pept_data_copy[:, 4] - pept_data_copy[:, 2]) / self.zMax * alpha[:, 2] * depth
        self.pept_data_with_spread[:, 4] = pept_data_copy[:, 4] + (pept_data_copy[:, 4] - pept_data_copy[:, 2]) / self.zMax * alpha[:, 3] * depth

        return self.pept_data_with_spread




class Simulator:
    '''Simulate PEPT data.
    '''

    def __init__(
        self,
        trajectory,
        sampling_times,
        shape_function,
        separation = 712,
        decay_energy = 0.6335,
        Zeff = 7.22,
        Aeff = 13,
        x_max = 500,
        y_max = 500
    ):
        '''Simulator class constructor.
        '''

        # Trajectory row: [time, X, Y, Z]
        self.trajectory = trajectory

        # Timepoints at which positrons are emitted
        self.sampling_times = sampling_times

        # Separation between PEPT screens
        self.separation = separation

        # Function which returns an [x, y, z] positions around a centre
        self.shape_function = shape_function

        # Beta+ endpoint/decay energy of tracer, in MeV. F-18 has a Beta+ decay energy of 0.6335 MeV
        self.decay_energy = decay_energy

        # Effective atomic weight Aeff and atomic number Zeff. For water, Aeff = 13, Zeff = 7.22
        self.Aeff = Aeff
        self.Zeff = Zeff

        # PEPT screens sizes
        self.x_max = x_max
        self.y_max = y_max


    def simulate(self):

        # Indices of trajectory times closest to sampling times
        location_indices = np.searchsorted(self.trajectory[:, 0], self.sampling_times, side='left')
        number_of_samples = len(location_indices)

        # PEPTdata row: [time, X1, Y1, X2, Y2]
        pept_data = []

        # For every position on the trajectory corresponding to a samplingTime,
        # generate a PEPTdata row
        for i in tqdm(range(0, number_of_samples)):

            # particle index in the trajectory for ray tracing
            particle_index = location_indices[i]

            # particleIndex for insertion could be the last one
            if particle_index >= len(self.trajectory):
                particle_index = len(self.trajectory) - 1

            # The positron emission point within the particle
            [xshape, yshape, zshape] = self.shape_function()
            xp = self.trajectory[particle_index][1] + xshape
            yp = self.trajectory[particle_index][2] + yshape
            zp = self.trajectory[particle_index][3] + zshape

            # Calculate incident/kinetic energy Ei of positron from distribution
            # approximated as gaussian

            # mean
            mu = self.decay_energy / 2
            # 99.7% of data will lie between [0, decayEnergy]
            sigma = mu / 3

            Ei = np.random.normal(mu, sigma)

            # Find positron annihilation point, corresponding to gamma radiation emission
            # using Palmer et al., 1992 equations
            b1 = 4.569 * self.Aeff / self.Zeff**1.209
            b2 = 1 / (2.873 - 0.02309 * self.Zeff)
            Rex = b1 * Ei * Ei / (b2 + Ei)

            # Annihilation point [xa, ya, za] from positron emission point
            [xa, ya, za] = np.random.normal(0, Rex / 2, 3)

            # The gamma ray emission point will therefore be
            # positron emission point + positron annihilation point
            xp += xa
            yp += ya
            zp += za

            # Try at most 100 random phi and theta angles until the ray falls within
            # the PEPT screens
            ray_try = 0
            while ray_try < 100:
                phi = np.random.uniform(0, np.pi)
                theta = np.random.uniform(0, np.pi)

                if phi == np.pi / 2 or theta == np.pi / 2:
                    continue

                x1 = xp - zp * np.tan(phi)
                x2 = xp + (self.separation - zp) * np.tan(phi)

                y1 = yp - zp / np.cos(phi) * np.tan(theta)
                y2 = yp + (self.separation - zp) / np.cos(phi) * np.tan(theta)

                # Check the rays fell within the PEPT screens
                if 0 < x1 < self.x_max and 0 < x2 < self.x_max:
                    if 0 < y1 < self.y_max and 0 < y2 < self.y_max:
                        pept_data.append(np.array([ self.sampling_times[i], x1, y1, x2, y2 ]))
                        break
                ray_try += 1

        self.pept_data = np.array(pept_data)
        return self.pept_data


    def add_noise(self, noise_ratio):
        noiser = Noise(self.trajectory, self.x_max, self.y_max, self.separation)
        number_of_noise_points = int( noise_ratio * len(self.sampling_times) )
        self.pept_data_with_noise = noiser.add_noise_to_pept(self.pept_data, number_of_noise_points)


    def add_spread(self, max_spread = 4, depth = 16):
        noiser = Noise(self.trajectory, self.x_max, self.y_max, self.separation)
        self.pept_data_with_noise = noiser.add_spread_to_pept(self.pept_data, max_spread, depth)


    def add_noise_and_spread(self, noise_ratio, max_spread = 4, depth = 16):
        noiser = Noise(self.trajectory, self.x_max, self.y_max, self.separation)
        number_of_noise_points = int( noise_ratio * len(self.sampling_times) )
        self.pept_data_with_noise = noiser.add_noise_to_pept(self.pept_data, number_of_noise_points)
        self.pept_data_with_noise = noiser.add_spread_to_pept(self.pept_data_with_noise, max_spread, depth)


    def change_trajectory(self, new_trajectory):
        self.trajectory = new_trajectory


    def change_sampling_times(self, new_sampling_times):
        self.sampling_times = new_sampling_times


    def change_shape(self, new_shape_function):
        self.shape_function = new_shape_function


    def write_csv(self, fname):
        np.savetxt(fname, self.pept_data, delimiter = '   ', newline = '\n  ')


    def write_noise_csv(self, fname):
        np.savetxt(fname, self.pept_data_with_noise, delimiter = '   ', newline = '\n  ')


