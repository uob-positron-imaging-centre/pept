#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools.
#
#    Copyright (C) 2019-2021 the pept developers
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


# File   : cg_birmingham_method.py
# License: GNU v3.0
# Author : OpenAI Codex (based on existing PEPT codebase style)
# Date   : 05.02.2026


import numpy as np

import pept

from pept.tracking.peptml import find_cutpoints, get_cutoffs, HDBSCAN
from pept.tracking.transformers import LinesCentroids

from .extensions.birmingham_method import birmingham_method


class CGBirminghamMethod(pept.base.LineDataFilter):
    '''Cluster-Guided Birmingham Method.

    This method extends the Birmingham method to support multiple tracers
    within a single sample by first clustering LOR-derived proxy points
    (cutpoints) and then refining each cluster independently using the
    Birmingham solver.
    '''

    def __init__(
        self,
        fopt = 0.5,
        min_cluster_size = 20,
        true_fraction = 0.15,
        max_tracers = 5,
        proxy_distance_threshold = 1.0,
        min_lors_per_cluster = 10,
        cutpoints_max_distance = 0.2,
        cutoffs = None,
        get_used = False,
    ):
        self.fopt = float(fopt)
        self.min_cluster_size = int(min_cluster_size)
        self.true_fraction = float(true_fraction)
        self.max_tracers = int(max_tracers)
        self.proxy_distance_threshold = float(proxy_distance_threshold)
        self.min_lors_per_cluster = int(min_lors_per_cluster)
        self.cutpoints_max_distance = float(cutpoints_max_distance)
        self.cutoffs = cutoffs
        self.get_used = bool(get_used)

        self.clusterer = HDBSCAN(
            self.true_fraction,
            max_tracers = self.max_tracers,
        )


    @property
    def cutoffs(self):
        return self._cutoffs


    @cutoffs.setter
    def cutoffs(self, cutoffs):
        if cutoffs is None:
            self._cutoffs = None
            return

        cutoffs = np.asarray(cutoffs, order = "C", dtype = float)
        if cutoffs.ndim != 1 or len(cutoffs) != 6:
            raise ValueError((
                "\n[ERROR]: cutoffs should be a one-dimensional array with "
                "values [min_x, max_x, min_y, max_y, min_z, max_z]. "
                f"Received {cutoffs}.\n"
            ))

        self._cutoffs = cutoffs


    def generate_proxy_points(self, sample_lines):
        if not isinstance(sample_lines, pept.LineData):
            sample_lines = pept.LineData(sample_lines)

        if len(sample_lines.lines) < 2:
            return np.empty((0, 6))

        if self.cutoffs is None:
            cutoffs = get_cutoffs(sample_lines.lines)
        else:
            cutoffs = self.cutoffs

        points = find_cutpoints(
            sample_lines,
            self.cutpoints_max_distance,
            cutoffs = cutoffs,
            append_indices = True,
        )

        return points.points


    def cluster_points(self, proxy_points):
        proxy_points = np.asarray(proxy_points, dtype = float, order = "C")
        if proxy_points.ndim != 2 or proxy_points.shape[1] < 6:
            return np.empty((0, 5)), []

        if len(proxy_points) < 2:
            return np.empty((0, 5)), []

        points = pept.PointData(
            proxy_points,
            columns = ["t", "x", "y", "z", "line_index1", "line_index2"],
        )
        labelled = self.clusterer.fit_sample(points)

        if len(labelled.points) == 0:
            return np.empty((0, 5)), []

        labels = labelled.points[:, -1].astype(int)
        valid = labels >= 0
        if not np.any(valid):
            return np.empty((0, 5)), []

        centres = []
        cluster_indices = []
        for label in np.unique(labels[valid]):
            mask = labels == label
            if not np.any(mask):
                continue

            centre_t = proxy_points[mask, 0].mean()
            centre_xyz = proxy_points[mask, 1:4].mean(axis = 0)
            cluster_size = int(mask.sum())
            centres.append(
                np.array([centre_t, centre_xyz[0], centre_xyz[1],
                          centre_xyz[2], cluster_size])
            )

            indices = np.unique(proxy_points[mask, 4:6].astype(int).ravel())
            cluster_indices.append(indices)

        if len(centres) == 0:
            return np.empty((0, 5)), []

        return np.vstack(centres), cluster_indices


    @staticmethod
    def _vectorize_lines(lines):
        # Work on a copy so we don't overwrite the original LoR endpoints.
        lines = np.asarray(lines[:, :7], dtype = float, order = "C").copy()
        direction = lines[:, 4:7] - lines[:, 1:4]
        norms = np.linalg.norm(direction, axis = -1)

        valid = norms > 0
        if np.any(valid):
            direction[valid] /= norms[valid][:, np.newaxis]
        direction[~valid] = 0.0

        lines[:, 4:7] = direction
        return lines


    def assign_lors_to_cluster(self, centre, lines, vector_lines = None):
        if len(lines) == 0:
            return np.empty((0, lines.shape[1]))

        centre = np.asarray(centre, dtype = float)
        centre_xyz = centre[1:4]

        if vector_lines is None:
            vector_lines = self._vectorize_lines(lines)

        d2 = LinesCentroids.distance_matrix(centre_xyz, vector_lines)
        d2 = np.maximum(d2, 0.0)

        threshold2 = self.proxy_distance_threshold ** 2
        mask = d2 <= threshold2

        if mask.sum() >= self.min_lors_per_cluster:
            return lines[mask]

        # Fallback: ensure enough lines by taking the closest ones
        k = min(max(self.min_lors_per_cluster, 1), len(lines))
        indices = np.argpartition(d2, k - 1)[:k]
        return lines[indices]


    def refine_cluster(self, cluster_lines):
        if len(cluster_lines) < max(2, self.min_lors_per_cluster):
            return None, None

        cluster_lines = np.asarray(cluster_lines, dtype = float, order = "C")
        location, used = birmingham_method(cluster_lines, self.fopt)
        return location, used


    def fit_sample(self, sample):
        if not isinstance(sample, pept.LineData):
            sample = pept.LineData(sample)

        lines = sample.lines
        attrs = sample.extra_attrs()

        if len(lines) < 2:
            return pept.PointData(
                np.empty((0, 5)),
                columns = ["t", "x", "y", "z", "error"],
                **attrs,
            )

        proxy_points = self.generate_proxy_points(sample)
        if len(proxy_points) == 0:
            return pept.PointData(
                np.empty((0, 5)),
                columns = ["t", "x", "y", "z", "error"],
                **attrs,
            )

        centres, cluster_indices = self.cluster_points(proxy_points)
        if len(centres) == 0:
            return pept.PointData(
                np.empty((0, 5)),
                columns = ["t", "x", "y", "z", "error"],
                **attrs,
            )

        locations = []
        used_lines = []

        for centre, indices in zip(centres, cluster_indices):
            if len(centre) and centre[-1] < self.min_cluster_size:
                continue

            cluster_lines = lines[indices] if len(indices) else lines[0:0]

            location, used = self.refine_cluster(cluster_lines)
            if location is None:
                continue

            locations.append(location)
            if self.get_used:
                used_lines.append((cluster_lines, used))

        if len(locations) == 0:
            return pept.PointData(
                np.empty((0, 5)),
                columns = ["t", "x", "y", "z", "error"],
                **attrs,
            )

        points = pept.PointData(
            np.vstack(locations),
            columns = ["t", "x", "y", "z", "error"],
            **attrs,
        )

        if self.get_used:
            lines_list = []
            for cluster_lines, used in used_lines:
                lines_list.append(sample.copy(
                    data = np.c_[cluster_lines, used],
                    columns = sample.columns + ["used"],
                ))
            points.attrs["_lines"] = lines_list

        return points


    def solve(self, lors):
        locations = self.fit_sample(lors)
        if not isinstance(locations, pept.PointData):
            return []

        results = []
        for row in locations.points:
            results.append({
                "position": (row[1], row[2], row[3]),
                "error": row[4],
            })

        return results
