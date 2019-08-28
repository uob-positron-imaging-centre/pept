/**
 *   pept is a Python library that unifies Positron Emission Particle
 *   Tracking (PEPT) research, including tracking, simulation, data analysis
 *   and visualisation tools
 *
 *   Copyright (C) 2019 Andrei Leonard Nicusan
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * File   : find_cutpoints_ext.h
 * License: License: GNU v3.0
 * Author : Andrei Leonard Nicusan <aln705@student.bham.ac.uk>
 * Date   : 01.07.2019
 */

#ifndef FIND_CUTPOINTS_EXT_H
#define FIND_CUTPOINTS_EXT_H

void  find_cutpoints_ext(const double *sample_lines, double *cutpoints, const unsigned int max_line, const double max_distance, const double *cutoffs);

#endif
