"""live_sim a evolution simulation
Copyright (C) 2019  Leo Basov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
long with this program. If not, see <https://www.gnu.org/licenses/>."""

import math

def modified_sigmoid(value):
	try:
		return 1.0/(1.0 + math.exp(-4.9*value))
	except OverflowError:
		if value > 0:
			return 1.0
		else:
			return 0.0

def linear_interpol(x_0, y_0, x_1, y_1, x):
	frac = (x - x_0)/(x_1 - x_0)

	return y_0*(1.0 - frac) + y_1*frac
