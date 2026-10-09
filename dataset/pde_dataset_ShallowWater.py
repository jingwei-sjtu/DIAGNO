# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import torch
import torch.nn as nn
from torch_harmonics.sht import RealSHT, InverseRealSHT, RealVectorSHT, InverseRealVectorSHT
from torch_harmonics.quadrature import (
    _precompute_longitudes, 
    legendre_gauss_weights, 
    lobatto_weights, 
    clenshaw_curtiss_weights)
import math
import numpy as np
from math import ceil

class ShallowWaterSolver(nn.Module):
    def __init__(self, nlat, nlon, dt, lmax=None, mmax=None, grid="equiangular", radius=6.37122E6, \
                 omega=7.292E-5, gravity=9.80616, havg=10.e3, hamp=120.):
        super().__init__()

        self.dt = dt
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

        self.register_buffer('radius', torch.as_tensor(radius, dtype=torch.float64))
        self.register_buffer('omega', torch.as_tensor(omega, dtype=torch.float64))
        self.register_buffer('gravity', torch.as_tensor(gravity, dtype=torch.float64))
        self.register_buffer('havg', torch.as_tensor(havg, dtype=torch.float64))
        self.register_buffer('hamp', torch.as_tensor(hamp, dtype=torch.float64))

        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.vsht = RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.ivsht = InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)

        self.lmax = lmax or self.sht.lmax
        self.mmax = lmax or self.sht.mmax

        if self.grid == "legendre-gauss":
            cost, quad_weights = legendre_gauss_weights(self.nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, quad_weights = lobatto_weights(self.nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, quad_weights = clenshaw_curtiss_weights(self.nlat, -1, 1)

        quad_weights = quad_weights.reshape(-1, 1)

        lats = -torch.arcsin(cost)
        lons = _precompute_longitudes(self.nlon)

        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l = l.expand(self.lmax, self.mmax)
        lap = - l * (l + 1) / self.radius**2
        invlap = - self.radius**2 / l / (l + 1)
        invlap[0] = 0.

        coriolis = 2 * self.omega * torch.sin(lats).reshape(self.nlat, 1)

        hyperdiff = torch.exp(torch.asarray((-self.dt / 2 / 3600.)*(lap / lap[-1, 0])**4))

        self.register_buffer('lats', lats)
        self.register_buffer('lons', lons)
        self.register_buffer('l', l)
        self.register_buffer('lap', lap)
        self.register_buffer('invlap', invlap)
        self.register_buffer('coriolis', coriolis)
        self.register_buffer('hyperdiff', hyperdiff)
        self.register_buffer('quad_weights', quad_weights)

    def grid2spec(self, ugrid):
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        return self.isht(uspec)

    def vrtdivspec(self, ugrid):
        vrtdivspec = self.lap * self.radius * self.vsht(ugrid)
        return vrtdivspec

    def getuv(self, vrtdivspec):
        return self.ivsht( self.invlap * vrtdivspec / self.radius)

    def gethuv(self, uspec):
        """Compute height and wind vector from spectral coefficients."""
        hgrid = self.spec2grid(uspec[:1])
        uvgrid = self.getuv(uspec[1:])
        return torch.cat((hgrid, uvgrid), dim=-3)

    def potential_vorticity(self, uspec):
        ugrid = self.spec2grid(uspec)
        pvrt = (0.5 * self.havg * self.gravity / self.omega) * (ugrid[1] + self.coriolis) / ugrid[0]
        return pvrt

    def dimensionless(self, uspec):
        uspec[0] = (uspec[0] - self.havg * self.gravity) / self.hamp / self.gravity
        uspec[1:] = uspec[1:] * self.radius / torch.sqrt(self.gravity * self.havg)
        return uspec

    def dudtspec(self, uspec):
        dudtspec = torch.zeros_like(uspec)

        ugrid = self.spec2grid(uspec)
        uvgrid = self.getuv(uspec[1:])

        tmp = uvgrid * (ugrid[1] + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        dudtspec[2] = tmpspec[0]
        dudtspec[1] = -1 * tmpspec[1]

        tmp = uvgrid * ugrid[0]
        tmp = self.vrtdivspec(tmp)
        dudtspec[0] = -1 * tmp[1]

        tmpspec = self.grid2spec(ugrid[0] + 0.5 * (uvgrid[0]**2 + uvgrid[1]**2))
        dudtspec[2] = dudtspec[2] - self.lap * tmpspec

        return dudtspec

    def galewsky_initial_condition(self):
        device = self.lap.device

        umax = 80.
        phi0 = torch.asarray(torch.pi / 7., device=device)
        phi1 = torch.asarray(0.5 * torch.pi - phi0, device=device)
        phi2 = 0.25 * torch.pi
        en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0)**2, device=device))
        alpha = 1. / 3.
        beta = 1. / 15.

        lats, lons = torch.meshgrid(self.lats, self.lons)

        u1 = (umax/en)*torch.exp(1./((lats-phi0)*(lats-phi1)))
        ugrid = torch.where(torch.logical_and(lats < phi1, lats > phi0), u1, torch.zeros(self.nlat, self.nlon, device=device))
        vgrid = torch.zeros((self.nlat, self.nlon), device=device)
        hbump = self.hamp * torch.cos(lats) * torch.exp(-((lons-torch.pi)/alpha)**2) * torch.exp(-(phi2-lats)**2/beta)

        ugrid = torch.stack((ugrid, vgrid))
        vrtdivspec = self.vrtdivspec(ugrid)
        vrtdivgrid = self.spec2grid(vrtdivspec)

        tmp = ugrid * (vrtdivgrid + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        phispec = self.invlap*tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity*(self.havg + hbump))

        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=vrtdivspec.dtype, device=device)
        uspec[0] = phispec
        uspec[1:] = vrtdivspec

        return torch.tril(uspec)

    def random_initial_condition(self, mach=0.1) -> torch.Tensor:
        device = self.lap.device
        ctype = torch.complex128 if self.lap.dtype == torch.float64 else torch.complex64

        llimit = mlimit = 60

        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=ctype, device=self.lap.device)
        uspec[:, :llimit, :mlimit] = torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit+1), device=device, dtype=ctype)) * torch.randn_like(uspec[:, :llimit, :mlimit])

        uspec[0] = self.gravity * self.hamp * uspec[0]
        uspec[0, 0, 0] += torch.sqrt(torch.tensor(4 * torch.pi, device=device, dtype=ctype)) * self.havg * self.gravity
        uspec[1:] = mach * uspec[1:] * torch.sqrt(self.gravity * self.havg) / self.radius

        return torch.tril(uspec)

    def timestep(self, uspec: torch.Tensor, nsteps: int) -> torch.Tensor:
        dudtspec = torch.zeros(3, 3, self.lmax, self.mmax, dtype=uspec.dtype, device=uspec.device)

        inew = 0
        inow = 1
        iold = 2

        for iter in range(nsteps):
            dudtspec[inew] = self.dudtspec(uspec)

            if iter == 0:
                dudtspec[inow] = dudtspec[inew]
                dudtspec[iold] = dudtspec[inew]
            elif iter == 1:
                dudtspec[iold] = dudtspec[inew]

            uspec = uspec + self.dt*( (23./12.) * dudtspec[inew] - (16./12.) * dudtspec[inow] + (5./12.) * dudtspec[iold] )

            uspec[1:] = self.hyperdiff * uspec[1:]

            inew = (inew - 1) % 3
            inow = (inow - 1) % 3
            iold = (iold - 1) % 3

        return uspec

    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        dlon = 2 * torch.pi / self.nlon
        radius = 1 if dimensionless else self.radius
        if polar_opt > 0:
            out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
        else:
            out = torch.sum(ugrid * self.quad_weights * dlon * radius**2, dim=(-2, -1))
        return out


class PdeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dt,
        nsteps,
        dims=(384, 768),
        grid="equiangular",
        pde="shallow water equations",
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
        
    ):
        self.num_examples = num_examples
        self.device = device
        self.stream = stream

        self.nlat = dims[0]
        self.nlon = dims[1]
        self.grid = grid

        self.nsteps = nsteps
        self.normalize = normalize
        self.nsteps_multiplier = 1
        
        if pde == "shallow water equations":
            lmax = ceil(self.nlat / 3)
            mmax = lmax
            dt_solver = dt / float(self.nsteps)
            
            self.solver = ShallowWaterSolver(self.nlat, self.nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid).to(self.device).float()
        else:
            raise NotImplementedError

        self.set_initial_condition(ictype=initial_condition)

        if self.normalize:
            inp0, _ = self._get_sample()
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)

    def __len__(self):
        length = self.num_examples if self.ictype == "random" else 1
        return length

    def set_initial_condition(self, ictype="random"):
        self.ictype = ictype

    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples

    def _get_sample(self):
        if self.ictype == "random":
            inp = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == "galewsky":
            inp = self.solver.galewsky_initial_condition()

        tar = self.solver.timestep(inp, self.nsteps * self.nsteps_multiplier)
        
        inp = self.solver.gethuv(inp)
        tar = self.solver.gethuv(tar)

        return inp, tar

    def __getitem__(self, index):
        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
                    tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        return inp.clone(), tar.clone()