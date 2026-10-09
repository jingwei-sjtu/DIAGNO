# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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
import torch.amp as amp
import torch.nn.functional as F
from typing import Optional
from abc import ABC, abstractmethod

from torch_harmonics.quadrature import _precompute_latitudes


def get_quadrature_weights(nlat: int, nlon: int, grid: str, tile: bool = False, normalized: bool = True) -> torch.Tensor:
    # area weights
    _, q = _precompute_latitudes(nlat=nlat, grid=grid)
    q = q.reshape(-1, 1) * 2 * torch.pi / nlon

    # numerical precision can be an issue here, make sure it sums to 1:
    if normalized:
        q = q / torch.sum(q) / float(nlon)

    if tile:
        q = torch.tile(q, (1, nlon)).contiguous()

    return q.to(torch.float32)



class SphericalLossBase(nn.Module, ABC):
    """Abstract base class for spherical losses that handles common initialization and integration."""

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", normalized: bool = True):
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

        # get quadrature weights - these sum to 1!
        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid, normalized=normalized)
        self.register_buffer("quad_weights", q)

    def _integrate_sphere(self, ugrid, mask=None):
        if mask is None:
            out = torch.sum(ugrid * self.quad_weights, dim=(-2, -1))
        elif mask is not None:
            out = torch.sum(mask * ugrid * self.quad_weights, dim=(-2, -1)) / torch.sum(mask * self.quad_weights, dim=(-2, -1))
        return out

    @abstractmethod
    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        """Abstract method that must be implemented by child classes to compute loss terms.

        Args:
            prd (torch.Tensor): Prediction tensor
            tar (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Computed loss term before integration
        """
        pass

    def _post_integration_hook(self, loss: torch.Tensor) -> torch.Tensor:
        """Post-integration hook. Commonly used for the roots in Lp norms"""
        return loss

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Common forward pass that handles masking and reduction.

        Args:
            prd (torch.Tensor): Prediction tensor
            tar (torch.Tensor): Target tensor
            mask (Optional[torch.Tensor], optional): Mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Final loss value
        """
        loss_term = self._compute_loss_term(prd, tar)
        # Integrate over the sphere for each item in the batch
        loss = self._integrate_sphere(loss_term, mask)
        # potentially call root
        loss = self._post_integration_hook(loss)
        # Average the loss over the batch dimension
        return torch.mean(loss)


class SquaredL2LossS2(SphericalLossBase):
    """
    Area-weighted Squared L2 loss (Mean Squared Error) for spherical data.
    Computes the squared difference, weighted by the physical area of each grid cell.
    """
    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        return torch.square(prd - tar)


class L1LossS2(SphericalLossBase):
    """
    Area-weighted L1 loss (Mean Absolute Error) for spherical data.
    Computes the absolute difference, weighted by the physical area of each grid cell.
    """
    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        return torch.abs(prd - tar)


class L2LossS2(SquaredL2LossS2):
    """
    Area-weighted L2 loss (Root Mean Squared Error) for spherical data.
    Computes the square root of the area-weighted MSE.
    """
    def _post_integration_hook(self, loss: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(loss)


class UnweightedL1Loss(nn.Module):
    """
    Unweighted L1 loss (Mean Absolute Error).
    Computes the standard arithmetic mean of the absolute differences without area weighting.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = torch.abs(prd - tar)
        
        # Aggregate over spatial dimensions (H, W) which are the last two dimensions
        if mask is not None:
            spatial_mean_loss = (loss * mask).sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8)
        else:
            spatial_mean_loss = loss.mean(dim=(-2, -1))
            
        # Average over batch (and channel if applicable)
        return torch.mean(spatial_mean_loss)


class UnweightedSquaredL2Loss(nn.Module):
    """
    Unweighted Squared L2 loss (Mean Squared Error).
    Computes the standard arithmetic mean of the squared differences without area weighting.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = torch.square(prd - tar)
        
        if mask is not None:
            spatial_mean_loss = (loss * mask).sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8)
        else:
            spatial_mean_loss = loss.mean(dim=(-2, -1))
            
        return torch.mean(spatial_mean_loss)


class UnweightedL2Loss(UnweightedSquaredL2Loss):
    """
    Unweighted L2 loss (Root Mean Squared Error).
    Computes the square root of the unweighted MSE for each sample, then averages over the batch.
    """
    def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = torch.square(prd - tar)
        
        if mask is not None:
            spatial_mean_loss = (loss * mask).sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8)
        else:
            spatial_mean_loss = loss.mean(dim=(-2, -1))
            
        # Take the square root BEFORE averaging over the batch dimension
        return torch.mean(torch.sqrt(spatial_mean_loss))
    

class ACCScoreS2(SphericalLossBase):
    """
    Anomaly Correlation Coefficient (ACC) for spherical data.
    
    Computed as the weighted Pearson correlation coefficient between predicted 
    and target anomalies. Area weighting is handled automatically via quadrature weights.
    
    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    grid : str, optional
        Grid type, by default "equiangular"
    """

    def _compute_loss_term(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        # ACC requires integration before division, skipping point-wise term computation.
        pass

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, clim: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the weighted Anomaly Correlation Coefficient (ACC).
        
        Args:
            prd (torch.Tensor): Prediction tensor [B, C, H, W]
            tar (torch.Tensor): Target tensor [B, C, H, W]
            clim (Optional[torch.Tensor], optional): Climatology tensor. 
                If None, assumes `prd` and `tar` are already anomalies. Defaults to None.
            mask (Optional[torch.Tensor], optional): Mask tensor (e.g., ocean mask). Defaults to None.

        Returns:
            torch.Tensor: Global ACC score (averaged over batch and channels by default)
        """
        # 1. Compute anomalies
        if clim is not None:
            prd_anom = prd - clim
            tar_anom = tar - clim
        else:
            prd_anom = prd
            tar_anom = tar

        # 2. Compute numerator: weighted covariance of predicted and target anomalies
        # cov = Integral( prd_anom * tar_anom )
        cov = self._integrate_sphere(prd_anom * tar_anom, mask)

        # 3. Compute denominator: geometric mean of their respective weighted variances
        # var_p = Integral( prd_anom^2 ), var_t = Integral( tar_anom^2 )
        var_p = self._integrate_sphere(torch.square(prd_anom), mask)
        var_t = self._integrate_sphere(torch.square(tar_anom), mask)

        # 4. Compute ACC and prevent division by zero
        eps = 1e-8
        acc = cov / torch.sqrt(var_p * var_t + eps)

        # _integrate_sphere reduces H, W dimensions, resulting in acc shape [B, C].
        # Return the mean over the batch to maintain consistency with other Metrics returning scalars.
        return acc.mean(dim=0)