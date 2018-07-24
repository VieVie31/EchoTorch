# -*- coding: utf-8 -*-
#
# File : echotorch/nn/LiESNCell.py
# Description : An Leaky-Integrated Echo State Network layer.
# Date : 26th of January, 2018
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

import torch
import torch.sparse
import torch.nn as nn
from torch.autograd import Variable
from .LiESNCell import LiESNCell
from RRCell import RRCell
import matplotlib.pyplot as plt


# Conceptor network
class ConceptorNet(LiESNCell):
    """
    Conceptor network
    """

    # Constructor
    def __init__(self, observer_dim, n0=50, aperture=0.5, level=0, *args, **kwargs):
        """
        Constructor
        :param n0:
        :param aperture:
        :param args:
        :param kwargs:
        """
        super(LiESNCell, self).__init__(*args, **kwargs)

        # Params
        self.observer_dim = observer_dim
        self.n0 = n0
        self.aperture = aperture
        self.level = level

        # Pattern layer
        if level == 0 or level == 1:
            self.pattern = RRCell(self.output_dim, self.output_dim)
        else:
            self.pattern = RRCell(self.input_dim, self.output_dim)
        # end if

        # Conceptor layer
        self.conceptors = list()
        for i in range(observer_dim):
            self.conceptors.append(RRCell(self.output_dim, self.output_dim, aperture))
        # end for

        # Ouput layer
        self.observer = RRCell(self.output_dim, self.observer_dim, torch.pow(aperture, -2))
    # end __init__

    ###############################################
    # PUBLIC
    ###############################################

    # Forward
    def forward(self, u, j, y=None):
        """
        Forward
        :param u: Input signal.
        :return: Resulting hidden states.
        """
        # Time length
        time_length = int(u.size()[1])

        # Number of batches
        n_batches = int(u.size()[0])

        # Outputs
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim))
        outputs = outputs.cuda() if self.hidden.is_cuda else outputs

        # Patterns
        if self.level == 0 and self.level == 1:
            patterns = Variable(torch.zeros(n_batches, time_length, self.output_dim))
        else:
            patterns = Variable(torch.zeros(n_batches, time_length, self.input_dim))
        # end if
        patterns = patterns.cuda() if self.hidden.is_cuda else patterns

        # States
        hidden_states = Variable(torch.zeros(n_batches, time_length, self.output_dim))
        hidden_states = hidden_states.cuda() if self.hidden.is_cuda else hidden_states

        # For each batch
        for b in range(n_batches):
            # Reset hidden layer
            self.reset_hidden()

            # For each steps
            for t in range(time_length):
                # Current input
                ut = u[b, t]

                # Compute input layer
                u_win = self.w_in.mv(ut)

                # Apply W to x
                x_w = self.w.mv(self.hidden)

                # Add everything
                x = u_win + x_w + self.w_bias

                # Save states
                hidden_states[b, t] = self.hidden

                # Get pattern
                if self.level == 0:
                    patterns[b, t] = x_w + u_win
                elif self.level == 1:
                    patterns[b, t] = u_win
                else:
                    patterns[b, t] = ut
                # end if

                # Apply activation function
                x = self.nonlin_func(x)

                # Add to outputs
                self.hidden.data = (self.hidden.mul(1.0 - self.leaky_rate) + x.view(self.output_dim).mul(self.leaky_rate)).data

                # New last state
                outputs[b, t] = self.hidden
            # end for
        # end for

        # Learn pattern
        self.pattern(patterns, hidden_states)

        # Learn conceptor
        self.conceptors[j](outputs, outputs)

        # Learn observer
        return self.observer(outputs, y)
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

# end ConceptorNet
