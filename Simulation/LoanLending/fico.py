# This file is from github.com/lydiatliu/delayedimpact
# Please follow the license below.

# BSD 3-Clause License

# Copyright (c) 2018, lydiatliu
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

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
"""Reading FICO Data"""

from __future__ import print_function
import numpy as np
import pandas as pd

DATA_DIR = '../'

PERF = 'transrisk_performance_by_race_ssa.csv'
CDF_BY_RACE = 'transrisk_cdf_by_race_ssa.csv'
OVERALL = 'totals.csv'

FILES = dict(cdf_by_race=CDF_BY_RACE,
             performance_by_race=PERF,
             overview=OVERALL
             )


def cleanup_frame(frame):
    """Make the columns have better names, and ordered in a better order"""
    frame = frame.rename(columns={'Non- Hispanic white': 'White'})
    frame = frame.reindex(columns=['Asian', 'White', 'Hispanic', 'Black'])
    return frame


def read_totals(data_dir=DATA_DIR):
    """Read the total number of people of each race"""
    # frame = cleanup_frame(pd.DataFrame.from_csv(data_dir + FILES['overview']))
    frame = cleanup_frame(pd.read_csv(data_dir + FILES['overview'], index_col=0, parse_dates=True))
    return {r: frame[r]['SSA'] for r in frame.columns}


def convert_percentiles(idx):
    """Convert percentiles"""
    pdf = [(300, 2.1),
           (350, 4.2),
           (400, 5.4),
           (450, 6.5),
           (500, 7.9),
           (550, 9.6),
           (600, 12.0),
           (650, 13.8),
           (700, 17.0),
           (750, 15.8),
           (800, 5.7),
           (850, 0),
           ]

    def convert_one(x):
        partial = 0
        for ((v, s), (v2, _)) in zip(pdf, pdf[1:]):
            if partial + s >= x:
                return v + (v2 - v) * (x - partial) / s
            partial += s

    return np.array(list(map(convert_one, idx)))


def parse_data(data_dir=DATA_DIR, filenames=None):
    """Parse sqf data set."""
    if filenames is None:
        filenames = [FILES['cdf_by_race'], FILES['performance_by_race']]

    # cdfs = cleanup_frame(pd.DataFrame.from_csv(data_dir + filenames[0]))
    # performance = 100 - cleanup_frame(pd.DataFrame.from_csv(data_dir + filenames[1]))
    cdfs = cleanup_frame(pd.read_csv(data_dir + filenames[0], index_col=0, parse_dates=True))
    performance = 100 - cleanup_frame(pd.read_csv(data_dir + filenames[1], index_col=0, parse_dates=True))
    return (cdfs / 100., performance / 100.)


def get_FICO_data(data_dir=DATA_DIR, do_convert_percentiles=True):
    """Get FICO data in desired format"""
    data_pair = parse_data(data_dir)
    totals = read_totals(data_dir)

    if do_convert_percentiles:
        for v in data_pair:
            v.index = convert_percentiles(v.index)
    cdfs = data_pair[0]
    performance = data_pair[1]
    return cdfs, performance, totals
