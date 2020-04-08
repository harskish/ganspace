# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import string
import numpy as np
from pathlib import Path
import requests
import pickle
import sys
import re

def prettify_name(name):
    valid = "-_%s%s" % (string.ascii_letters, string.digits)
    return ''.join(map(lambda c : c if c in valid else '_', name))

# Add padding to sequence of images
# Used in conjunction with np.hstack/np.vstack
# By default: adds one 64th of the width of horizontal padding
def pad_frames(strip, pad_fract_horiz=64, pad_fract_vert=0, pad_value=None):
    dtype = strip[0].dtype
    if pad_value is None:
        if dtype in [np.float32, np.float64]:
            pad_value = 1.0
        else:
            pad_value = np.iinfo(dtype).max
    
    frames = [strip[0]]
    for frame in strip[1:]:
        if pad_fract_horiz > 0:
            frames.append(pad_value*np.ones((frame.shape[0], frame.shape[1]//pad_fract_horiz, 3), dtype=dtype))
        elif pad_fract_vert > 0:
            frames.append(pad_value*np.ones((frame.shape[0]//pad_fract_vert, frame.shape[1], 3), dtype=dtype))
        frames.append(frame)
    return frames


def download_google_drive(url, output_name):
    print('Downloading', url)
    session = requests.Session()
    r = session.get(url, allow_redirects=True)
    r.raise_for_status()

    # Google Drive virus check message
    if r.encoding is not None:
        tokens = re.search('(confirm=.+)&amp;id', str(r.content))
        assert tokens is not None, 'Could not extract token from response'

        url = url.replace('id=', f'{tokens[1]}&id=')
        r = session.get(url, allow_redirects=True)
        r.raise_for_status()

    assert r.encoding is None, f'Failed to download weight file from {url}'

    with open(output_name, 'wb') as f:
        f.write(r.content)

def download_generic(url, output_name):
    print('Downloading', url)
    session = requests.Session()
    r = session.get(url, allow_redirects=True)
    r.raise_for_status()

    # No encoding means raw data
    if r.encoding is None:
        with open(output_name, 'wb') as f:
            f.write(r.content)
    else:
        download_manual(url, output_name)

def download_manual(url, output_name):
    outpath = Path(output_name).resolve()
    while not outpath.is_file():
        print('Could not find checkpoint')
        print(f'Please download the checkpoint from\n{url}\nand save it as\n{outpath}')
        input('Press any key to continue...')

def download_ckpt(url, output_name):
    if 'drive.google' in url:
        download_google_drive(url, output_name)
    elif 'mega.nz' in url:
        download_manual(url, output_name)
    else:
        download_generic(url, output_name)