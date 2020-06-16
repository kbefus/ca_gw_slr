#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:23:19 2018

@author: kbefus
"""

import os
#%%
research_dir = r'/mnt/762D83B545968C9F'
model_type = 'mhhw_noghb'
wdir = os.path.join(research_dir,'model_{}'.format(model_type))
out_fname = os.path.join(wdir,'run_script.sh')
script_path = os.path.join(wdir,'scripts','cgw_driver_wcoast_vark_{}_24Oct19.py'.format(model_type))
scripttype = '#!/bin/bash'

nruns = 5
sleep_cmd = 'sleep {}s\n'
sleep_sec = 10.

script_cmd = 'python {} &\n'

with open(out_fname,'w') as fout:
    fout.write(scripttype)
    fout.write('\n')
#    fout.write('conda activate gw3 \n') # activate conda environment
#    fout.write('. ~/.profile \n') # make sure path is set to include user/bin
    for i in range(nruns):
        fout.write(script_cmd.format(script_path))
        fout.write(sleep_cmd.format(sleep_sec))

# Run in unix terminal with "bash -x run_script.sh"