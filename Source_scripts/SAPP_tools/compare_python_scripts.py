#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:38:43 2021

@author: gent
"""

import difflib
import numpy as np

SAPP_payne_path = "../SAPP_spectroscopy/Payne/SAPP_best_spec_payne_v1p1.py"
Mikhail_payne_path = "../../../Payne/test2obs6runclean_mikhail.py"

text1 = open(SAPP_payne_path).readlines()
text2 = open(Mikhail_payne_path).readlines()

differences_in_script = []

for line in difflib.unified_diff(text1, text2):
    
    differences_in_script.append(line)
    
# differences_in_script = np.array(differences_in_script,dtype=str)

np.savetxt("differences_script_test.txt",differences_in_script,fmt='%s')