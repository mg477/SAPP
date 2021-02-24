#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:04:13 2020

@author: gent
"""

from subprocess import call
import os


pdf_list = os.listdir("pdfs")

for pdf_index in range(len(pdf_list)):

    pdf_name = pdf_list[pdf_index].replace(".pdf","")
    
    print("Converting pdf to eps")

    call(["pdf2ps", f"pdfs/{pdf_list[pdf_index]}", f"eps/{pdf_name}.eps"])
    
    print(f"Converted {pdf_name}")
    print("======================================")
