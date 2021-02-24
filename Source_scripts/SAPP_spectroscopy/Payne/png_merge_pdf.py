#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:41:34 2020

@author: gent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:11:59 2020

@author: gent
"""

import os
import img2pdf
import numpy as np

from PIL import Image

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
#    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
#    alpha = im.convert('RGBA').split()[-1]
    alpha = im.convert('RGBA').getchannel('A')
    # Create a new background image of our matt color.
    # Must be RGBA because paste requires both images have the same format
    # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
    bg = Image.new("RGBA", im.size, bg_colour + (255,))
    bg.paste(im, mask=alpha)
    return bg

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    
    """ source of this function: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil """ 
    
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel --> why?
    return background

def merger(output_path,input_paths):
    
    with open(output_path, "wb") as f:
        
        f.write(img2pdf.convert([i for i in input_paths]))
        
        
def remove_alpha_channel_list(input_paths):
    
    for i in input_paths:   
            
        print(i)
            
        im_old = Image.open(i)
        
        im_new = pure_pil_alpha_to_color_v2(im_old,color=(255, 255, 255)) # removes transparancy (apparently its an issue)
        
        im_new_path = i # overwriting the image

        im_new.save(im_new_path) # saving 


if __name__ == '__main__':
        
    star_name_use_file = "UVES"
    
#    path_list = np.loadtxt("linemask_comparison_figures_list_vmac_range_pngs.txt",dtype=str) # you need to update this via ls -Ftr direc/ > subzones_list.txt       

#    path_list = np.loadtxt("linemask_comparison_figures_list_vmac_best_pngs.txt",dtype=str) # you need to update this via ls -Ftr direc/ > subzones_list.txt       

    path_list = np.loadtxt("UVES_PLATO_bmk_hr10_comparison_list.txt",dtype=str)

    paths = []
    
#    print(path_list)
  
    for path_index in range(len(path_list)):
        
        paths.append("PLATO/model_obs_comparisons/UVES/"+path_list[path_index])
        
    remove_alpha_channel_list(paths)
    
    merger(f'PLATO/{star_name_use_file}_PLATO_bmk_hr10_comparison_cheb_0.pdf', paths)
    
        
    
