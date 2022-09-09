import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

def harmonise_input_type (x1, x2) :
    '''
    Check if a given input is a tuple or not.
    
    Parameters
    ----------
    x1 : tuple or ndarray
      tuple with mean and standard deviation, 
      or p.d.f (as an array) of first parameter.

    x2 : tuple or ndarray
      tuple with mean and standard deviation, 
      or p.d.f (as an array) of second parameter.
      
    Returns
    -------
      new ``(x1, x2)`` pair with type flag (0 if tuple,
      1 if ndarray). 
    '''
    flag = 0
    if type (x1)==tuple : 
        if type(x2)!=tuple :
            x2 = (np.mean (x2), np.std (x2), np.std (x2))
    elif type (x2)==tuple : 
        x1 = (np.mean (x1), np.std (x1), np.std (x1))
    else :
        flag = 1
    return x1, x2, flag

def std_comparison (m1, sigma_plus_1, sigma_minus_1,
                    m2, sigma_plus_2, sigma_minus_2) :
    '''
    Perform sigma comparison by assessing if mean
    value of each distribution is less than one 
    sigma away from the other.
    '''
    if m1 > m2 :
        return (np.abs (m1-m2)<sigma_minus_1 and np.abs (m1-m2)<sigma_plus_2)
    else :
        return (np.abs (m1-m2)<sigma_plus_1 and np.abs (m1-m2)<sigma_minus_2)

def consistency_check_summary_plot (x1, x2, check, 
                                    label_1='seis',
                                    label_2='FliPer') :
    '''
    Create consistency check summary plot.
    '''
    fig, ax = plt.subplots (1, 1)
    ax.set_xlabel (r'$\log g$ (dex)')

    if type (x1)!=tuple :
        if x1 is not None :
            ax.hist (x1, color='blue', alpha=1, density=True, label=label_1)
        if x2 is not None :
            ax.hist (x2, color='orange', alpha=0.5, density=True, label=label_2)
        ax.set_ylabel ('p.d.f')
        
    else :
        if x1 is not None :
            print (x1[1], x1[2])
            xerr = np.array ([x1[1], x1[2]]).reshape ((2,1))
            ax.errorbar (x1[0], 0, xerr=xerr, capsize=8,
                         fmt='x', color='blue', label=label_1)
        if x2 is not None :
            xerr = np.array ([x2[1], x2[2]]).reshape ((2,1))
            ax.errorbar (x2[0], 0, xerr=xerr, capsize=8,
                        fmt='x', color='orange', label=label_2)
        ax.get_yaxis().set_visible(False)
        
    if check :
        ax.set_title ('Consistency test successful.')
    else :
        ax.set_title ('Consistency test unsuccessful.')
        
    ax.legend ()
            
def consistency_check (x1, x2, test='std_comparison',
                       show=False) :
    '''
    Perform the consistency check (currently a Tukey test
    between two distributions, but other consistency
    checks could be used) and returns the corresponding 
    rejection criterion.
    
    Parameters
    ----------
    x1 : tuple or ndarray
      tuple with mean and standard deviation, 
      or p.d.f (as an array) of first parameter.

    x2 : tuple or ndarray
      tuple with mean and standard deviation, 
      or p.d.f (as an array) of second parameter.
      
    test : str
      type of statistical test to perform in order
      to assess the compatibility of the two 
      distributions.
      
    Returns
    -------
      boolean assessing if the consistency check 
      was successful or not. 

    '''
    x1, x2, flag = harmonise_input_type (x1, x2)
            
    if flag==0 :
        if test=='tukey':
            warnings.warn ('tukey test cannot be performed when one of the inputs is not p.d.f, switching to "std_comparison".', 
                           stacklevel=3)
            test = 'std_comparison'
        if test=='std_comparison' :
            check = std_comparison (*x1, *x2)
        else :
            raise Exception ('Unknwown test option.')
        
    elif flag==1 :
        if test=='tukey' :
            a1 = np.repeat ('1', x1.size)
            a2 = np.repeat ('2', x2.size)
            tukey = pairwise_tukeyhsd(endog=np.concatenate ((x1, x2)),
                              groups=np.concatenate ((a1, a2)),
                              alpha=0.05)
            if show :
                print (tukey)
            check = not (tukey.reject)
        elif test=='std_comparison' :
            check = std_comparison (np.mean (x1), np.std (x1), np.std (x1),
                                   np.mean (x2), np.std (x2), np.std (x2))
        else :
            raise Exception ('Unknwown test option.')
            
    if show :
        consistency_check_summary_plot (x1, x2, check)

    return check

def select_logg (logg_seis=None, logg_fliper=None,
                 show=False, test='std_comparison') :
    '''
    Select logg distribution considering the PDF of 
    Fliper and seismic logg.
    
    Parameters
    ----------
    logg_seis : tuple or ndarray
      Tuple with mean and standard deviation or 
      p.d.f as an array for seismic logg. 
      Optional, default ``None``.
    
    logg_fliper : tuple or ndarray
      Tuple with mean and standard deviation or 
      p.d.f as an array for FliPer logg. 
      Optional, default ``None``.
      
    test : str
      type of statistical test to perform in order
      to assess the compatibility of the two 
      distributions.
        
    Returns
    -------
    Depending on input type, tuple or p.d.f corresponding to the choosen logg, 
    and flag. In case no input logg is given, the returned logg is ``None``. 
    
    Notes
    -----
    Meaning of the returned flag:
        0: ``logg_seis`` is chosen and is consistent with ``logg_fliper``.
        1: ``logg_seis`` is chosen and is not consistent with ``logg_fliper``.
        2: ``logg_seis`` is chosen and ``logg_fliper`` non available.
        3: ``logg_fliper`` is chosen and ``logg_seis`` non available.
        4: no logg available.
    '''
    if logg_fliper is None :
        return logg_seis, 2
    if logg_seis is None :
        return logg_fliper, 3
    if logg_seis is None and logg_fliper is None :
        return None, 4
    
    check = consistency_check (logg_seis, logg_fliper, test=test,
                                show=show)
    
    if check :
        return logg_seis, 0
    else :
        return logg_seis, 1


def test_select_logg (param1=(4.0, 0.1, 0.1), 
                      param2=(4.0, 0.1, 0.1),
                      generate_pdf=False, n1=100,
                      n2=100, test='std_comparison') :
    '''
    Test ``select_logg`` function.
    
    Parameters
    ----------
    param1 : tuple
      Mean, std, and size of first distribution.
        
    param2 : tuple
      Mean, std and size of second distribution.
        
    xlim : tuple
      xaxis boundary of the illustrative plot.
      
    generate_pdf : bool 
      if set to ``True``, a Gaussian distribution
      will be randomly generated from the given
      parameters. In this case, only the first
      sigma value of param1 and param2 is considered.
      
    n1 : int
      number of elements in the first distribution.
      
    n2 : 
      number of elements in the second distribution.
    '''
    
    if generate_pdf :
        if param1 is not None : 
            logg1 = np.random.normal(param1[0], param1[1], n1)
        else :
            logg1 = None
        if param2 is not None :
            logg2 = np.random.normal(param2[0], param2[1], n2)
        else :
            logg2 = None
    else :
        logg1, logg2 = param1, param2
        
    logg, flag = select_logg (logg_seis=logg1, logg_fliper=logg2,
                              show=True, test=test)

    return logg, flag
