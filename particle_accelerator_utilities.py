"""
Computes frequently used conversions used in particle accelerator 
physics.

Routines in this module:
    
gammabeta2gamma(GBx,GBy,GBz)
gamma2KE(gamma)
KE2gamma(KE)
gamma2beta(gamma)
beta2gamma(beta)
beta2KE(beta)

"""

import numpy as np

def gammabeta2gamma(GBx,GBy,GBz):
    """Computes the Lorentz Parameter from the normalized momentum

    sdfjdf
        
    Parameters
    ----------
    GBx : array_like
        x normalized momentum
    GBy : array_like
        y normalized momentum
    GBz : array_like
        z normalized momentum    

    Returns
    -------
    array_like
        Lorentz Parameter (Gamma)
    """
    
    return np.sqrt(1+GBx**2+GBy**2+GBz**2)

def gamma2KE(gamma):
    """Computes the kinetic energy from the Lorentz Parameter 
        
    Parameters
    ----------
    gamma : array_like
        Lorentz Parameter

    Returns
    -------
    array_like
        Kinetic Energy
    """
    
    mc2 = 0.5109989461e6;
    return (gamma-1)*mc2;

def KE2gamma(KE):
    """Computes the Lorentz Parameter from the kinetic energy
        
    Parameters
    ----------
    KE : array_like
        Kinetic Energy

    Returns
    -------
    array_like
        Lorentz Parameter
    """
    
    mc2 = 0.5109989461e6
    return (1+KE/mc2)
    
def gamma2beta(gamma):
    """Computes the normalized velocity from the kinetic energy
        
    Parameters
    ----------
    gamma : array_like
        Lorentz Parameter

    Returns
    -------
    array_like
        normalized velocity
    """
    
    return np.sqrt(1-1/gamma**2)

def beta2gamma(beta):
    """Computes the Loretnz factor from the normalized velocity
        
    Parameters
    ----------
    beta : array_like
        normalized velocity

    Returns
    -------
    array_like
        Loretnz factor
    """
    
    return 1/np.sqrt(1-beta**2)

def beta2KE(beta):
    """Computes the kinetic energy from the normalized velocity 
        
    Parameters
    ----------
    beta : array_like
        normalized velocity

    Returns
    -------
    array_like
        Kinetic Energy
    """
    
    return gamma2KE(beta2gamma(beta))
    
if __name__ == '__main__':
    pass