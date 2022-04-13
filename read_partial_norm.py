"""
This module is for working with the files associated with the partially normalized inputs and outputs of IMPACTZ/T
    
    Particle distribution in partially normalized units:
    x [m] ,GBx [px/gamma/beta], y [m], GBy [py/gamma/beta], 
    z [m], GBz [pz/gamma/beta]
        
Routines in this module: 

read_fort(filename)
read_fort_norm_momentum(filename,skiprows=1)
norm_momentum_to_IMPACTZ(df, rf_freq,charge=-0.160000409601E-18,qoverm = -0.195692801440E-05)
convert_phase_space(df)
convert_cylind(df)
convert_partially_norm(df,kinetic_energy)
get_stripe_data(file_name)
read_norm_stripes(filename,stripe_df)
read_IMPACTZ_stripes(filename,stripe_df,kinetic_energy,rf_freq)
remove_phase_space_corr(df,dim,inds)

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

import pyPartAnalysis.particle_accelerator_utilities as pau
import pyPartAnalysis.IMPACTZ_analysis as imp

def read_fort(filename):    
    """Reads text file of IMPACTT standard fort outputs
    
    Reads in the fort files with the following extensions (and information):
        fort.18
            reference particle information
        fort.24
            X RMS size info
        fort.25
            Y RMS size info
        fort.26
            Z RMS size info
        fort.27
            Maximum amplitude info
        fort.28
            Load balance and loss diagnostic
        fort.29
            Cubic roots of 3rd moments of beam dist
        fort.30
            Square roots of wth moments of beam dist
            
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.**
    
    Returns
    -------
    DataFrame
        Columns depend on the fort file extension  
    """
    
    col_names = {'.18':['t','dist','gamma','KE','beta','Rmax','energy_deviation'],
                 '.24':['t','z','avgX','rmsX','avgPx','rmsPx','alphaX','rmsEmitN'],
                 '.25':['t','z','avgY','rmsY','avgPy','rmsPy','alphaY','rmsEmitN'],
                 '.26':['t','z','rmsZ','avgPz','rmsPz','alphaZ','rmsEmitN'],
                 '.27':['t','z','maxX','maxPx','maxY','maxPy','maxZ','maxPz'],
                 '.28':['t','z','minPE','maxPE','totalPart'],
                 '.29':['t','z','X','Px','Y','Py','Z','Pz'],
                 '.30':['t','z','X','Px','Y','Py','Z','Pz']}
    
    _, file_extension = os.path.splitext(filename)
    
    if file_extension in col_names.keys():
        df = pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names[file_extension])
    else:
        df = pd.DataFrame()
        
    return df

def read_fort_norm_momentum(filename,skiprows=1):
    """Reads text file of partially normalized IMPACTZ/T particle distributions 
    
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    
    Returns
    -------
    DataFrame
        Particle distribution in partially normalized units:
        x [m] ,GBx [px/gamma/beta], y [m], GBy [py/gamma/beta], 
        z [m], GBz [pz/gamma/beta]
    """
    df = pd.read_csv(filename,
                     header=None, 
                     delimiter=r"\s+",
                     names=['x','GBx','y','GBy','z','GBz'],
                     skiprows=skiprows)

    return df

def norm_momentum_to_IMPACTZ(df, rf_freq,charge=-0.160000409601E-18,qoverm = -0.195692801440E-05):
    ps_df = df.copy()
    
    col_names = ['x','px','y','py','phase','delta','q_over_m','charge','id']
    
    c = 299792458;
    omega = 2*np.pi*rf_freq;
    gamma = pau.gammabeta2gamma(df.GBx,df.GBy,df.GBz)
    gamma0 = np.mean(gamma)
    beta0 = pau.gamma2beta(gamma0)
    betaZ = df.GBz/gamma
    
    transNorm = c/omega;
    transAngle = gamma0*beta0;
    
    ps_df['id'] = np.arange(1,df.shape[0]+1)
    ps_df['q_over_m'] = qoverm
    ps_df['charge'] = charge
    
    ps_df['x'] = ps_df['x']/transNorm
    ps_df['y'] = ps_df['y']/transNorm
    
    ps_df['px'] = ps_df['GBx']
    ps_df['py'] = ps_df['GBy']
    ps_df['delta'] = gamma0 - gamma
    
    ps_df['phase'] = omega/betaZ/c*ps_df['z']
    
    ps_df = ps_df[col_names]
    
    return ps_df


def convert_phase_space(df):
    """Converts from partially normalized IMPACTZ/T to phase space distributions
    
    This takes the partially normalized coordinates that are output by IMPACTZ/T and 
    converts them to physical phase space coordinates. See read_fort_norm_momentum 
    for partially normalized coordinates. Note that arctan was added to the angle
    calculation so that the transverse angles are valid at low energies.
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution in partially normalized coordinates
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    df_copy = df.copy()
    
    df_copy['xp'] = np.arctan(df_copy['GBx']/df_copy['GBz'])
    df_copy['yp'] = np.arctan(df_copy['GBy']/df_copy['GBz'])
    betagamma2 = df_copy.GBx**2+df_copy.GBy**2+df_copy.GBz**2;
    gamma = np.sqrt(1+betagamma2);
    gamma_mean = np.mean(gamma)
    beta_mean = pau.gamma2beta(gamma_mean)
    df_copy['delta'] = (gamma - gamma_mean)/gamma_mean
    df_copy = df_copy.drop(columns=['GBx', 'GBy', 'GBz'])
    
    df_copy = df_copy[['x','xp','y','yp','z','delta']] 
    
    return df_copy

def convert_cylind(df):
    """Converts from partially normalized IMPACTZ/T to cyclindrical phase space
    
    This takes the partially normalized coordinates that are output by IMPACTZ/T and 
    converts them to ccylindrical phase space coordinates, i.e. (r,pr), (theta,ptheta), 
    and (z,pz) where p is the physical momentum, e.g. px = GBx*m_e*c.
    
    Note that the angle iis measured wrt the +x axis, with cw being negative and 
    ccw being positive.
    
    See read_fort_norm_momentum for partially normalized coordinates.
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution in partially normalized coordinates
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    # mc = 2.73092453*10**-22
    mc = 1
    
    df_copy = df.copy()
    
    df_copy['px'] = df_copy['GBx']*mc
    df_copy['py'] = df_copy['GBy']*mc
    df_copy['pz'] = df_copy['GBz']*mc
    # betagamma2 = df_copy.GBx**2+df_copy.GBy**2+df_copy.GBz**2;
    # gamma = np.sqrt(1+betagamma2);
    # gamma_mean = np.mean(gamma)
    # beta_mean = pau.gamma2beta(gamma_mean)
    # df_copy['delta'] = (gamma - gamma_mean)/gamma_mean
    # df_copy = df_copy.drop(columns=['GBx', 'GBy', 'GBz'])
    
    df_copy['r'] = np.sqrt(df_copy['x']**2 + df_copy['y']**2)
    df_copy['pr'] = np.sqrt(df_copy['px']**2 + df_copy['py']**2)
    # df_copy['theta'] = np.arctan(df_copy['y']/df_copy['x'])
    df_copy['theta'] = np.arctan2(df_copy['y'],df_copy['x'])
    df_copy['ptheta'] = df_copy['px']*np.cos(df_copy['theta']) + df_copy['py']*np.sin(df_copy['theta'])
    
    df_copy = df_copy[['r','pr','theta','ptheta','z','pz']] 
    
    return df_copy

def convert_partially_norm(df,kinetic_energy):
    """Converts from phase space distributions to partially normalized IMPACTZ/T
    
    This takes the physical phase space coordinates and converts them to 
    partially normalized coordinates that are output by IMPACTZ/T. See 
    read_fort_norm_momentum for partially normalized coordinates.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P        
    
    Returns
    -------
    DataFrame
        Particle Distribution in partially normalized coordinates

    """
    
    df_copy = df.copy()
    
    gamma_mean = pau.KE2gamma(kinetic_energy)
    gamma = df_copy['delta']*gamma_mean + gamma_mean
    beta = pau.gamma2beta(gamma)
    
    Bx = beta*np.sin(df['xp'])
    By = beta*np.sin(df['yp'])
    Bz = np.sqrt(beta**2 - Bx**2 - By**2)
    
    df_copy['GBx'] = gamma*Bx
    df_copy['GBy'] = gamma*By
    df_copy['GBz'] = gamma*Bz
    
    df_copy = df_copy.drop(columns=['xp', 'yp', 'delta'])
    df_copy = df_copy[['x','GBx','y','GBy','z','GBz']]
    
    return df_copy

def get_stripe_data(file_name):
    """Reads text files of stripe id
    
    The expected format of the text file is a single column corresponding to the stripe id. 
    It is assumed that the row of the file correpsonds to the particle ID, starting from 1.
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the stripe information
    
    Returns
    -------
    DataFrame
        StripeNum
    """
    
    stripe_df = pd.read_csv(file_name,header=None, delimiter=r"\s+",names=['stripe_id'],dtype='Int64')
    stripe_df.index = np.arange(1, stripe_df.shape[0]+1)
    return stripe_df

def read_norm_stripes(filename,stripe_df):
    """Reads partially normalized IMPACTZ/T into a physical units with multi-indexing
    
    Particles are indexed first by the particle id ("id"), then by the 
    stripe id ("stripe_id")
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    stripe_df : DataFrame
        StripeNum
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    """
    
    df = (read_fort_norm_momentum(filename)
            .sort_index()
            .pipe(convert_phase_space))
    
    df.index = np.arange(1, df.shape[0]+1)
    df.index.name = 'id'

    df['stripe_id'] = stripe_df.loc[stripe_df.index.isin(df.index.get_level_values('id')),:]
    df = df.set_index('stripe_id',append=True)

    return df

def read_IMPACTZ_stripes(filename,stripe_df,kinetic_energy,rf_freq):
    """Reads IMPACTZ into a physical units with multi-indexing
    
    Particles are indexed first by the particle id ("id"), then by the 
    stripe id ("stripe_id")
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    stripe_df : DataFrame
        StripeNum
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    """
    
    df = (imp.read_IMPACTZ_dist(filename)
            .sort_index()
            .pipe(imp.IMPACTZ_to_phase_space,kinetic_energy,rf_freq))

    df['stripe_id'] = stripe_df.loc[stripe_df.index.isin(df.index.get_level_values('id')),:]
    df = df.set_index('stripe_id',append=True)

    return df

def remove_phase_space_corr(df,dim,inds):
    """Removes correlation using using the specified particles
    
    Removes a linear correlation in phase space for all particles using a 
    subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in partially normalized units:
        x(m),GBX,y(m),GBy,z,GBz
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    inds : array_like
        Indices of the particles that are used to calculate the correlation

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed in partially normalized units:
        x(m),GBX,y(m),GBy,z,GBz
    """
    
    dim_dict = {'x':['x','GBx'],
                'y':['y','GBy'],
                'z':['z','GBz']}
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df[dim_dict[dim][0]] - fit_df[dim_dict[dim][0]].mean()
    y = y.values.reshape(-1,1)
    x = fit_df[dim_dict[dim][1]] - fit_df[dim_dict[dim][1]].mean()
    x = x.values.reshape(-1,1)
    model = LinearRegression()
    model.fit(x, y)
    
    x_total = df_new[dim_dict[dim][1]]
    x_total = x_total.values.reshape(-1,1)
    y_predict = model.predict(x_total)
    y_original = df_new[dim_dict[dim][0]]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dim_dict[dim][0]] = y_uncorr
    
    return df_new
    
if __name__ == '__main__':
    pass