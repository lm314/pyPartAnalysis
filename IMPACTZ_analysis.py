"""
Allows for analysis of distributions created by IMPACTZ/T. 

Routines in this module:
    
read_IMPACTZ_dist(filename)
read_stripe_data(filename)
read_fort(filename)
read_slice_info(filename)
filter_stripe(df,stripeNum)
IMPACTZ_to_phase_space(df, kinetic_energy, rf_freq)
filter_stripe(df,stripeNum)
det_plot_scale(ps_df)
plot_phase_space_density(df,dim,num_bins,ax=None,**plt_kwargs)
plot_transverse_density(df,num_bins,ax=None,**plt_kwargs)
bunching_factor(x, wavelengths)
transverse_bin(df,num_bin_x,num_bin_y)
bunching_factor_area(df,wavelengths,num_pixels=[32,32])
plot_bunching_factor_area(df,b0,ax=None,fig=None,**plt_kwargs)
plot_bunching_factor_vs_wavelength(wavelengths,b0,ax=None,logscale = False,ymin=0,**plt_kwargs)
get_transport_matrix_SVD(input_df,output_df)
get_transport_matrix(input_df,output_df)
get_twiss_parameters(df)
remove_phase_space_corr(df,dim,inds)
add_phase_space_corr(df,dim,slope)
add_transverse_radial(df)
gen_stripe_id(df,stripe_df)
read_IMPACTZ_stripes(filename,stripe_df,kinetic_energy,rf_freq)
make_mean_zero(df)
get_iterable(x)
get_poly_area(df,dim,deg,num_pixels=[32,32],cut_off=1)
get_poly_fit(df,dim,deg,cut_off)
filter_by_counts(df,column,bins,cutoff)
get_twiss_z_slices(df,bins)
make_sigma_mat(alpha,beta)
calculate_twiss_mismatch(sigma,sigma0)
print_twiss(df,kinetic_energy)
twiss_ellipse(alpha,beta,emit,xy=[0,0],scalex=0,scaley=0,**ell_kwargs)
twiss_ellipse_parametric(alpha,beta,emit,num_points=100,xy=[0,0])
get_angle_twiss(alpha,beta)
twiss_ellipse_parametric(alpha,beta,emit,num_points=100,xy=[0,0])
get_angle_twiss(alpha,beta)
normalized_coord(df)
get_phase_advance(df_norm,dims = ['x','y','z'])

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numba import jit
import collections
import os

import pyPartAnalysis.particle_accelerator_utilities as pau

def read_IMPACTZ_dist(filename):
    """Reads text file of IMPACTZ particle distributions 
    
    This takes the normalized coordinates that are output by IMPACTZ and 
    converts them to physical phase space coordinates. See read_IMPACTZ_dist 
    for normalized coordinates
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    
    Returns
    -------
    DataFrame
        Particle distribution in normalized units:
        x*c/omega,xp*beta*gamma,y*c/omega,yp*beta*gamma,phase(rad),
        -deltaGamma i.e gamma0-gamma, charge/mass, charge, particle id    
    """
    
    col_names = ['x','px','y','py','phase','delta','q_over_m','charge','id']
    
    df = (pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names)
              .astype({'id': int})
              .set_index('id'))
    return df

def read_stripe_data(filename):
    """Reads text files of stripe id and particle id
    
    The expected format of the text file is two columns, with the first giving 
    the stripe number and the second giving the particle id of the the last 
    particle in that stripe
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the stripe information
    
    Returns
    -------
    DataFrame
        StripeNum and endIndex
    """
        
    return pd.read_csv(filename, delimiter=",")

def read_slice_info(filename): 
    """Reads text file of IMPACTZ slice information
    
    Reads the slice information for the beam at the specified location in the 
    IMPACTZ lattice. 
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.20*
    
    Returns
    -------
    DataFrame
        The returned columns are:
                bunch length (m)
                # of particles per slice
                current per slice (A)
                X normalized Emittance per slice (m-rad)
                Y normalized Emittance per slice (m-rad)
                dE/E 
                uncorrelated energy spread per slice (eV)
                x_rms fo each slice (m)
                y_rms fo each slice (m)
                X mismatch factor
                Y mismatch factor 
    """
    col_names = ['bunch_length','num_part','current','x_emit_n','y_emit_n',
                 'dE_E','uncorr_E_spread','x_rms','y_rms','x_mismatch',
                 'y_mismatch']
    
    df = pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names)

    return df
   
def read_fort(filename):    
    """Reads text file of IMPACTZ standard fort outputs
    
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
        fort.31
            Number of particles for each charge state
            
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
    
    col_names = {'.18':['dist','absPhase','gamma','KE','beta','Rmax'],
                 '.24':['z','avgX','rmsX','avgPx','rmsPx','alphaX','rmsEmitN'],
                 '.25':['z','avgY','rmsY','avgPy','rmsPy','alphaY','rmsEmitN'],
                 '.26':['z','avgPhase','rmsPhase','avgPz','rmsPz','alphaZ','rmsEmitN'],
                 '.27':['z','maxX','maxPx','maxY','maxPy','maxPhase','maxDelta'],
                 '.28':['z','minPE','maxPE','totalPart'],
                 '.29':['z','X','Px','Y','Py','phase','delta'],
                 '.30':['z','X','Px','Y','Py','phase','delta'],
                 '.31':['z','numPart']}
    
    _, file_extension = os.path.splitext(filename)
    
    if file_extension in col_names.keys():
        df = pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names[file_extension])
    else:
        df = pd.DataFrame()
        
    return df

def IMPACTZ_to_phase_space(df, kinetic_energy, rf_freq):
    """Converts from IMPACTZ to phase space distributions
    
    This takes the normalized coordinates that are output by IMPACTZ and 
    converts them to physical phase space coordinates. See read_IMPACTZ_dist 
    for normalized coordinates
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution in normalized coordinates
    kinetic_energy : float
        The mean kinetic energy of the distribution in IMPACTZ
    rf_freq : float
        The RF frequency used in IMPACTZ for normalization
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    c = 299792458;
    omega = 2*np.pi*rf_freq;
    gamma0 = pau.KE2gamma(kinetic_energy)
    beta0 = pau.gamma2beta(gamma0)
    
    transNorm = c/omega;
    transAngle = gamma0*beta0;
    
    ps_df = df.drop(['q_over_m','charge'], axis=1)
    ps_df['x'] = transNorm*ps_df['x']
    ps_df['y'] = transNorm*ps_df['y']
    ps_df['px'] = ps_df['px']/transAngle
    ps_df['py'] = ps_df['py']/transAngle
    gamma = gamma0-ps_df['delta'];
    beta = np.sqrt(1-1/gamma**2);
    betaX = beta*np.sin(ps_df['px'])
    betaY = beta*np.sin(ps_df['py'])
    betaZ = np.sqrt(beta**2-betaX**2-betaY**2)
    
    ps_df['phase'] = -betaZ*c*ps_df['phase']/2/np.pi/rf_freq
    ps_df['delta'] = -ps_df['delta']/gamma0/beta0**2
    
    ps_df.rename(columns = {'phase':'z','px':'xp','py':'yp'}, inplace = True)
    
    return ps_df

def filter_stripe(df,stripeNum):
    """Filters out particles not in specified stripes
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution in normalized coordinates with two-level 
        indexing, the first being the particle id and the second being 
        the stripe id
    stripeNum : list
        The stripe numbers of the particles to be returned
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units for the specified stripes
    """
    
    dist_df = df.copy()
    return dist_df.loc[(slice(None),stripeNum),:]
       
def det_plot_scale(ps_df,cutoff = 0.75):
    """Gives the scalings and labels for a physical particle distribution
    
    Takes the 6 dimensional phase space and returns the associated scaling 
    and appropriate label for plotting for that dimension.
    
    Parameters
    ----------
    ps_df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    cutoff : float
        Indicates the cutoff value for scale, e.g. 0.75 means 
        that a value of 750 will return a scale of 3 instead 
        of 0. Must be greater than 0 and equal to or less 
        than 1.
    
    Returns
    -------
    dict of {str : dict of {str : int} and {str : str}}
        A multilvel dictionary with the scaling and label with units for each 
        dimension. Can be accessed using the the coordinate name at the first 
        level ['x','xp','y','yp','z','delta']. The second level can be 
        accessed using 'scaling' or 'label'
    """
    
    scaleSteps = 3
    maxExtents = abs(pd.concat([ps_df.max(axis=0), ps_df.min(axis=0)],axis=1))
    maxExtent = maxExtents.max(axis=1)
    maxExtent[maxExtent==0] = 1
    scale = np.floor(np.log10(maxExtent)*cutoff)
    scale = scaleSteps*np.floor(scale/scaleSteps);
    
    space_labels = {3:'(km)',
                    0:'(m)',
                    -3:'(mm)',
                    -6:'(um)',
                    -9:'(nm)',
                    -12:'(pm)',
                    -15:'(fm)'}
    transverse_angle_labels = {0:'(rad)',
                               -3:'(mrad)',
                               -6:'(urad)',
                               -9:'(nrad)',
                               -12:'(prad)'}
    #ps_df.columns.values
    
    scale_info = {name:
                  {'scale':scale[idx],
                   'label':space_labels[scale[idx]] if idx in [0,2,4] 
                       else transverse_angle_labels[scale[idx]] if idx in [1,3] 
                       else f"x10^{int(scale[idx]):d}" } 
                  for idx, name in enumerate(ps_df.columns.values)}
    return scale_info

def plot_phase_space_density(df,dim,num_bins,ax=None,cutoff = 0.75,**plt_kwargs):
    """Plots the specfied phase space with scaling and appropriate labels
    
    The defaults can be overwritten using the keyword arguments for 
    **plt_kwargs that a the same as those supplied to matplotlib.pyplot.hist2d.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.hist2d
        
    
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted
    """
    
    scale_info = det_plot_scale(df,cutoff)

    if ax is None:
        ax = plt.gca()

    if(dim=='x'):
        ax.set(xlabel=f"x {scale_info['x']['label']}",
               ylabel=f"theta_x {scale_info['xp']['label']}")
        ax.hist2d(df['x']*10**-scale_info['x']['scale'], 
                   df['xp']*10**-scale_info['xp']['scale'],
                   bins = num_bins,
                   **plt_kwargs)

    elif(dim=='y'):
        ax.set(xlabel=f"y {scale_info['y']['label']}",
               ylabel=f"theta_y {scale_info['yp']['label']}")
        ax.hist2d(df['y']*10**-scale_info['y']['scale'], 
                   df['yp']*10**-scale_info['yp']['scale'],
                   bins = num_bins,
                   **plt_kwargs)
        
    elif(dim=='z'):
        ax.set(xlabel=f"z {scale_info['z']['label']}",
               ylabel=f"delta {scale_info['delta']['label']}")
        ax.hist2d(df['z']*10**-scale_info['z']['scale'], 
                   df['delta']*10**-scale_info['delta']['scale'],
                   bins = num_bins,
                   **plt_kwargs)
        
    return ax

# Finish documentation for below function

def plot_transverse_density(df,num_bins,ax=None,cutoff = 0.75,**plt_kwargs):
    """Plots the transverse density plot with scaling and appropriate labels
    
    The defaults can be overwritten using the keyword arguments for 
    **plt_kwargs that a the same as those supplied to matplotlib.pyplot.hist2d.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    num_bins : int
        Number of bins are the same in both dimensions by default
    matplotlib.axes : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.hist2d
        
    
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted
    """
    
    scale_info = det_plot_scale(df,cutoff)

    if ax is None:
        ax = plt.gca()

    ax.set(xlabel=f"x {scale_info['x']['label']}",
           ylabel=f"y {scale_info['y']['label']}")
    ax.hist2d(df['x']*10**-scale_info['x']['scale'], 
              df['y']*10**-scale_info['y']['scale'],
              bins = num_bins,
              **plt_kwargs)
    
    return ax

def plot_transverse_angle_density(df,num_bins,ax=None,cutoff = 0.75,**plt_kwargs):
    """Plots the transverse angle density with scaling and appropriate labels
    
    The defaults can be overwritten using the keyword arguments for 
    **plt_kwargs that a the same as those supplied to matplotlib.pyplot.hist2d.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    num_bins : int
        Number of bins are the same in both dimensions by default
    matplotlib.axes : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.hist2d
        
    
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted
    """
    
    scale_info = det_plot_scale(df,cutoff)

    if ax is None:
        ax = plt.gca()

    ax.set(xlabel=f"Theta X {scale_info['xp']['label']}",
           ylabel=f"Theta y{scale_info['yp']['label']}")
    ax.hist2d(df['xp']*10**-scale_info['xp']['scale'], 
              df['yp']*10**-scale_info['yp']['scale'],
              bins = num_bins,
              **plt_kwargs)
    
    return ax


def bunching_factor_numpy(x, wavelengths):
    """Computes the bunching factor from position data using only Numpy
    
    Slightly slower implementation than the numba version

    Parameters
    ----------
    x : array_like, shape=(M,)
        The position data
    wavelengths : array_like, shape=(N,)
        The wavelengths for which the bunching factor is calculated

    Returns
    -------
    array_like, shape=(N,)
        The complex bunching factor
    """
    
    temp = x.reshape(np.size(x),1)
    b0 = np.sum(np.exp(2j * np.pi * temp / wavelengths),axis=0)/np.size(x)
    
    return b0



@jit(nopython=True,fastmath=True)
def bunching_factor(x, wavelengths):
    """Computes the bunching factor from position data

    Parameters
    ----------
    x : array_like, shape=(M,)
        The position data
    wavelengths : array_like, shape=(N,)
        The wavelengths for which the bunching factor is calculated

    Returns
    -------
    array_like, shape=(N,)
        The complex bunching factor
    """
    b0 = np.zeros(shape=np.shape(wavelengths),dtype=np.complex128)

    for inx, wavelength in enumerate(wavelengths):
        for x_val in x:
            b0[inx] += np.exp(2j * np.pi * x_val / wavelength )
    
    if(np.shape(x)[0] > 0):
        b0 = b0/np.shape(x)[0]
    return b0

@jit(nopython=True,fastmath=True)
def bunching_factor_scalar(x, wavelength):
    """Computes the bunching factor from position data for scalar wavelength

    Parameters
    ----------
    x : array_like, shape=(M,)
        The position data
    wavelengths : array_like, shape=(N,)
        The wavelengths for which the bunching factor is calculated

    Returns
    -------
    array_like, shape=(N,)
        The complex bunching factor
    """
    b0 = np.zeros(shape=1,dtype=np.complex128)
        
    for x_val in x:
        b0+= np.exp(2j * np.pi * x_val / wavelength )
                
    if(np.shape(x)[0] > 0):
        b0 = b0/np.shape(x)[0]
    return b0

def transverse_bin(df,num_bin_x,num_bin_y):
    """Splits particles into x and y bins
    
    Splits particles into categories based on the x and y positions. 
    The positions of the bins are equidistant and span from the min to the 
    max particle position in each dimension.

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    num_bin_x : int
        Must be a positive integer.
    num_bin_y : int
        Must be a positive integer.

    Returns
    -------
    Dataframe
        A copy of df but with bins_x and bin_y appended.
    """
    
    df_new = df.copy()
    labels_x = np.arange(0, num_bin_x)
    labels_y = np.arange(0, num_bin_y)
    
    dimension = "x"
    minx = df_new[dimension].min(axis=0)
    maxx = df_new[dimension].max(axis=0)
    bins_x = np.linspace(minx,maxx,num_bin_x+1)
    df_new["bins_x"] = pd.cut(df_new[dimension], bins=bins_x, labels=labels_x, include_lowest=True)
    df_new["bins_x"] = df_new['bins_x'].cat.codes
    
    dimension = "y"
    miny = df_new[dimension].min(axis=0)
    maxy = df_new[dimension].max(axis=0)
    bins_y = np.linspace(miny,maxy,num_bin_y+1)
    df_new["bins_y"] = pd.cut(df_new[dimension], bins=bins_y, labels=labels_y, include_lowest=True)
    df_new["bins_y"] = df_new['bins_y'].cat.codes
    
    return df_new
    
def bunching_factor_area(df,wavelengths,num_pixels=[32,32]):
    """Calculates the bunching factor at transverse positions
    
    Calculates the bunching factor at the wavelengths 

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    wavelengths : array_like, shape(n,)
        Must be a positive integer.
    num_pixels : array_like, shape (2,), default [32,32]
        Must be a positive integer.

    Returns
    -------
    ndarray, complex, shape(num_bin_x,num_bin_y,n)
        Bunching factor at the transverse positions
    """
    
    df_new = transverse_bin(df,num_pixels[0],num_pixels[1])
    
    b0_data = (df_new.groupby(['bins_x','bins_y'])["z"]
          .apply(lambda x: bunching_factor(x.to_numpy(), wavelengths)))
    
    b0 = np.zeros((num_pixels[1],num_pixels[0],len(wavelengths)),dtype=complex)

    for idx, df_select in b0_data.groupby(level=[0, 1]):
        nonzero = get_iterable(np.isnan(df_select.values[0]))
        for idz,cond in enumerate(nonzero):
            if(~cond):
                b0[idx[1],idx[0],idz] = df_select.values[0][idz]
                
    return b0

def plot_bunching_factor_area(df,b0,ax=None,fig=None,cutoff = 0.75,**plt_kwargs):
    """Plot the bunching factor
    
    Calculates the bunching factor at the wavelengths 

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    b0 : ndarray, complex, shape(num_bin_x)
        bunching factor at the transverse positions
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    fig : matplotlib.figure, optional, default None
        Figure the plot will be plotted on
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.imshow

    Returns
    -------
    fig : matplotlib.figure
        Figure the plot will be plotted on
    ax : matplotlib.axes
        Axes the plot will be plotted on
    """    
    
    scale_info = det_plot_scale(df,cutoff)

    if fig is None:
        fig = plt.figure()    
    if ax is None:
        ax = plt.subplot()
        
    min_x = df["x"].min(axis=0)
    max_x = df["x"].max(axis=0)
    min_y = df["y"].min(axis=0)
    max_y = df["y"].max(axis=0)
    dx = (max_x-min_x)*10**-scale_info['x']['scale']
    dy = (max_y-min_y)*10**-scale_info['y']['scale']
    extent = [min_x*10**-scale_info['x']['scale'],
              max_x*10**-scale_info['x']['scale'],
              min_y*10**-scale_info['y']['scale'],
              max_y*10**-scale_info['y']['scale']]
    # flipud is used as imshow inverts the numpy matrix b0.
    im = ax.imshow(np.flipud(b0),interpolation='nearest',extent=extent,aspect=dx/dy)
    ax.set_xlabel(f"x {scale_info['x']['label']}")
    ax.set_ylabel(f"y {scale_info['y']['label']}")
    ax.set(**plt_kwargs)
    fig.colorbar(im,ax=ax)
    
    return fig, ax

def plot_bunching_factor_vs_wavelength(wavelengths,b0,ax=None,logscale = False,ymin=0,**plt_kwargs):
    """Plot the bunching factor vs wavelength
    
    Plots the magnitude of the bunching factor vs the wavelength

    Parameters
    ----------
    b0 : ndarray, float, shape(num_bin_x)
        bunching factor at the transverse positions
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    fig : matplotlib.figure, optional, default None
        Figure the plot will be plotted on
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.hist2d or 
        matplotlib.pyplot.semilogy depending on whether logscale is true 
        or false

    Returns
    -------
    matplotlib.axes
        Axes the plot will be plotted on
    """  
    
    if ax is None:
        ax = plt.gca()
    
    wavelengths = wavelengths*1e9;

    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("|b0| (arb units)")
    if(logscale==False):
        ax.plot(wavelengths,b0,**plt_kwargs)
    else:
        ax.semilogy(wavelengths,b0,**plt_kwargs)
        
    if(logscale==True and ymin <= 0):
        ax.set_xlim([min(wavelengths),max(wavelengths)])
        ax.set_ylim([None,None])
    else:
        ax.set_xlim([min(wavelengths),max(wavelengths)])
        ax.set_ylim([ymin,None])
        
    return ax

def get_transport_matrix_SVD(input_df,output_df):
    """Calculates the linear transport matrix for 6d phase space using SVD
    
    Uses sklearn.linear_model.LinearRegression to fit the 6D transport matrix. 
    Is generally slower than get_transport_matrix as the sklearn package uses 
    SVD for the calculation which is more robust but slower.

    Parameters
    ----------
    input_df : DataFrame
        Initial particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    output_df : DataFrame
        Final particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P

    Returns
    -------
    reg.coef_ : float
        6x6 linear transport matrix
    reg : sklearn.linear_model.LinearRegression
        Fitted linear model
    """
    
    idx = input_df.index.intersection(output_df.index)
    idx_id = [ii[0] for ii in idx]
    reg = LinearRegression().fit(input_df.loc[idx_id],output_df.loc[idx_id])
    return [reg.coef_,reg] 

def get_transport_matrix(input_df,output_df):
    """Calculates the linear transport matrix for 6d phase space
    
    Uses numpy least squares to calculate the 6D linear transport matrix. 
    Assumes input_df and output_df are sorted in ascending order by their 
    particle id values.

    Parameters
    ----------
    input_df : DataFrame
        Initial particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    output_df : DataFrame
        Final particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P

    Returns
    -------
    x[0].T : float
        6x6 linear transport matrix
    x : numpy.linalg.lstsq
        All outputs from the function
    """
    
    idx = input_df.index.get_level_values('id').intersection(output_df.index.get_level_values('id'))
    if(len(idx)!=output_df.shape[0] or len(idx)!=input_df.shape[0]):
        x = np.linalg.lstsq(input_df.loc[input_df.index.get_level_values('id').isin(idx),:],
                            output_df.loc[output_df.index.get_level_values('id').isin(idx),:],
                            rcond=None)
    else:
        x = np.linalg.lstsq(input_df,
                            output_df,
                            rcond=None)
        
    return [x[0].T,x]

def get_twiss_parameters(df):
    """Removes correlation using using the specified particles
    
    Removes a linear correlation in phase space for all particles using a 
    subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P

    Returns
    -------
    emit: numpy.ndarray
        Geometric Emittance in the x,y,z dimensions respectively
    alpha: numpy.ndarray
        Alpha Twiss parameter in the x,y,z dimensions respectively
    beta: numpy.ndarray
        Beta Twiss parameter in the x,y,z dimensions respectively        
    """   
    cov_mat = df.cov()
    emit = np.empty((0,3))
    for ii in range(0,3):
        emit = np.append(emit,np.sqrt(np.linalg.det(cov_mat.iloc[(2*ii):(2*ii+2),(2*ii):(2*ii+2)].to_numpy())))
    beta = np.diagonal(cov_mat,offset=0)[0:5:2]/emit
    alpha = -np.diagonal(cov_mat,offset=1)[0:5:2]/emit    
    
    return emit, alpha, beta

def remove_phase_space_corr(df,dim,inds,dimremove=0):
    """Removes correlation using using the specified particles
    
    Removes a linear correlation in phase space for all particles 
    using a  subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    inds : array_like
        Indices of the particles that are used to calculate the correlation
    dimremove : {0, 1}
        Indicates which dimension of phase space the correlation will be 
        removed from. 0 for the spatial or 1 for the compliment.

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    dim_dict = {'x':['x','xp'],
                'y':['y','yp'],
                'z':['z','delta']}
    
    ind_y = dimremove
    ind_x = np.mod(dimremove+1,2)
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df[dim_dict[dim][ind_y]] - fit_df[dim_dict[dim][ind_y]].mean()
    y = y.values.reshape(-1,1)
    x = fit_df[dim_dict[dim][ind_x]] - fit_df[dim_dict[dim][ind_x]].mean()
    x = x.values.reshape(-1,1)
    
    model = LinearRegression()
    model.fit(x, y)
    
    x_total = df_new[dim_dict[dim][ind_x]]
    x_total = x_total.values.reshape(-1,1)
    
    y_predict = model.predict(x_total)
    y_original = df_new[dim_dict[dim][ind_y]]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dim_dict[dim][ind_y]] = y_uncorr
    
    return df_new

def remove_phase_space_corr_poly(df,dim,inds,dimremove=0,degree=1):
    """Removes polynomial correlation using using the specified particles
    
    Removes a polynomial correlation in phase space for all particles using a 
    subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    inds : array_like
        Indices of the particles that are used to calculate the correlation
    dimremove : {0, 1}
        Indicates which dimension of phase space the correlation will be 
        removed from. 0 for the spatial or 1 for the compliment.
    degree : int > 0
        Degree of the polynomial for which the correlation will be removed. 
        Defaults to a linear correlation.

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    dim_dict = {'x':['x','xp'],
                'y':['y','yp'],
                'z':['z','delta']}
    
    ind_y = dimremove
    ind_x = np.mod(dimremove+1,2)
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df[dim_dict[dim][ind_y]] - fit_df[dim_dict[dim][ind_y]].mean()
    y = y.values.reshape(-1,1)
    x = fit_df[dim_dict[dim][ind_x]] - fit_df[dim_dict[dim][ind_x]].mean()
    x = x.values.reshape(-1,1)
    
    poly = PolynomialFeatures(degree=degree)
    X_ = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_,y)
    
    x_total = df_new[dim_dict[dim][ind_x]]
    x_total = x_total.values.reshape(-1,1)
    x_total_ = poly.fit_transform(x_total)
    
    y_predict = model.predict(x_total_)
    y_original = df_new[dim_dict[dim][ind_y]]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dim_dict[dim][ind_y]] = y_uncorr
    
    return df_new

def remove_2D_linear_corr(df,dim,inds,dimremove):
    """Removes multidimensional linear correlation using the specified particles
    
    Removes a multidimensional linear correlation in phase space for all particles using a 
    subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : list {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dependent Variable
    inds : array_like
        Indices of the particles that are used to calculate the correlation
    dimremove : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Indicates which dimension of phase space the correlation will be 
        removed from.

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """   
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df.loc[:,dimremove] - fit_df.loc[:,dimremove].mean()
    y = y.values.reshape(-1,1)
    x = fit_df.loc[:,dim] - fit_df.loc[:,dim].mean()
    
    poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias = False)
    X_ = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_,y)
    
    x_total = df_new.loc[:,dim]

    x_total_ = poly.fit_transform(x_total)
    
    y_predict = model.predict(x_total_)
    y_original = df_new.loc[:,dimremove]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dimremove] = y_uncorr
    
    return df_new


def add_phase_space_corr(df,dim,slope):
    """Adds correlation to phase space
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    slope : float
        Units of z/delta or m/rad

    Returns
    -------
    DataFrame
        Particle distribution with correlation added in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    dim_dict = {'x':['x','xp'],
                'y':['y','yp'],
                'z':['z','delta']}
    
    fit_df = df.copy()
    y = fit_df[dim_dict[dim][0]] - fit_df[dim_dict[dim][0]].mean()
    y = y.values.reshape(-1,1)
    x = fit_df[dim_dict[dim][1]] - fit_df[dim_dict[dim][1]].mean()
    x = x.values.reshape(-1,1)
    
    y_predict = slope*x
    y_corr = y - y_predict
    fit_df.loc[:,dim_dict[dim][0]] = y_corr
    
    return fit_df

def add_transverse_radial(df):
    """Adds transverse radial phase space coordinates to dataframe
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P

    Returns
    -------
    DataFrame
        Particle distribution with radial position and angle:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P,r,rp
    """
    
    df_new = df.copy()
    df_new['r'] = np.sqrt(df_new['x']**2 + df_new['y']**2)
    df_new['rp'] = np.sqrt(df_new['xp']**2 + df_new['yp']**2)
    return df_new    

def gen_stripe_id(df,stripe_df):
    """Adds stripe_id for particles
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    stripe_df : DataFrame
        StripeNum and endIndex

    Returns
    -------
    Dataframe
        Copy of df but with stripe id appended        
    """
    
    num_particles = stripe_df['endIndex'] - stripe_df['endIndex'].shift(1,fill_value=0)
    df_new = df.copy()
    df_new['stripe_id'] = np.repeat(stripe_df['stripeNum'].values, num_particles)[df_new.index-1]
    return df_new    

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
        StripeNum and endIndex
    kinetic_energy : float
        The mean kinetic energy of the distribution in IMPACTZ
    rf_freq : float
        The RF frequency used in IMPACTZ for normalization
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    """
    
    df = (read_IMPACTZ_dist(filename)
                   .sort_index()
                   .pipe(IMPACTZ_to_phase_space,kinetic_energy,rf_freq)
                   .pipe(gen_stripe_id,stripe_df)
                   .set_index('stripe_id',append=True))    
    return df

def make_mean_zero(df):
    """Makes the mean of each column zero
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    
    Returns
    -------
    Dataframe
        df but the mean is zero 
    """
    
    # Removes mean from all columns of data frame
    df_new = df.copy()
    return df_new-df_new.mean()

def get_iterable(x):
    """Converts an object to iterable if not already
    
    Parameters
    ----------
    x : array_like
        array to be iterable
    
    Returns
    -------
    array_like
        iterable array 
    """
    
    # Makes object iterable if not already
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)
    
def get_poly_area(df,dim,deg,num_pixels=[32,32],cut_off=1):
    """Polynomial Fit to particles in x and y bins
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Phase Space dimension to be fitted.
    deg : int
        Polynomial degree of fitting
    num_pixels : array_like, shape=(2,1)
        number of bins in the x and y directions
    cutoff : int
        The number of particles below which the fitting does not occur
     
    
    Returns
    -------
    numpy.polyfit, shape(num_pixels)
        polynomial fit to phase space
    """
    
    df_new = transverse_bin(df,num_pixels[0],num_pixels[1])
    
    poly_data = (df_new.groupby(['bins_x','bins_y'])
               .apply(lambda x: get_poly_fit(x,dim,deg,cut_off)))
    
    poly = np.zeros((num_pixels[0],num_pixels[1],deg+1),dtype=float)

    for idx, df_select in poly_data.groupby(level=[0, 1]):
        nonzero = get_iterable(np.isnan(df_select.values[0]))
        for idz,cond in enumerate(nonzero):
            if(~cond):
                poly[idx[0],idx[1],idz] = df_select.values[0][idz]
        
    return poly  

def get_poly_fit(df,dim,deg,cut_off):
    """Slices distribution along column and filters slices with low counts.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Phase Space dimension to be fitted.
    deg : int
        Polynomial degree of fitting
    cutoff : int
        The number of particles below which the fitting does not occur
    
    Returns
    -------
    numpy.polyfit
        polynomial fit to phase space
    """

    if(len(df)>cut_off):
        dim_dict = {'x':['x','xp'],
                   'y':['y','yp'],
                   'z':['z','delta']}
            
        x = df.loc[:,dim_dict[dim][1]] - df.loc[:,dim_dict[dim][1]].mean()
        x = x.values.flatten()
        y = df.loc[:,dim_dict[dim][0]] - df.loc[:,dim_dict[dim][0]].mean()
        y = y.values.flatten()
        poly = np.polyfit(x=x,y=y,deg=deg)
    else:
        poly = np.zeros(deg+1,)
    return poly

def filter_by_counts(df,column,bins,cutoff):
    """Slices distribution along column and filters slices with low counts.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    column : str
        Name of the column that will be sliced for filter
    bins : int
        Number of bins to slice the distribution
    cutoff : int
        The number of particles below which the slice is removed
    
    Returns
    -------
    Dataframe
        Filtered version of df
    """
    
    df_copy = df.copy()
    df_copy['bin'] = pd.cut(df_copy[column], bins=bins)
    bin_freq = df_copy.loc[:,[column,'bin']].groupby('bin').count()
    df_copy = df_copy.loc[:,['delta','bin']].merge(bin_freq, 
                    on='bin', 
                    how='left',
                    suffixes=("_bin", 
                              "_bin_freq"))
    df_copy.columns = [column,'bin','freq']
    ind = df_copy.freq.values > cutoff
    
    return df.iloc[ind,:]

def get_twiss_z_slices(df,bins):
    """Caulate the alpha,beta,emit,z_mean for z slice of the distribution.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    bins : float
        Number of z slices
    
    Returns
    -------
    alpha : array_like, shape=(len(bins),)
        Twiss alpha parameter
    beta : array_like, shape=(len(bins),)
        Twiss beta parameter
    emit : array_like, shape=(len(bins),)
        Geometric Emittance
    z_mean : array_like, shape=(len(bins),)
        Mean z value of the slice
    """
    
    group_bins = pd.cut(df['z'],bins=bins)
    df_groups = df.groupby(group_bins)
    
    z_mean = df_groups['z'].mean().values
    
    a = (df_groups.apply(lambda x: get_twiss_parameters(x)))
    emit, alpha, beta = np.hstack(a)
    beta = np.reshape(beta, [-1,3],order='C')
    emit = np.reshape(emit, [-1,3],order='C')
    alpha = np.reshape(alpha, [-1,3],order='C')

    return alpha,beta,emit,z_mean

def make_sigma_mat(alpha,beta):
    """Creates a sigma matrix from the alpha and beta Twiss parameters
          
    Sigma Matrix is of the following form:
        
        |beta  -alpha|
        |-alpha gamma|  
        
    Parameters
    ----------
    alpha : array_like
        Alpha Twiss Parameter
    beta : array_like
        Beta Twiss Parameter
        
    Returns
    -------
    sigmax : ndarray, shape=(2,2)
        x sigma matrix
    sigmay : ndarray, shape=(2,2)
        y sigma matrix
    sigmaz : ndarray, shape=(2,2)
        z sigma matrix
    """
    
    gamma = (1+alpha**2)/beta
    sigmax = []
    sigmay = []
    sigmaz = []
    
    for b,a,g in zip(beta[:,0:3],alpha[:,0:3],gamma[:,0:3]):
        sigmax.append(np.array([[b[0],-a[0]],[-a[0],g[0]]]))
        sigmay.append(np.array([[b[1],-a[1]],[-a[1],g[1]]]))
        sigmaz.append(np.array([[b[2],-a[2]],[-a[2],g[2]]]))
    
    return sigmax,sigmay,sigmaz

def calculate_twiss_mismatch(sigma,sigma0):
    """Calulates the mismatch of the Twiss parameters.
        
        Note that the inputs are the sigma matrix, which has the form:
        sigma[1,1] = beta
        sigma[1,2] = -alpha
        sigma[2,1] = -alpha
        sigma[2,2] = gamma
        
        The mismatch parameter is greater than or equal to 1, with a 
        perfect match giving 1.
        
    Parameters
    ----------
    sigma : ndarray, shape=(2,2)
        Measured sigma matrix
    sigma_0 : ndarray, shape=(2,2)
        Design sigma matrix
        
    Returns
    -------
    float
        Twiss mismatch
    """
    
    bmag = []
    for sig in sigma:
        bmag.append(0.5*np.trace(sig @ np.linalg.inv(sigma0))) 
        
    return bmag

def print_twiss(df,kinetic_energy):
    """Prints Twiss Parameters
    
    Includes the normalized emittance, emittance, alpha, beta, and gamma.
        
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    kinetic_energy: float
        Average kinetic energy of distribution
    """
    
    gamma_mean = pau.KE2gamma(kinetic_energy)
    beta_mean = pau.gamma2beta(gamma_mean)
    
    emit, alpha, beta = get_twiss_parameters(df)
    emitn = gamma_mean*beta_mean*emit
    gamma = (1+alpha**2)/beta
    
    print('emitn =',emitn)
    print('alpha =',alpha)
    print('beta =',beta)
    print('gamma =',gamma)
    print('emit =',emit)

def twiss_ellipse(alpha,beta,emit,xy=[0,0],scalex=0,scaley=0,**ell_kwargs):
    """Creates a ellipse patch from Twiss parameters
    
    Uses matplotlib.patches.Ellipse to create a patch from the Twiss parameters. 
    Patches can be added to an existing figure using
        
        fig, ax = plt.subplots()
        ax.add_patch(ell)
          
    Parameters
    ----------
    alpha : float
    beta : float (>0)
    emit : float (>0)
    xy : array_like, shape=(2,)
        Center of the ellipse
    scalex : int (>0)
        Scale of the ellipse along x axis (*10**scalex)
    scaley : int (>0)
        Scale of the ellipse along y axis (*10**scalex)
    **ell_kwargs
        kwarg for matplotlib.patches.Ellipse
        
    Returns
    -------
     matplotlib.patches.Ellipse
        Ellipse created from Twiss parameters
    """
    
    gamma = (1+alpha**2)/beta

    xWaist = np.sqrt(emit/beta);
    thetaMax = np.sqrt(emit*beta);
    angle = get_angle_twiss(alpha,beta)

    ell = Ellipse(xy=xy, width=2*xWaist*10**scalex, height=2*thetaMax*10**scaley, angle = 180+angle,**ell_kwargs)
    
    return ell

def twiss_ellipse_parametric(alpha,beta,emit,num_points=100,xy=[0,0]):
    """Creates a ellipse patch from Twiss parameters
    
    Uses matplotlib.patches.Ellipse to create a patch from the Twiss parameters. 
    Patches can be added to an existing figure using
        
        fig, ax = plt.subplots()
        ax.add_patch(ell)
          
    Parameters
    ----------
    alpha : float
    beta : float (>0)
    emit : float (>0)
    xy : array_like, shape=(2,)
        Center of the ellipse
        
    Returns
    -------
     x : ndarray
        x coordinates of ellipse
     y : ndarray
        y coordinates of ellipse
    """
    
    xWaist = np.sqrt(emit/beta);
    thetaMax = np.sqrt(emit*beta);
    m = -alpha/beta
    t = np.linspace(0,2*np.pi,num_points);

    b = xWaist
    a = thetaMax
    x = a*np.cos(t)
    y = b*np.sin(t)
    y = y + x*m;
    x += xy[0] 
    y += xy[1] 
    
    return x,y

def get_angle_twiss(alpha,beta):
    """Calculates the angle of the distribution in phase space
          
    Parameters
    ----------
    alpha : array_like
        Alpha Twiss Parameter
    beta : array_like
        Beta Twiss Parameter
        
    Returns
    -------
     array_like
        phase space angle
    """
    
    gamma = (1+alpha**2)/beta
    return np.rad2deg(0.5*np.arctan2(2*alpha,gamma-beta))

def normalized_coord(df):
    """Normalizes phase space distributions to spherical space
    
    This  physical phase space coordinates and converts them 
    to normalized coordinates where the betatron phase change 
    is more easily observed. Normilization is the following:
    
        w(phi) = u/sqrt(beta) = a*cos(phi)
        
        dw(phi)/dphi = sqrt(beta)*u' + alpha/sqrt(beta)*u
                     = -a*sin(phi)
                     
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    
    Returns
    -------
    DataFrame
        Particle distribution in normalized units:
        x,xp,y,yp,z,deltaGamma/gamma~deltaP/P
    """
    
    df_norm = df.copy()
    
    emit, alpha, beta = get_twiss_parameters(df_norm)
    df_norm.x = df.x/np.sqrt(beta[0])
    df_norm.xp = np.sqrt(beta[0])*df.xp + df.x*alpha[0]/np.sqrt(beta[0]) 
    df_norm.y = df.y/np.sqrt(beta[1])
    df_norm.yp = np.sqrt(beta[1])*df.yp + df.y*alpha[1]/np.sqrt(beta[1]) 
    df_norm.z = df.z/np.sqrt(beta[2])
    df_norm.delta = np.sqrt(beta[2])*df.delta + df.z*alpha[2]/np.sqrt(beta[2]) 
    
    return df_norm

def get_phase_advance(df_norm,dims = ['x','y','z']):
    """Gets the angle of the distribution
    
    Calculates the betatron phase angle from the particle 
    distribution in normalized coordinates as output by 
    normalized_coord
             
    Parameters
    ----------
    df : DataFrame
        Particle distribution in normalized units:
        x,xp,y,yp,z,deltaGamma/gamma~deltaP/P
    dims: List {'x', 'y', 'z'}
    
    List : 
    
    Returns
    -------
    ndarray, shape(num_bin_x,num_bin_y,n)
        Bunching factor at the transverse positions
    """
    
    dim_dict = {'x':['x','xp'],
            'y':['y','yp'],
            'z':['z','delta']}
    
    df_copy = df_norm.copy()
    
    df_copy = df_copy - df_copy.mean(axis=0)
    deg = np.zeros([df_copy.shape[0],len(dims)])
    for ii,dim in enumerate(dims):
        deg[:,ii] = np.arctan2(df_copy.loc[:,dim_dict[dim][1]],
                               df_copy.loc[:,dim_dict[dim][0]])
    
    return pd.DataFrame(deg, columns = dims)

def binning(df,dim,num_bin):
    """Bins Dataframe based on column
    
    Adds a column to the dataframe named "bin_" plus 
    the specified column.
                     
    Parameters
    ----------
    df : DataFrame
        Arbitrary dataframe
    dim: str
        Column of df along which binning will occur
    num_bin : int
        Number of bins to divide along
        
    Returns
    -------
    DataFrame
        Copy of the initial dataframe with an additional 
        column for the binning names "bin_" plus 
        the specified column.
    """
    
    df_copy = df.copy()

    labels = np.arange(0, num_bin)
    bin_name = "bin_" + dim;
    
    minVal = df_copy[dim].min(axis=0)
    maxVal = df_copy[dim].max(axis=0)
    bins = np.linspace(minVal,maxVal,num_bin+1)
    df_copy[bin_name] = pd.cut(df_copy[dim], bins=bins, labels=labels, include_lowest=True)
    
    return df_copy

if __name__ == '__main__':
    pass
