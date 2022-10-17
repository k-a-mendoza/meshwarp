import numpy as np

class StaticOffset:
    
    def __init__(self,offset=0):
        self.offset = offset
        
    def __call__(self,grid):
        offset_grid = np.ones(grid.shape)*self.offset
        return offset_grid


class SinOffset:
    
    def __init__(self, wavelength, amplitude=1.0, phase_shift=0):
        self.phase_shift  = np.pi*2*phase_shift/wavelength
        self.frequency    = np.pi*2/wavelength
        self.amplitude=amplitude
        
    def __call__(self, grid):
        offset_grid = self.amplitude*np.sin(grid*self.frequency - self.phase_shift)
        return offset_grid

class CosOffset:
    
    def __init__(self, wavelength, amplitude=1.0, phase_shift=0):
        
        self.phase_shift  = np.pi*2*phase_shift/wavelength
        self.frequency    = np.pi*2/wavelength
        self.amplitude=amplitude
        
    def __call__(self, grid):
        offset_grid = self.amplitude*np.cos(grid*self.frequency - self.phase_shift)
        return offset_grid

class CosLobeOffset:
    
    def __init__(self, wavelength, amplitude=1.0, phase_shift=0):
        self.phase_shift  = np.pi*2*phase_shift/wavelength
        self.frequency    = np.pi*2/wavelength
        self.amplitude=amplitude
        
    def __call__(self, grid):
        offset_grid = self.amplitude*np.cos(grid*self.frequency - self.phase_shift)**2
        return offset_grid

def get_offset_obj(name,**kwargs):
    """
    returns a spline object
    
    
    Parameters
    ==========
    name : str
        can be any named y spline 'east_coast', 'west_coast', 'gulf', in which case the spline object 
    corresponds to points defined in a previous database. If name=='x', a StaticOffset object is created instead.
    for StaticOffset objects please supply a keyword-argument offset
    
    offset : float
        optional argument used when requesting a StaticOffset object. The amount in grid coordinates to offset the spline
        
    Returns
    =======
    spline : 
        a callable spline object which returns the expected x offset for a given y coordinate
    """
    if 'static' in name:
        spline = StaticOffset(**kwargs)
    elif 'sin' in name:
        spline = SinOffset(**kwargs)
    elif 'coslobe' in name:
        spline = CosLobeOffset(**kwargs)
    elif 'cos' in name:
        spline = CosOffset(**kwargs)
    return spline



def adjust_mesh_x_direction(mesh, trapezoid_x_kwargs, trapezoid_y_kwargs, x_offset,
                            spline_source=None,delta_remove=None):
    """
    
    adjusts mesh nodes x coordinates to follow a N-S oriented coastline using a trapezoidal weighting scheme.
    
    given a function f(y)=x which defines roughly where the coastline is, the coordinates are adjusted like:
    
    x'(x,y) = x(x,y) + W1_x(x)*f(y-y0)*W2_y(y)
    
    windows W1 and W2 are defined like so:
    
    1 |         -----------
      |       /             \
    W |     /                 \
      |   /                     \
    0 |-----------------------------
         x0    x1         x2     x3
         
    the functions themselves have a range of [0,1] and exhibit triangular regions on both the upper and lower side 
    of a flat-topped region with a weight of 1. This allows for coastline adjustments to only affect areas out of the central
    dense mesh if desired. The inclusion of 2 separate W functions allows for assymetric coastline weighting dependent on the 
    constraints of the problem.
    
   
    Parameters
    ==========
    
    mesh : HexMesh
        a hexmesh object
        
    trapezoid_x_kwargs : dict
        keyword-arguments defining the x extent of the mesh x_coordinates transformation. Needs to have
        the keys x0, x1, x2, and x3 to define the trapezoid window.
        
        
    trapezoid_y_kwargs : dict
        keyword-arguments defining the y extent of the mesh x_coordinates transformation. Needs to have
        the keys x0, x1, x2, and x3 to define the trapezoid window.
        
    x_offset : float
        the static shift to apply to the mesh prior to determining offset
        
    SplineSource
        a spline source from the warp module.
    
    diff_adjust : bool
        if True, adjusts the local mesh by the difference instead of simply the spline values. Defaults to False
        
    Returns
    =======
    mesh : HexMesh
    
    """
        
    y_extent   = apply_hann_window(mesh.y_grid, **trapezoid_y_kwargs)
    x_extent   = apply_hann_window(mesh.x_grid, **trapezoid_x_kwargs)

    splined_values = spline_source(mesh.y_grid)-x_offset
    if delta_remove is not None:
        splined_values-=delta_remove
    
    effect = y_extent*splined_values*x_extent
    mesh.x_grid+=effect
    return mesh, effect

def adjust_mesh_y_direction(mesh, trapezoid_x_kwargs, trapezoid_y_kwargs, y_offset,
                            spline_source=None,delta_remove=None):
    """
    
    adjusts mesh nodes x coordinates to follow a N-S oriented coastline using a trapezoidal weighting scheme.
    
    given a function f(y)=x which defines roughly where the coastline is, the coordinates are adjusted like:
    
    x'(x,y) = x(x,y) + W1(x)*f(x-x0)*W2(y)
    
    windows W1 and W2 are defined like so:
    
    1 |         -----------
      |       /             \
    W |     /                 \
      |   /                     \
    0 |-----------------------------
         x0    x1         x2     x3
         
    the functions themselves have a range of [0,1] and exhibit triangular regions on both the upper and lower side 
    of a flat-topped region with a weight of 1. This allows for coastline adjustments to only affect areas out of the central
    dense mesh if desired. The inclusion of 2 separate W functions allows for assymetric coastline weighting dependent on the 
    constraints of the problem.
    
   
    Parameters
    ==========
    
    mesh : HexMesh
        a hexmesh object
        
    trapezoid_x_kwargs : dict
        keyword-arguments defining the x extent of the mesh x_coordinates transformation. Needs to have
        the keys x0, x1, x2, and x3 to define the trapezoid window.
        
        
    trapezoid_y_kwargs : dict
        keyword-arguments defining the y extent of the mesh x_coordinates transformation. Needs to have
        the keys x0, x1, x2, and x3 to define the trapezoid window.
        
    x_offset : float
        the static shift to apply to the mesh prior to determining offset
        
    spline_source : SplineSource
        a spline source from the warp module.
        
    Returns
    =======
    mesh : HexMesh
    
    """        
    y_extent   = apply_hann_window(mesh.y_grid, **trapezoid_y_kwargs)
    x_extent   = apply_hann_window(mesh.x_grid, **trapezoid_x_kwargs)

    splined_values = spline_source(mesh.x_grid)-y_offset
    if delta_remove is not None:
        splined_values-=delta_remove
    effect = y_extent*splined_values*x_extent
    mesh.y_grid+=effect
    return mesh, effect


def apply_hann_window(grid, x0, x1, x2, x3):
    """
    applies a tophat function between x1 and x2, and mirrored halves of a hann window between 
    x0-x1 and x2-x3.
    
    Returns
    =======
    alpha : 
    """
    alpha = np.ones(grid.shape)
    alpha[(grid<x0) | (grid>x3)]=0
    
    first_cosine_values = (grid>x0) & (grid<x1)
    first_wavelength = x1-x0
    alpha[first_cosine_values] =  np.sin(np.pi*(grid[first_cosine_values]-x0)/(2*first_wavelength))**2
    second_cosine_values = (grid>x2) & (grid<x3)
    second_wavelength = x3-x2
    alpha[second_cosine_values]=1-np.sin(np.pi*(grid[second_cosine_values]-x2)/(2*second_wavelength))**2
    
    if np.all(alpha==0):
        print('all alpha values are zero. is something wrong?')
    return alpha