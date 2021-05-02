'''
Description: 
Author: heji
Date: 2021-04-26 15:19:09
LastEditTime: 2021-04-29 20:30:18
LastEditors: GuoYi
'''

import math 
import torch 
import numpy as np 
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


def computeDeltasCube(geo, alpha):
    # Get coords of Img(0,0)
    P0 = {'x': -(geo['sVoxelX']/2 - geo['dVoxelX']/2) + geo['offOriginX'], 
          'y': -(geo['sVoxelY']/2 - geo['dVoxelY']/2) + geo['offOriginY']}
    
    # Get coors from next voxel in each direction
    Px0 = {'x': P0['x'] + geo['dVoxelX'], 'y': P0['y']} 
    Py0 = {'x': P0['x'], 'y': P0['y'] + geo['dVoxelY']}
    
    P = {'x': P0['x'] *math.cos(alpha)-P0['y'] *math.sin(alpha),
         'y': P0['x'] *math.sin(alpha)+P0['y'] *math.cos(alpha)}

    Px = {'x': Px0['x'] *math.cos(alpha)-Px0['y'] *math.sin(alpha),
          'y': Px0['x'] *math.sin(alpha)+Px0['y'] *math.cos(alpha)}

    Py = {'x': Py0['x'] *math.cos(alpha)-Py0['y'] *math.sin(alpha),
          'y': Py0['x'] *math.sin(alpha)+Py0['y'] *math.cos(alpha)}

    # Scale coords so detector pixels are 1x1
    Px['y'] =Px['y']/geo['dDetecU']
    P['y']  =P['y']/geo['dDetecU']
    Py['y'] =Py['y']/geo['dDetecU']

    # Compute unit vector of change between voxels
    deltaX ={'x': Px['x']-P['x'], 'y': Px['y']-P['y']}
    deltaY ={'x': Py['x']-P['x'], 'y': Py['y']-P['y']}

    return P, deltaX, deltaY



def PixelIndexCal_cuda(geo):
    mod = SourceModule("""
    __global__ void KernelPixelIndexCal_cuda(
                            const float *geo, const float *xyzorigin, 
                            const float *deltaX, const float *deltaY, 
                            const float *alpha, const float *angle, 
                            const int *mode, float *u, float *w)
    {
        const int indX = blockIdx.x * blockDim.x + threadIdx.x;
        const int indY = blockIdx.y * blockDim.y + threadIdx.y;

        int extent = (int)geo[3], nVoxelX = (int)geo[8], nVoxelY = (int)geo[9];

        unsigned long idx = indX*nVoxelY+indY;
        
        if ((indX>=nVoxelX) || (indY>=nVoxelY))
            return;

        float DSD = geo[0], DSO = geo[1], nDetecU = geo[2], sVoxelX = geo[4], sVoxelY = geo[5], dVoxelX = geo[6], dVoxelY = geo[7];

        float P_x = xyzorigin[0] + indX * deltaX[0] + indY * deltaY[0];
        float P_y = xyzorigin[1] + indX * deltaX[1] + indY * deltaY[1];

        float S_x = DSO;
        float S_y;

        if (mode[0] == 0)
            S_y = P_y;
        else if (mode[0] == 1)
            S_y = 0.0;

        float vectX = P_x - S_x;
        float vectY = P_y - S_y;

        float t = (DSO - DSD - S_x) / vectX;
        float y = vectY * t + S_y;

        float detindx = y + nDetecU / 2;

        float realx = -1*sVoxelX/2 + dVoxelX/2 + indX *dVoxelX;
        float realy = -1*sVoxelY/2 + dVoxelY/2 + indY *dVoxelY;

        float weight = (DSO + realy *sin(alpha[0]) - realx *cos(alpha[0])) / DSO;
        
        weight = 1 / (weight *weight);

        if (detindx > (nDetecU-2))
            detindx = nDetecU-2;
        if (detindx < 1)
            detindx = 1;

        float tmp_index = detindx + nDetecU * angle[0];

        if (extent == 1)
        {
            u[idx] = tmp_index;
            w[idx] = weight;
        }
        else if (extent == 2)
        {
            if ((detindx - (int)detindx) > 0.5)
            {
                u[idx*extent+0] = tmp_index - 1.0;
            }
            else if ((detindx - (int)detindx) < 0.5)
            {
                u[idx*extent+0] = tmp_index + 1.0;
            }

            u[idx*extent+1] = tmp_index;

            w[idx*extent+0] = weight/2;
            w[idx*extent+1] = weight/2;
        }
        else if (extent == 3)
        {
            u[idx*extent+0] = tmp_index - 1.0;
            u[idx*extent+1] = tmp_index;
            u[idx*extent+2] = tmp_index + 1.0;

            w[idx*extent+0] = weight/3;
            w[idx*extent+1] = weight/3;
            w[idx*extent+2] = weight/3;
        }
    }
    """)

    KernelPixelIndexCal_cuda = mod.get_function("KernelPixelIndexCal_cuda")


    nTheads = 32  # nTheads is no more than 32, becuase the total nTheads in one block should be no more than 1024, i.e., block=(nTheads, nTheads, 1),  nTheads*nTheads*1 <= 1024.
    nBlocks = (geo['nVoxelX'] + nTheads - 1) // nTheads

    alphas = np.linspace(geo['start_angle'], geo['end_angle'], geo['views'], False)
    sino_indices = torch.zeros(geo['nVoxelX']*geo['nVoxelY']*geo['extent'], geo['views']).type(torch.LongTensor)
    sino_weights = torch.zeros(geo['nVoxelX']*geo['nVoxelY']*geo['extent'], geo['views']).type(torch.FloatTensor)

    for angle in range(geo['views']):
        alpha = -alphas[angle]
        xyzorigin_dic, deltaX_dic, deltaY_dic = computeDeltasCube(geo, alpha)
        indices = np.zeros(geo['nVoxelX']*geo['nVoxelY']*geo['extent'], dtype=np.float32)
        weights = np.zeros(geo['nVoxelX']*geo['nVoxelY']*geo['extent'], dtype=np.float32)
        tmp_geo = np.array([geo[i] for i in ['DSD', 'DSO', 'nDetecU', 'extent', 'sVoxelX', 'sVoxelY', 'dVoxelX', 'dVoxelY', 'nVoxelX', 'nVoxelY']], dtype=np.float32)
        xyzorigin = np.array(list(xyzorigin_dic.values()), dtype=np.float32)
        deltaX = np.array(list(deltaX_dic.values()), dtype=np.float32)
        deltaY = np.array(list(deltaY_dic.values()), dtype=np.float32)
        tmp_angle = np.array([angle], dtype=np.float32)
        tmp_mode = np.array([0 if geo['mode'] is 'parallel' else 1])
        tmp_alpha = np.array([alpha], dtype=np.float32)

        KernelPixelIndexCal_cuda(drv.In(tmp_geo), drv.In(xyzorigin), drv.In(deltaX), drv.In(deltaY), 
                                 drv.In(tmp_alpha), drv.In(tmp_angle), drv.In(tmp_mode), drv.InOut(indices), 
                                 drv.InOut(weights), block=(nTheads, nTheads, 1), grid=(nBlocks, nBlocks))

        sino_indices[:, angle] = torch.from_numpy(indices)
        sino_weights[:, angle] = torch.from_numpy(weights)

    return sino_indices.view(-1), sino_weights.view(-1)

