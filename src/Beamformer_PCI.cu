// Written by Sua Bae
// 2/22/2021
//    Beamformer for Power Cavitation Imaging (PCI) with L22-14
//
#include <cuda.h>
#include <cuda_runtime.h>

#define nThdPerBlk_max 256
#define PI 3.14159265

__constant__ float            trans_aElePos[256];	// [m] transducer element position in x
__constant__ unsigned int 		trans_nNumEle;		// num of txdcr elements
__constant__ unsigned int 		rf_nSdim;			// num of samples of RF data
__constant__ unsigned int 		rf_nCdim; 			// num of channels of RF data
__constant__ unsigned int 		rf_nPdim; 			// num of pulses (num of tx/rx) of RF data = num of focused pulses for each burst

__constant__ unsigned int 		bf_nXdim; 			// num of pixels in x
__constant__ unsigned int 		bf_nZdim;			// num of pixels in z
__constant__ float 			bf_dx; 			    // [m] x pixel size 
__constant__ float 			bf_dz;			    // [m] z pixel size 
__constant__ float 			bf_nXstart; 			// [m] 1st pixel position in x 
__constant__ float 			bf_nZstart;		    // [m] 1st pixel position in z

__constant__ float 			nFs; 				// [Hz] sampling frequency of RF data
__constant__ float 			nSoundSpeed;			// [m/s] sound speed of material
__constant__ float 			nRxFnum;				// f-number for receive beamforming
__constant__ float 			nLensCorr; 			// [m] lens correction valuem, will be ADDED to the round-trip delay
__constant__ float			nOffset_m; 			// [m] offset of acquired Rf Data, will be SUBTRACTED from the round-trip delay (for example, P.startDepth*nWavelength_m), 
        
__global__ void _Beamformer_PCI(float* aBfData_zxp, float* aRfData_scp, float* aTxDelay_zx)
{
	/* 
	Beamforming the received data acquired by transmiting one focused beam (creating a whole 2D image from one tx/rx channel data)
	Using TXPD for transmit delay value
	
	INPUT:   aRfData_scp   	: RF data,          dimension order = (sample, channel, pulse)
			aTxDelay_zx		: Tx delay computed by TXPD of Verasonics, dimension order = (z, x)
	OUTPUT:  aBfData_zxp   	: Beamformed data , dimension order = (z, x, pulse)
	
    !!! Thread must not be defined in hreadIdx.z dimension !!!
    !!! blockIdx.z must equal to rf_nPdim
	*/
 
	int xidx = threadIdx.x + blockIdx.x * blockDim.x;// BF scanline index
	int zidx = threadIdx.y + blockIdx.y * blockDim.y;// BF depth index
	int pidx = blockIdx.z; // pulse index 
            
    //if (zidx < bf_nZdim) { // not required because blockDim.z = bf_nZdim
        if (xidx < bf_nXdim) {

            // 1) BF positions
            float x = bf_nXstart + xidx*bf_dx; // [m]
            float z = bf_nZstart + zidx*bf_dz; // [m]
		
			// 2) Aperture size
			float nAptSize_m = z / nRxFnum + (trans_aElePos[1]-trans_aElePos[0]); // plus a pitch to prevent nApod from being NaN and to use at least one channel

			// 4) Calc transmit delay
			float nTxDist = aTxDelay_zx[zidx + xidx*bf_nZdim]; // [m] transmit delay computed by TXPD of Verasonics

			float nRf = 0;
			float c_x, nDistance_m, nApod, nRxDist, nDelay_m, nDelay_pixel; 
			int nDelay_int, nAdd;
			float nDelay_frc, nRf1, nRf2, nRf_intp;
			int nAdd_output;
			#pragma unroll
			for (int cidx = 0; cidx < rf_nCdim; cidx++)
			{
				// 4) Channel positions
				c_x = trans_aElePos[cidx]; // [m] Transducer element x position
			
				nDistance_m = abs(c_x - x); // lateral distance from the beamforming point to the current channel (element)
				if (nDistance_m <= nAptSize_m / 2) // if the element is within the aperture, compute!!
				{
					// 5) Calc Apodization window
					nApod = (0.53836 + 0.46164*cos(2 * 3.141592653*nDistance_m / nAptSize_m));   // Hamming window equation
			
					nRxDist = sqrt((x - c_x)*(x - c_x) + z*z); // [m] dim = (samples for BF) x 1
					nDelay_m = (nTxDist + nRxDist) + nLensCorr - nOffset_m; //[m]
					nDelay_pixel = nDelay_m/nSoundSpeed*nFs; // [pixel] real = int + frc // [meter] ->[pixel] 
					nDelay_int = floor(nDelay_pixel); 
			
					if (nDelay_int < rf_nSdim-1 ) // proceed only when the sample exists
					{
						nDelay_frc 	= nDelay_pixel - nDelay_int; // fractional value
						nAdd 		= nDelay_int + cidx*rf_nSdim + pidx*rf_nSdim*rf_nCdim; // (s,c,p) address of aIqData_sca wihtout considering I/Q
						// (4) linear interpolation				
						nRf1 = aRfData_scp[nAdd];
						nRf2 = aRfData_scp[nAdd + 1];
						nRf_intp = nRf1*(1 - nDelay_frc) + nRf2*nDelay_frc;
			
						// (6) apply aperture & accumulate
						nRf = nRf + nRf_intp*nApod;
					}
			
				}
			}
			
			// (7) export
			nAdd_output = zidx + xidx*bf_nZdim + pidx*bf_nZdim*bf_nXdim;  // (z,x,p) address of aBfData_zxp
			aBfData_zxp[nAdd_output] = nRf; // channel-summed beamformed data

			
        }
    //}
	
}



