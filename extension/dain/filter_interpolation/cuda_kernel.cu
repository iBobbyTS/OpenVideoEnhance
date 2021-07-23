#include <stdio.h>

#include "cuda_kernel.cuh"


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

#define DEBUG (0)
#ifndef BLOCKDIMX
#define BLOCKDIMX (32)
#endif
#ifndef BLOCKDIMY
#define BLOCKDIMY (16)
#endif
using at::Half;




//forward path of our layer
template <typename scalar_t>
__global__ void FilterInterpolationLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		const scalar_t* __restrict__    input1,    		const scalar_t* __restrict__    input2,    	const scalar_t* __restrict__    input3, 	scalar_t*   output

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;


	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {

		float fx = input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i  ];
		float fy = input2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i  ];

		float x2 = (float)(w_i) + fx;
		float y2 = (float)(h_i) + fy;


		if(x2 >= 0.0f && y2 >=0.0f && x2 <= (float)(w -1) && y2 <= (float)(h-1)
            && fabs(fx) < (float)(w)/2.0f && fabs(fy) < (float)(h)/2.0f){
			int ix2_L = int(x2) + 1 - (int)(filter_size / 2);
			int iy2_T = int(y2) + 1 - (int)(filter_size / 2);
			int ix2_R = ix2_L + filter_size;
			int iy2_B = iy2_T + filter_size;

            float alpha = x2 - (int)(x2);
            float beta = y2 - (int)(y2);


			//TODO: here is a bug that if the iy2_B or ix2_R gets out of the border, than there is no enough pixels to warp the target one.
			for (int c_i = 0 ; c_i < channel ; c_i++){

                float TL = 0.0f;
                for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                    int _filter_j = min(max(0, filter_j), h - 1);
                    for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i ), w - 1);
                    TL += input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] *
							input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] ;
                    }
                }

                float TR = 0.0f;
                for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    TR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BL = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BL += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BR = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ] =
                            (1-alpha)*(1-beta)*TL +
							alpha*(1-beta)*TR +
							(1-alpha)*beta*BL +
							alpha*beta*BR;

//					for( int filter_i = ix2_L; filter_i < ix2_R ; filter_i ++ ){
//						int _filter_i = min(max(0, filter_i),w - 1);
//						output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ] +=
//							input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] *
//							input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] *
////							exp( -(fabs((float) filter_j - y2) + fabs((float) filter_i - x2)) / (float)(filter_size)); // the distance weight
//							exp( -(fabs((float) filter_j - y2) + fabs((float) filter_i - x2)) ); // the distance weight
//
////							if(w_i == 141 && h_i == 316 && c_i == 0 ){
////printf("gpu: %f, %f,%f,%f\n",input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] ,
////input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i],
////exp( -(fabs((float) filter_j - y2) + fabs((float) filter_i - x2)) / (float)(filter_size)),
////output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ]
//// );
////}
//
//					}
//				}
			}
		} else{
			//the warping data is out of range, we fill it with zeros
			for(int c_i = 0 ;  c_i < channel; c_i ++){
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] = input1[off + c_i* input1_c_stride+ h_i * input1_h_stride + w_i];
			}
		}
	}
	return ;

}


int FilterInterpolationLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		at::Tensor&  input1,    		at::Tensor&  input2,    	at::Tensor&  input3, 	at::Tensor&  output

		)
{
	int error = 1 ;

	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
		AT_DISPATCH_FLOATING_TYPES(input1.type(), "DepthFlowProjection_gpu_backward", ([&] {
FilterInterpolationLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,

			input1.data<scalar_t>(),input2.data<scalar_t>(),input3.data<scalar_t>(), output.data<scalar_t>()
			);
 					}));

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}
