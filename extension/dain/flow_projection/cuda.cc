#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h> //works for 1.0.0

#include "cuda_kernel.cuh"

int FlowProjectionLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  count,
		at::Tensor&  output,
		int fillhole
		)
{

	int error = 1 ;

	int channel = input1.size( 1);
	if(channel!= 2) return error;
	int batch = input1.size(0);

	int h = input1.size(2);
	int w = input1.size(3);

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int count_b_stride = count.stride(0);
	int count_c_stride = count.stride(1);
	int count_h_stride = count.stride(2);
	int count_w_stride = count.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);
//    printf("In gpu forward\n");
	error = FlowProjection_gpu_forward_kernel(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch,fillhole,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,
			count,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("FlowProjectionLayer_gpu_forward", &FlowProjectionLayer_gpu_forward, "FlowProjection forward (CUDA)");
}
