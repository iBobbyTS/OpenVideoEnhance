#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cstdio>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>

#include "cuda_kernel.cuh"


int FilterInterpolationLayer_gpu_forward(
    at::Tensor&  input1,
    at::Tensor&  input2,
    at::Tensor&  input3,
    at::Tensor&  output
){
	int error = 1 ;

	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = FilterInterpolationLayer_gpu_forward_kernel(
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			input1,
			input2,
			input3,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}
	return error;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &FilterInterpolationLayer_gpu_forward, "FilterInterpolation forward (CUDA)");
}
