#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int FlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		at::Tensor& input1,
		at::Tensor& count,
		at::Tensor& output

		);


