#include <stdio.h>

#include "cuda_kernel.cuh"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32
#define FULL_MASK 0xffffffff

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

using at::Half;

template<typename scalar_t>
__forceinline__ __device__ scalar_t warpReduceSum(scalar_t val) {
        for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_down_sync(FULL_MASK, val, offset);
        return val;
}

template<typename scalar_t>
__forceinline__ __device__ scalar_t blockReduceSum(scalar_t val) {

        static __shared__ scalar_t shared[32];
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;

        val = warpReduceSum(val);

        if (lane == 0)
                shared[wid] = val;

        __syncthreads();

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

        if (wid == 0)
                val = warpReduceSum(val);

        return val;
}


template <typename scalar_t>
__global__ void channels_first(const scalar_t* __restrict__ input, scalar_t* rinput, int channels, int height, int width, int pad_size)
{

    // n (batch size), c (num of channels), y (height), x (width)
    int n = blockIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.z;

    int ch_off = threadIdx.x;
    scalar_t value;

    int dimcyx = channels * height * width;
    int dimyx = height * width;

    int p_dimx = (width + 2 * pad_size);
    int p_dimy = (height + 2 * pad_size);
    int p_dimyxc = channels * p_dimy * p_dimx;
    int p_dimxc = p_dimx * channels;

    for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
      value = input[n * dimcyx + c * dimyx + y * width + x];
      rinput[n * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] = value;
    }
}


template<typename scalar_t>
__global__ void correlation_forward(scalar_t* __restrict__ output, const int nOutputChannels,
                const int outputHeight, const int outputWidth, const scalar_t* __restrict__ rInput1,
                const int nInputChannels, const int inputHeight, const int inputWidth,
                const scalar_t* __restrict__ rInput2, const int pad_size, const int kernel_size,
                const int max_displacement, const int stride1, const int stride2) {

        int32_t pInputWidth = inputWidth + 2 * pad_size;
        int32_t pInputHeight = inputHeight + 2 * pad_size;

        int32_t kernel_rad = (kernel_size - 1) / 2;

        int32_t displacement_rad = max_displacement / stride2;

        int32_t displacement_size = 2 * displacement_rad + 1;

        int32_t n = blockIdx.x;
        int32_t y1 = blockIdx.y * stride1 + max_displacement;
        int32_t x1 = blockIdx.z * stride1 + max_displacement;
        int32_t c = threadIdx.x;

        int32_t pdimyxc = pInputHeight * pInputWidth * nInputChannels;

        int32_t pdimxc = pInputWidth * nInputChannels;

        int32_t pdimc = nInputChannels;

        int32_t tdimcyx = nOutputChannels * outputHeight * outputWidth;
        int32_t tdimyx = outputHeight * outputWidth;
        int32_t tdimx = outputWidth;

        int32_t nelems = kernel_size * kernel_size * pdimc;

        // element-wise product along channel axis
        for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
                for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
                        int x2 = x1 + ti * stride2;
                        int y2 = y1 + tj * stride2;

                        float acc0 = 0.0f;

                        for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                                for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                                        // THREADS_PER_BLOCK
                                        #pragma unroll
                                        for (int ch = c; ch < pdimc; ch += blockDim.x) {

                                                int indx1 = n * pdimyxc + (y1 + j) * pdimxc
                                                                + (x1 + i) * pdimc + ch;
                                                int indx2 = n * pdimyxc + (y2 + j) * pdimxc
                                                                + (x2 + i) * pdimc + ch;
                                                acc0 += static_cast<float>(rInput1[indx1] * rInput2[indx2]);
                                        }
                                }
                        }

                        if (blockDim.x == warpSize) {
                            __syncwarp();
                            acc0 = warpReduceSum(acc0);
                        } else {
                            __syncthreads();
                            acc0 = blockReduceSum(acc0);
                        }

                        if (threadIdx.x == 0) {

                                int tc = (tj + displacement_rad) * displacement_size
                                                + (ti + displacement_rad);
                                const int tindx = n * tdimcyx + tc * tdimyx + blockIdx.y * tdimx
                                                + blockIdx.z;
                                output[tindx] = static_cast<scalar_t>(acc0 / nelems);
                        }
            }
        }
}

int correlation_forward_cuda_kernel(at::Tensor& output,
                                    int ob,
                                    int oc,
                                    int oh,
                                    int ow,
                                    int osb,
                                    int osc,
                                    int osh,
                                    int osw,

                                    at::Tensor& input1,
                                    int ic,
                                    int ih,
                                    int iw,
                                    int isb,
                                    int isc,
                                    int ish,
                                    int isw,

                                    at::Tensor& input2,
                                    int gc,
                                    int gsb,
                                    int gsc,
                                    int gsh,
                                    int gsw,

                                    at::Tensor& rInput1,
                                    at::Tensor& rInput2,
                                    int pad_size,
                                    int kernel_size,
                                    int max_displacement,
                                    int stride1,
                                    int stride2,
                                    int corr_type_multiply,
                                    cudaStream_t stream) 
{

   int batchSize = ob;

   int nInputChannels = ic;
   int inputWidth = iw;
   int inputHeight = ih;

   int nOutputChannels = oc;
   int outputWidth = ow;
   int outputHeight = oh;

   dim3 blocks_grid(batchSize, inputHeight, inputWidth);
   dim3 threads_block(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "channels_first_fwd_1", ([&] {

  channels_first<scalar_t><<<blocks_grid,threads_block, 0, stream>>>(
      input1.data<scalar_t>(), rInput1.data<scalar_t>(), nInputChannels, inputHeight, inputWidth, pad_size);

  }));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.type(), "channels_first_fwd_2", ([&] {

  channels_first<scalar_t><<<blocks_grid,threads_block, 0, stream>>> (
      input2.data<scalar_t>(), rInput2.data<scalar_t>(), nInputChannels, inputHeight, inputWidth, pad_size);

  }));

   dim3 threadsPerBlock(THREADS_PER_BLOCK);
   dim3 totalBlocksCorr(batchSize, outputHeight, outputWidth);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "correlation_forward", ([&] {

   correlation_forward<scalar_t><<<totalBlocksCorr, threadsPerBlock, 0, stream>>> 
                        (output.data<scalar_t>(), nOutputChannels, outputHeight, outputWidth,
                         rInput1.data<scalar_t>(), nInputChannels, inputHeight, inputWidth,
                         rInput2.data<scalar_t>(),
                         pad_size,
                         kernel_size,
                         max_displacement,
                         stride1,
                         stride2);

  }));

  cudaError_t err = cudaGetLastError();


  // check for errors
  if (err != cudaSuccess) {
    printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}
