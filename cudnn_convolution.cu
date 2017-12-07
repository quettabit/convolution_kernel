#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <utility>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


int main(int argc, const char* argv[]) {


    int gpu_id = 0;
    int img_ht = 2048;
    int img_wd = 2048;


    cudaSetDevice(gpu_id);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/1,
                                        /*image_height=*/img_ht,
                                        /*image_width=*/img_wd));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/1,
                                        /*in_channels=*/1,
                                        /*kernel_height=*/3,
                                        /*kernel_width=*/3));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/0,
                                             /*pad_width=*/0,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CONVOLUTION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));

    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));




    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/1,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
                                          &convolution_algorithm));

    size_t workspace_bytes;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
    std::cerr << "Workspace size: " <<  workspace_bytes  << "bytes"
            << std::endl;


    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_dims = img_ht * img_wd;
    int image_bytes = image_dims * sizeof(float);
    float *h_input = new float[image_bytes];
    for(int i=0; i< image_dims; i++){
        h_input[i] = 1;
    }

    float* d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice);

    float* d_output{nullptr};
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    // clang-format off
    const float kernel_template[3][3] = {
    {0.5, 0.5, 0.5},
    {0.5, 0.5, 0.5},
    {0.5, 0.5, 0.5}
    };
    // clang-format on

    float h_kernel[1][1][3][3];
    for (int kernel = 0; kernel < 1; ++kernel) {
    for (int channel = 0; channel < 1; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
    }

    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_kernel,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));



    float* h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);


    std::vector<std::pair<int,int> > miss;

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            //std::cout<<h_output[i*height +j]<<" ";
            if(h_output[i*height +j] != 4.5){
                miss.push_back(std::make_pair(i,j));
            }
        }
        //std::cout<<"\n";
    }

    std::cout<<miss.size()<<"\n";
    for(int i=0;i<miss.size();i++){
        std::cout<<miss[i].first<<","<<miss[i].second<<"\n";
    }

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}
