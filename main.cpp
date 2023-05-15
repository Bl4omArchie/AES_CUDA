#include <iostream>
#include "encrypt_kernel.h"


// Define the number of blocks and threads per block
#define BLOCK_NUM 8
#define THREAD_NUM 128

// Example data
const int data_size = BLOCK_NUM * THREAD_NUM;
u32_t host_input[data_size];
u32_t host_output[data_size];

// Function to invoke the kernel and perform encryption
void performEncryption()
{
    // Declare device pointers
    u32_t* dev_input;
    u32_t* dev_output;
    u32_t* dev_sm_te1;
    u32_t* dev_sm_te2;
    u32_t* dev_sm_te3;
    u32_t* dev_sm_te4;

    // Allocate device memory
    cudaMalloc((void**)&dev_input, sizeof(u32_t) * data_size);
    cudaMalloc((void**)&dev_output, sizeof(u32_t) * data_size);
    cudaMalloc((void**)&dev_sm_te1, sizeof(u32_t) * 256);
    cudaMalloc((void**)&dev_sm_te2, sizeof(u32_t) * 256);
    cudaMalloc((void**)&dev_sm_te3, sizeof(u32_t) * 256);
    cudaMalloc((void**)&dev_sm_te4, sizeof(u32_t) * 256);

    // Copy input data to device
    cudaMemcpy(dev_input, host_input, sizeof(u32_t) * data_size, cudaMemcpyHostToDevice);

    // Invoke the kernel
    encrypt_Kernel<<<BLOCK_NUM, THREAD_NUM>>>(dev_input, dev_output, 0, 0, dev_sm_te1, dev_sm_te2, dev_sm_te3, dev_sm_te4);

    // Copy output data from device
    cudaMemcpy(host_output, dev_output, sizeof(u32_t) * data_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_sm_te1);
    cudaFree(dev_sm_te2);
    cudaFree(dev_sm_te3);
    cudaFree(dev_sm_te4);
}

int main()
{
    // Initialize input data
    for (int i = 0; i < data_size; i++)
    {
        host_input[i] = i;
    }

    // Perform encryption
    performEncryption();

    // Print the encrypted output
    std::cout << "Encrypted Output: ";
    for (int i = 0; i < data_size; i++)
    {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
