#ifndef ENCRYPT_KERNEL_H
#define ENCRYPT_KERNEL_H

// Declaration of the encryption function
extern "C" void encrypt_Kernel(u32_t* dev_input,
                               u32_t* dev_output,
                               size_t pitch_a,
                               size_t pitch_b,
                               u32_t* dev_sm_te1,
                               u32_t* dev_sm_te2,
                               u32_t* dev_sm_te3,
                               u32_t* dev_sm_te4);

#endif // ENCRYPT_KERNEL_H