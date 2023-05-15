#include <cuda_runtime.h>

__global__ static void encrypt_Kernel(u32_t* dev_input,
                                      u32_t* dev_output,
                                      size_t pitch_a,
                                      size_t pitch_b,
                                      u32_t* dev_sm_te1,
                                      u32_t* dev_sm_te2,
                                      u32_t* dev_sm_te3,
                                      u32_t* dev_sm_te4)
{
    // local thread index and global index
    int tid = threadIdx.x;
    int index = THREAD_NUM * (BLOCK_NUM * blockIdx.y + blockIdx.x) + threadIdx.x;
    u32_t w1, w2, w3, w4, s1, s2, s3, s4;
    // store the T-boxes and sbox in shared memory.
    __shared__ u32_t sm_te1[256], sm_te2[256], sm_te3[256], sm_te4[256];
    __shared__ u8_t sm_sbox[256];

    if (tid < 256)
    {
        // load dev_sm_te1, dev_sm_te2, dev_sm_te3, dev_sm_te4, and const_sm_sbox to shared memory variables sm_te1, sm_te2, sm_te3, sm_te4, and sm_sbox;
        sm_te1[tid] = dev_sm_te1[tid];
        sm_te2[tid] = dev_sm_te2[tid];
        sm_te3[tid] = dev_sm_te3[tid];
        sm_te4[tid] = dev_sm_te4[tid];
        sm_sbox[tid] = const_sm_sbox[tid];
    }

    // load the cipher blocks, all the global memory transactions are coalesced.
    // The original plain text load from files, due to the read procedure reverse the byte order of the 32-bit words, So a reverse process was necessary.
    w1 = dev_input[index];
    w1 = ((w1 >> 24) & 0xFF) | ((w1 >> 8) & 0xFF00) | ((w1 << 8) & 0xFF0000) | ((w1 << 24) & 0xFF000000);
    // first round AddRoundKey: ex-or with round key
    w1 ^= const_m_ke[0];
    // round transformation: a set of table lookup operations.

    #pragma unroll
    for (int i = 1; i < 10; i++)
    {
        s1 = (sm_te1[w1 >> 24] ^ sm_te2[(w1 >> 16) & 0xFF] ^ sm_te3[(w1 >> 8) & 0xFF] ^ sm_te4[w1 & 0xFF]) ^ const_m_ke[i * 4];
        w1 = s1;
    }
    // The final round doesn’t include the MixColumns
    w1 = (u32_t)(sm_sbox[s1 >> 24]) << 24;          // SubBytes and ShiftRows
    w1 |= (u32_t)(sm_sbox[(s1 >> 16) & 0xFF]) << 16;
    w1 |= (u32_t)(sm_sbox[(s1 >> 8) & 0xFF]) << 8;
    w1 |= (u32_t)(sm_sbox[s1 & 0xFF]);
    w1 ^= const_m_ke[ROUNDS * 4];                   // AddRoundKey
    w1 = ( (w1>>24)&0xFF)|(( w1>>8)&0xFF00)|( (w1<<8)&0xFF0000) |( (w1<<24)&0xFF000000) ;

    dev_output[index] = w1 ; //store the cipher text
}   