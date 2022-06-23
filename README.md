# GEMM using OpenCL

Learning OpenCL : trying a matrix computation using OpenCL
I hope this code will help you as it helped me to understand some of OpenCL
withoud having support form OpenACC, OpenMP (GPU target) and a programmable GPU.

## My config

Number of platforms                               1
  Platform Name                                   Intel Gen OCL Driver
  Platform Vendor                                 Intel
  Platform Version                                OpenCL 2.0 beignet 1.3

[...]
Number of devices                                 1
  Device Name                                     Intel(R) HD Graphics Skylake ULT GT1
  Device Vendor                                   Intel
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.0 beignet 1.3
  Driver Version                                  1.3
  Device OpenCL C Version                         OpenCL C 2.0 beignet 1.3
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Max compute units                               12
  Max clock frequency                             1000MHz
  Device Partition                                (core)
    Max number of sub-devices                     1
    Supported partition types                     None, None, None
  Max work item dimensions                        3
  Max work item sizes                             512x512x512
  Max work group size                             512
  Preferred work group size multiple              16
[...]
  Queue properties (on host)                      
    Out-of-order execution                        No
    Profiling                                     Yes
  Queue properties (on device)                    
    Out-of-order execution                        Yes
    Profiling                                     Yes
    Preferred size                                16384 (16KiB)
    Max size                                      262144 (256KiB)
  Max queues on device                            1
  Max events on device                            1024

## Work done

We propose here an implementation of a matrix multiplication CUDA kernel.
We compute : 

$$C = AB$$

Where : 

- $$A_{i,j} = 1 $$
- $$B_{i,j} = 1 $$
- $A, B and C$ are square matrix of order N

The result is necessarly $C_{i,j} = N$

## Usage

### Required

To compile and launch this program, make sure you installed :

-  gcc
-  OpenCL (minimum 1.2)
-  Drivers :
   -  BEIGNET for Intel GPU
   -  AMDGPU for AMD GPU
   -  Supports from nVidia form nVidia GPU

### Compilation

A GNU Makefile is given in this little project, just type "make" to build
program

### Execution

To execute this program, you can specify the matrix length (or try the default size : N = 8)


