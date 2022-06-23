#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* gpu */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE				(128)
#define MAX_SOURCE_SIZE		(0x100000)

/* Device info */
/* Use clinfo command or read /sys/class/misc/mali0/device/gpuinfo file */
#define NUM_ENTRIES					1

void init_vec(double *vec, double val, int len)
{
    for (int i = 0; i < len; i++)
    {
			vec[i] = val;
	}
}

void print_vec(double *vec, int len)
{
	for ( int j = 0; j < len; j++)
	{
		for (int i = 0; i < len; i++)
		{
			printf("%lf ", vec[i]);
		}
		printf("\n");
	}
}

int main ( int argc, char ** argc)
{

	int n_len = (argc == 1 ) ? 8 : atoi ( argv[1]);
	int len = n_len * n_len;
	double * a = (double *) malloc ( len * sizeof ( double));
	double * b = (double *) malloc ( len * sizeof ( double));
	double * c = (double *) malloc ( len * sizeof ( double));
	
	init_vec ( a, 1.0, len);
	init_vec ( b, 1.0, len);
	init_vec ( c, 0.0, len);
	
	
	cl_mem a_buff = NULL;
	cl_mem b_buff = NULL;
	cl_mem c_buff = NULL;
	
	cl_platform_id platform_id = NULL;
	cl_uint n_platforms;

	cl_device_id device_id = NULL;
	cl_uint n_devices;	

	cl_context context = NULL;
	cl_kernel kernel = NULL;
	cl_program program = NULL;
	
	cl_command_queue command_queue = NULL;
	cl_int ret;
	
	
	/* Load the source code containing the kernel */
	FILE * fp = NULL;
	char fileName[] = "./gemm.cl";
	char * source_str;
	size_t source_size;
	
	fp = fopen ( fileName, "r");
	if ( !fp)
	{
		fprintf ( stderr, "Failed to lead kernel.\n");
		exit(1);
	}
	
	source_str = (char *) malloc ( MAX_SOURCE_SIZE);
	source_size = fread ( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose ( fp);
	
	// Platform
	ret = clGetPlatformIDs ( NUM_ENTRIES, &platform_id, &n_platforms);
	if ( ret != CL_SUCCESS)
	{
		printf ( "Failed to get platform ID.\n");
		goto error;
	}
	// Device
	ret = clGetDeviceIDs ( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if ( ret != CL_SUCCESS)
	{
		printf ( "Failed to get device ID.\n");
		goto error; 
	}
	// Context
	context = clCreateContext ( NULL, 1, &device_id, NULL, NULL, &ret);
	if ( ret != CL_SUCCESS)
	{
		printf ("Failed to create OpenCL context/\n");
		goto error;
	}
	command_queue = clCreateCommandQueue ( context, device_id, 0, &ret);
	if ( ret != CL_SUCCESS)
	{
		printf ( "Failed to create command queue %d\n", (int) ret);
		goto error;
	}
	// Memory Buffer
	a_buff = clCreateBuffer ( context, CL_MEM_READ_ONLY, len * sizeof ( double), NULL, &ret);
	b_buff = clCreateBuffer ( context, CL_MEM_READ_ONLY, len * sizeof ( double), NULL, &ret);
	c_buff = clCreateBuffer ( context, CL_MEM_WRITE_ONLY, len * sizeof ( double), NULL, &ret);

	ret = clEnqueueWriteBuffer ( command_queue, a_buff, CL_TRUE, 0, len * sizeof ( double), (void *) a, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer ( command_queue, b_buff, CL_TRUE, 0, len * sizeof ( double), (void *) b, 0, NULL, NULL);
	
	// Create Kernel Program from source
	program = clCreateProgramWithSource ( context, 1, (const char **) &source_str, ( const size_t *) &source_size, &ret);
	if ( ret != CL_SUCCESS)
	{
		printf ("Failed to create OpenCL program from source %d\n", (int) ret);
		goto error;
	}
	
	// Build Kernel Program
	ret = clBuildProgram ( program, 1, &device_id, NULL, NULL, NULL);
	if ( ret != CL_SUCCESS)
	{
		printf ("Failed to build program %d\n", (int) ret);
		char build_log[16348];
		clGetProgramBuildInfo ( program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
		printf ("Error in kernel: %s\n", build_log);
		goto error;
	}
	
	// Create OpenCL Kernel
	kernel = clCreateKernel ( program, "dgemm_gpu", &ret);
	if (ret != CL_SUCCESS) {
		printf("Failed to create kernel %d\n", (int) ret);
		goto error; 
	}
	ret = clSetKernelArg ( kernel, 0, sizeof(cl_mem), (void *) &a_buff);
	ret = clSetKernelArg ( kernel, 1, sizeof(cl_mem), (void *) &b_buff);
	ret = clSetKernelArg ( kernel, 2, sizeof(cl_mem), (void *) &c_buff);
	ret |= clSetKernelArg ( kernel, 3, sizeof (cl_int), (void *) &n_len);
	if (ret != CL_SUCCESS) {
		printf("Failed to set kernel arguments %d\n", (int) ret);
		goto error;
	}
	
	// Executing OpenCL Kernel
	
	size_t local_work_size[2] = { 16, 16 };
	size_t global_work_size[2] = { 64, 64 };
		
	ret = clEnqueueNDRangeKernel ( command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	if ( ret != CL_SUCCESS)
	{
		printf("Failed to execute kernel for execution %d\n", (int) ret);
		goto error;		
	}
	
	// MemCpy Device To Host
	ret = clEnqueueReadBuffer ( command_queue, c_buff, CL_TRUE, 0, len*sizeof ( double), (void *) c, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to copy data from device to host %d\n", (int) ret);
		goto error;
	}
	
	printf ( "Results : \n");
	print_vec ( c, n_len);
	/* Finalization */
error:

    /* free device resources */
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	clReleaseMemObject(a_buff);
	clReleaseMemObject(b_buff);
	clReleaseMemObject(c_buff);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

    /* free host resources */
	free(source_str);
	free(a);
	free(b);
	free(c);
	return 0;
}
