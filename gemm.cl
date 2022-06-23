__kernel void dgemm_gpu (__global const double *a, __global const double *b, __global double *c, const int N)
{
	uint dim = get_work_dim();
	int gid[3] = { 0, 0, 0},
		gsize[3]  = { 0, 0, 0};
	for ( int i = 0; i < dim; i++)
	{
		gid[i] = get_global_id(i);
		gsize[i]  = get_global_size(i);
	}
	int ii = gid[0]+gsize[0]*gid[1]+gsize[1]*gid[2];
	int ix = ii / N;
	int ij = ii % N;
	
	double acc = 0.0;
	for	( int i = 0; i < N; i++)
	{
		acc += a[ix * N + i] * b [ i + ij * N];
	}
	c[ii] = acc;
}
