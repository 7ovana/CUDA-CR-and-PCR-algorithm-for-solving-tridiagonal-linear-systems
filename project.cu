/* CUDA Project: Solving a tridiagonal system on GPUs
 
a: lower diagonal
b: diagonal
c: upper diagonal
y: A*x
x: solution of the system, x = inv(A)*y

*/

#include <stdio.h>
# include <assert.h>
#define NTPB 8

__host__ void thomas(float *a, float *b, float *c, float *y, float *x, int n){
    
    /// ------------------ forward elimination ------------------------ ///
    c[0] /= b[0];
    y[0] /= b[0];

    for (int i = 1; i < n; i++){
        float tmp = b[i] - a[i]*c[i-1];
        if (i < n-1) c[i] /= tmp;
        y[i] = (y[i] - a[i]*y[i-1]) / tmp;
    }

    /// ------------------ backward substitution ------------------------ ///
    x[n-1] = y[n-1];
    for (int i = n-2; i>=0; i--) x[i] = float(y[i] - c[i] * x[i+1]);
}

__global__ void CR(float *a, float *b, float *c, float *y, float *x, int n){

    float k1, k2;
    __shared__ float a_s[NTPB];
    __shared__ float b_s[NTPB];
    __shared__ float c_s[NTPB];
    __shared__ float y_s[NTPB];

    int stride = 2;
    int i = stride*(threadIdx.x + 1) - 1;  // from 1 to n-1

    /// ------------------------ forward elimination ---------------------------- ///
    
    k1 = a[i] / b[i-1]; 
    k2 = 0.0;

    a_s[threadIdx.x] = - a[i-1]*k1; // a[0] is already 0, so no problem there

    if (i < n-1){
        k2 = c[i] / b[i+1];
        b_s[threadIdx.x] = b[i] - c[i-1]*k1 - a[i+1]*k2;
        c_s[threadIdx.x] = - c[i+1]*k2;
        y_s[threadIdx.x] = y[i] - y[i-1]*k1 - y[i+1]*k2;
    }
    else{ // last equation
        b_s[threadIdx.x] = b[i] - c[i-1]*k1;
        c_s[threadIdx.x] = 0.0;
        y_s[threadIdx.x] = y[i] - y[i-1]*k1;
    }
    __syncthreads();

    while (stride < n/2){
        i = stride*(threadIdx.x + 1) - 1;
 
        if (threadIdx.x < NTPB/stride){           
            int delta = stride/2;
            k1 = a_s[i] / b_s[i-delta];
            a_s[i] = - a_s[i-delta]*k1;

            if (threadIdx.x < NTPB/stride - 1){
                k2 = c_s[i] / b_s[i+delta];
                b_s[i] -= c_s[i-delta]*k1 + a_s[i+delta]*k2;
                c_s[i] = -c_s[i+delta]*k2;
                y_s[i] -= y_s[i-delta]*k1 + y_s[i+delta]*k2;
            }
            else{ // last equation
                b_s[i] -= c_s[i-delta]*k1;
                c_s[i] = 0.0;
                y_s[i] -= y_s[i-delta]*k1;
            }
        }
        stride *= 2;
    }
    __syncthreads();

    /// ------------- log2(n)-th step: solving a 2 unknowns system --------------- ///
    if (threadIdx.x == 0){                  
            x[n-1] = (y_s[n/4 - 1] - y_s[n/2 - 1]*b_s[n/4 - 1] / a_s[n/2 - 1]) / (c_s[n/4 - 1] - b_s[n/2-1]*b_s[n/4 - 1]/a_s[n/2 - 1]);
            x[n/2-1] = (y_s[n/4 - 1] - c_s[n/4 - 1]*x[n-1]) / b_s[n/4 - 1];
    }

    /// ------------------------ backward substitution --------------------------- ///
    stride /= 2;
    // other even unknowns
    while (NTPB/stride < NTPB){
        i = stride*(2*threadIdx.x + 1) - 1;
        if (threadIdx.x < NTPB/stride){
            if (threadIdx.x == 0){ // first unknown
                x[i] = (y_s[(i-1)/2] - c_s[(i-1)/2]*x[i + stride]) / b_s[(i-1)/2];
            }
            else{
                x[i]=(y_s[(i-1)/2] - a_s[(i-1)/2]*x[i - stride] - c_s[(i-1)/2]*x[i+stride])/b_s[(i-1)/2];
            }
        }
        stride /= 2;
    }
    __syncthreads();

    // odd unknowns
    if(threadIdx.x < NTPB){
        i = 2*threadIdx.x;
        if(i==0){ // first unknown
            x[0]=(y[0]-c[0]*x[1])/b[0];
        }
        else{ 
            x[i]=(y[i]-c[i]*x[i+1]-a[i]*x[i-1])/b[i];
        }
    }
}

__global__ void PCR(float *a, float *b, float *c, float *y, float *x, int n){

    float k1, k2;
    __shared__ float a_s[NTPB];
    __shared__ float b_s[NTPB];
    __shared__ float c_s[NTPB];
    __shared__ float y_s[NTPB];

    int stride = 2;
    int i = stride*threadIdx.x + blockIdx.x;  // from 0 to NTPB_PCR-2, by 2 for the first block, // from 1 to NTPB_PCR-1, by 2 for the second block
                                              
    if (i == 0){ // first equation
        //k1 = 0.0;
        k2 = c[i] / b[i+1];
        a_s[threadIdx.x] = 0.0;
        b_s[threadIdx.x] = b[i] - a[i+1]*k2;
        c_s[threadIdx.x]=-c[i+1]*k2;
		y_s[threadIdx.x]=y[i]-y[i+1]*k2;
    }
    else if (i == n-1){ // last equation
        k1 = a[i] / b[i-1];
        //k2 = 0.0;
        a_s[threadIdx.x] = -a[i-1]*k1;
        b_s[threadIdx.x] = b[i] - c[i-1]*k1;
        c_s[threadIdx.x] = 0.0;
        y_s[threadIdx.x] = y[i] - y[i-1]*k1;
    }
    else{
        k1 = a[i] / b[i-1];
        k2 = c[i] / b[i+1];
        a_s[threadIdx.x] = -a[i-1]*k1;
        b_s[threadIdx.x] = b[i] - c[i-1]*k1 - a[i+1]*k2;
        c_s[threadIdx.x] = - c[i+1]*k2;
        y_s[threadIdx.x] = y[i] - y[i-1]*k1 - y[i+1]*k2;
    }
    __syncthreads();

    float a_tmp, b_tmp, c_tmp, y_tmp;
    i = threadIdx.x;

    while(stride <= n/2){
        int delta = stride/2;
        if(i-delta < 0){ // first equation
            k2 = c_s[i] / b_s[i+delta];
			a_tmp = 0.0;
			b_tmp = b_s[i] - a_s[i+delta]*k2;
			c_tmp = -c_s[i+delta]*k2;
			y_tmp = y_s[i]-y_s[i+delta]*k2;
        }
        else if(i+delta>n-1){ // last equation
			k1 = a_s[i] / b_s[i-delta];
			a_tmp = -a_s[i-delta]*k1;
			b_tmp = b_s[i] - c_s[i-delta]*k1;
			c_tmp = 0.0;
			y_tmp = y_s[i] - y_s[i-delta]*k1;
		}
		else{
			k1 = a_s[i] / b_s[i-delta];
			k2 = c_s[i] / b_s[i+delta];
			a_tmp = -a_s[i-delta]*k1;
			b_tmp = b_s[i] - a_s[i+delta]*k2 - c_s[i-delta]*k1;
			c_tmp = -c_s[i+delta]*k2;
			y_tmp = y_s[i] - y_s[i+delta]*k2 - y_s[i-delta]*k1;
        }
        __syncthreads(); // wait for all threads to finish, then assign
        a_s[i] = a_tmp;
        b_s[i] = b_tmp;
        c_s[i] = c_tmp;
        y_s[i] = y_tmp;
        __syncthreads(); // in order to update vectors for all threads

        stride *= 2;
    }
    // solve for all x
    x[2*threadIdx.x+blockIdx.x] = y_s[threadIdx.x] / b_s[threadIdx.x];
}

int main(void){

    float *a, *b, *c, *y, *x;
    float *a_gpu, *b_gpu, *c_gpu, *y_gpu, *x_gpu;
    int n = 16;
    printf("n = %d\n", n);

    a = (float*) malloc(n*sizeof(float));
    b = (float*) malloc(n*sizeof(float));
    c = (float*) malloc(n*sizeof(float));
    y = (float*) malloc(n*sizeof(float));
    x = (float*) malloc(n*sizeof(float));

    cudaMalloc(&a_gpu, n*sizeof(float));
    cudaMalloc(&b_gpu, n*sizeof(float));
    cudaMalloc(&c_gpu, n*sizeof(float));
    cudaMalloc(&y_gpu, n*sizeof(float));
    cudaMalloc(&x_gpu, n*sizeof(float));
    
    // Laplace operator and y = all ones
    a[0] = 0.; b[0] = 2.; c[0] = -1.;  y[0] = 1.;
    a[n-1] = -1.; c[n-1] = 0; b[n-1] = 2.; y[n-1] = 1.;
    for (int i=0; i<n-1; i++){
        a[i] = -1.; b[i] = 2.; c[i] = -1.; y[i] = 1.;
    }

    float temps;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(a_gpu, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu, c, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    CR<<<1,NTPB>>>(a_gpu, b_gpu, c_gpu, y_gpu, x_gpu, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temps, start, stop);

    cudaMemcpy(x, x_gpu, n*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Resultat avec CR:\n");
    for (int i=0; i<n; i++) printf("%.5f\n", x[i]);
    printf("CR: Time elapsed on GPU: %f ms\n", temps);

    cudaEventRecord(start, 0);
    PCR<<<2,NTPB>>>(a_gpu, b_gpu, c_gpu, y_gpu, x_gpu, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temps, start, stop);

    cudaMemcpy(x, x_gpu, n*sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nResultat avec PCR:\n");
    for (int i=0; i<n; i++) printf("%.5f\n", x[i]);
    printf("PCR: Time elapsed on GPU: %f ms\n", temps);

    cudaEventRecord(start, 0);
    thomas(a, b, c, y, x, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temps, start, stop);

    printf("\nResultat avec Thomas:\n");
    for (int i=0; i<n; i++) printf("%.5f\n", x[i]);
    printf("Thomas: Time elapsed on CPU: %f ms\n", temps);

    free(a); free(b); free(c); free(y); free(x);
    cudaFree(a_gpu); 
    cudaFree(b_gpu); 
    cudaFree(c_gpu);
    cudaFree(y_gpu);
    cudaFree(x_gpu);
}