// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

void cpu_image_convolution(float *hData, float *hOutputData, int kernalDim, float *kernal, int height, int width, float factor){
    int kernalX = kernalDim/2;
    int kernalY = kernalDim/2;
    float sum = 0;
    int kernalval = 0;
    int ii = 0;
    int jj = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            kernalval = 0;
            sum = 0;
            for(int x = -kernalX; x <= kernalX; ++x){
                for(int y = -kernalY; y <= kernalY; ++y){
                    ii = i + x;
                    jj = j + y;
                    if((ii >= 0 && ii < height) && (jj >= 0 && jj < width)){
                        sum += (hData[ii*height + jj] * (factor * kernal[kernalval]));
                    }                    
                    kernalval++;
                }   
            }
            hOutputData[i*height + j] = sum;
        }
    }
}

__global__ void gpu_image_convolution(float *dData, float *dOutputData, int kernalDim, float *kernal, int height, int width, float factor){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int kernalX = kernalDim/2;
    int kernalY = kernalDim/2;
    float sum = 0;
    int kernalval = 0;
    int ii = 0;
    int jj = 0;
    if(i < height && j < width){
        for(int x = -kernalX; x <= kernalX; ++x){
            for(int y = -kernalY; y <= kernalY; ++y){
                ii = i + x;
                jj = j + y;
                if((ii >= 0 && ii < height) && (jj >= 0 && jj < width)){
                    sum += (dData[ii*height + jj] * (factor * kernal[kernalval]));
                }                    
                kernalval++;
            }   
        }
        dOutputData[i*height + j] = sum;
    }
}

__constant__ float const_kernal[1024];
__global__ void gpu_image_convolution_shared(float *dData, float *dOutputData, int kernalDim, int height, int width, float factor){
	__shared__ float sharedMemory[16][16];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int kernalX = kernalDim/2;
    int kernalY = kernalDim/2;
    float sum = 0;
    int kernalval = 0;
    int ii = 0;
    int jj = 0;
    if(i < height && j < width){       
        sharedMemory[threadIdx.x][threadIdx.y] = dData[i*height + j];
    	__syncthreads();

        for(int x = -kernalX; x <= kernalX; ++x){
            for(int y = -kernalY; y <= kernalY; ++y){
                ii = threadIdx.x + x;
                jj = threadIdx.y + y;
                if((ii >= 0 && ii < 16) && (jj >= 0 && jj < 16)){
                    sum += (sharedMemory[ii][jj] * (factor * const_kernal[kernalval]));
                }else{
                    int iii = ii - threadIdx.x + i;
                    int jjj = jj - threadIdx.y + j;
                    if(iii >= 0 && iii < height && jjj >=0 && jjj < width){
                        sum += (dData[iii*height + jjj] * (factor * const_kernal[kernalval]));
                    }
                }                    
                kernalval++;
            }   
        }
        dOutputData[i*height + j] = sum;
    }
}

int main(int argc, char **argv){
    int kernalChoice;
    char inputFile[1000];
    char outputFile[1000];
    float factor = 1;
    // program kernalChoice, filename.pgm, outputname.pgm, factor
    if(argc != 5){
        printf("Usage error: imgConv kernalChoice inputfile.pgm outputfile.pgm factor");
        exit(EXIT_FAILURE);
    }else{
        kernalChoice = atoi(argv[1]);
        strcpy(inputFile, argv[2]);
        strcpy(outputFile, argv[3]);
        factor = atof(argv[4]);
    }
    
    const char *imageName = inputFile;
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageName, "/data");

    if(imagePath == NULL){
        printf("Unable to find image: %s\n", imageName);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);
    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageName, width, height);

    float sharpen[] = {0.0,-1.0,0.0,
                         -1.0,5.0,-1.0,
                          0.0,-1.0,0.0}; // Sharpen

    float emboss_three[] = {-2.0,-1.0,0.0,
                     -1.0,1.0,1.0,
                      0.0,1.0,2.0}; // Emboss 3x3AS

    float emboss_five[] = {1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,-1}; // Emboss
    float average_five[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}; // Average

    float *kernal[4] = {sharpen, emboss_three, emboss_five, average_five};

    int kernalDim = 0;
    if(kernalChoice == 0 || kernalChoice == 1){
        kernalDim = 3;
    }else if(kernalChoice == 2 || kernalChoice == 3){
        kernalDim = 5;
    }else{
        printf("Invalid kernal choice!\n");
        exit(EXIT_FAILURE);
    }

    float *hOutputCPU = (float *) malloc(sizeof(float) * size);
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);    
    cpu_image_convolution(hData, hOutputCPU, kernalDim, kernal[kernalChoice],height, width, factor);
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms) \n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    float timeSerial = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    char outputFilename[1024];
    const char *outPath = "./output/cpu_";
    strcpy(outputFilename, outPath);
    strcpy(outputFilename + strlen(outPath), outputFile);
    sdkSavePGM(outputFilename, hOutputCPU, width, height);
    printf("Wrote '%s'\n", outputFilename);

    dim3 blockSize(16, 16);
    dim3 gridSize((height + blockSize.y - 1) / blockSize.y, (width + blockSize.x - 1) / blockSize.x);
    float *dData = NULL;
    float *dOutputData = NULL;
    float *dKernal = NULL;
    float *hOutputData = (float *) malloc(size);
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    checkCudaErrors(cudaMalloc((void **) &dOutputData, size));
    checkCudaErrors(cudaMalloc((void **) &dKernal, (kernalDim*kernalDim*sizeof(float))));

    /**
     * Global Memory GPU
     * 
     */
    checkCudaErrors(cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dKernal, kernal[kernalChoice], (kernalDim*kernalDim * sizeof(float)), cudaMemcpyHostToDevice));
    
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);

    cudaEventRecord(launch_begin,0);
    gpu_image_convolution<<<gridSize, blockSize>>>(dData, dOutputData, kernalDim, dKernal, height, width, factor);
    cudaEventRecord(launch_end,0);
    cudaDeviceSynchronize();

    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);
    checkCudaErrors(cudaMemcpy(hOutputData, dOutputData, size, cudaMemcpyDeviceToHost));
    printf("GPU Time: %f ms\n", time);
    printf("Speedup: %f\n", timeSerial/time);
    printf("%.2f Mpixels/sec\n",
           (width *height / (time / 1000.0f)) / 1e6);
    outPath = "./output/gpu_global_";
    strcpy(outputFilename, outPath);
    strcpy(outputFilename + strlen(outPath), outputFile);
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);


    /**
     * Shared and constant memory GPU
     * 
     */
    checkCudaErrors(cudaMemcpyToSymbol(const_kernal, kernal[kernalChoice],(kernalDim*kernalDim * sizeof(float))));

    cudaEventRecord(launch_begin,0);
    gpu_image_convolution_shared<<<gridSize, blockSize>>>(dData, dOutputData, kernalDim, height, width, factor);
    cudaEventRecord(launch_end,0);
    cudaDeviceSynchronize();

    time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);
    checkCudaErrors(cudaMemcpy(hOutputData, dOutputData, size, cudaMemcpyDeviceToHost));
    printf("GPU Time: %f ms\n", time);
    printf("Speedup: %f\n", timeSerial/time);
    printf("%.2f Mpixels/sec\n",
           (width *height / (time / 1000.0f)) / 1e6);
    outPath = "./output/gpu_shared_";
    strcpy(outputFilename, outPath);
    strcpy(outputFilename + strlen(outPath), outputFile);
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);

    free(hOutputCPU);
    checkCudaErrors(cudaFree((void *) dData));
    checkCudaErrors(cudaFree((void *) dOutputData));
    checkCudaErrors(cudaFree((void *) dKernal));
    free(hOutputData);

    return 0;
}