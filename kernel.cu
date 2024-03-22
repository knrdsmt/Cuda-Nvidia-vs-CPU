#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define WIDTH 20000
#define HEIGHT 20000
#define BLOCK_SIZE 16

void randomImageCPU(float* image) {
    srand(time(NULL));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        image[i] = (float)rand() / RAND_MAX;
    }
}

__global__ void bilinearInterpolationKernel(float* input, float* output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float fx = (float)x / WIDTH;
    float fy = (float)y / HEIGHT;

    int x1 = (int)fx;
    int y1 = (int)fy;

    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float dx = fx - x1;
    float dy = fy - y1;

    output[y * WIDTH + x] = input[y1 * WIDTH + x1] * (1 - dx) * (1 - dy) +
        input[y1 * WIDTH + x2] * dx * (1 - dy) +
        input[y2 * WIDTH + x1] * (1 - dx) * dy +
        input[y2 * WIDTH + x2] * dx * dy;
}

void bilinearInterpolationCPU(float* input, float* output) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float fx = (float)x / WIDTH;
            float fy = (float)y / HEIGHT;

            int x1 = (int)fx;
            int y1 = (int)fy;

            int x2 = x1 + 1;
            int y2 = y1 + 1;

            float dx = fx - x1;
            float dy = fy - y1;

            output[y * WIDTH + x] = input[y1 * WIDTH + x1] * (1 - dx) * (1 - dy) +
                input[y1 * WIDTH + x2] * dx * (1 - dy) +
                input[y2 * WIDTH + x1] * (1 - dx) * dy +
                input[y2 * WIDTH + x2] * dx * dy;
        }
    }
}

__global__ void gaussianBlurKernel(float* input, float* output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float sum = 0.0f;
    int count = 0;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            if (x + ox < 0 || x + ox >= WIDTH || y + oy < 0 || y + oy >= HEIGHT)
                continue;

            float value = input[(y + oy) * WIDTH + (x + ox)];
            sum += value;
            ++count;
        }
    }

    output[y * WIDTH + x] = sum / count;
}

void gaussianBlurCPU(float* input, float* output) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float sum = 0.0f;
            int count = 0;

            for (int oy = -1; oy <= 1; ++oy) {
                for (int ox = -1; ox <= 1; ++ox) {
                    if (x + ox < 0 || x + ox >= WIDTH || y + oy < 0 || y + oy >= HEIGHT)
                        continue;

                    float value = input[(y + oy) * WIDTH + (x + ox)];
                    sum += value;
                    ++count;
                }
            }

            output[y * WIDTH + x] = sum / count;
        }
    }
}

__global__ void sobelEdgeDetectionKernel(float* input, float* output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    float sumX = 0.0f;
    float sumY = 0.0f;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            if (x + ox < 0 || x + ox >= WIDTH || y + oy < 0 || y + oy >= HEIGHT)
                continue;

            float value = input[(y + oy) * WIDTH + (x + ox)];
            sumX += value * Gx[oy + 1][ox + 1];
            sumY += value * Gy[oy + 1][ox + 1];
        }
    }

    output[y * WIDTH + x] = sqrt(sumX * sumX + sumY * sumY);
}

void sobelEdgeDetectionCPU(float* input, float* output) {
    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float sumX = 0.0f;
            float sumY = 0.0f;

            for (int oy = -1; oy <= 1; ++oy) {
                for (int ox = -1; ox <= 1; ++ox) {
                    if (x + ox < 0 || x + ox >= WIDTH || y + oy < 0 || y + oy >= HEIGHT)
                        continue;

                    float value = input[(y + oy) * WIDTH + (x + ox)];
                    sumX += value * Gx[oy + 1][ox + 1];
                    sumY += value * Gy[oy + 1][ox + 1];
                }
            }

            output[y * WIDTH + x] = sqrt(sumX * sumX + sumY * sumY);
        }
    }
}

int main() {
    float* h_image = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    float* h_output = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    float* d_image, * d_output;

    cudaMalloc((void**)&d_image, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&d_output, WIDTH * HEIGHT * sizeof(float));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(WIDTH / threads.x, HEIGHT / threads.y);

    printf("Generating Image\n\n");

    randomImageCPU(h_image);
    cudaMemcpy(d_image, h_image, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    ////// bilinear
    printf("Performing Bilinear Interpolation on GPU\n");

    clock_t start = clock();

    bilinearInterpolationKernel << <blocks, threads >> > (d_image, d_output);

    cudaDeviceSynchronize();

    clock_t end = clock();
    double time_spent_gpu = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken for Bilinear Interpolation on GPU: %f seconds\n", time_spent_gpu);

    printf("Performing Bilinear Interpolation on CPU\n");

    start = clock();

    bilinearInterpolationCPU(h_image, h_output);

    end = clock();
    double time_spent_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken for Bilinear Interpolation on CPU: %f seconds\n", time_spent_cpu);
    if (time_spent_gpu < time_spent_cpu) {
        printf("\nBilinear Interpolation executed by GPU is %f times faster than CPU\n\n", time_spent_cpu / time_spent_gpu);
    }

    ///// gaussian

    printf("Performing Gaussian Blur on GPU\n");

    start = clock();

    gaussianBlurKernel << <blocks, threads >> > (d_image, d_output);

    cudaDeviceSynchronize();

    end = clock();
    double time_spent_gpu_blur = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken for Gaussian Blur on GPU: %f seconds\n", time_spent_gpu_blur);

    printf("Performing Gaussian Blur on CPU\n");

    start = clock();

    gaussianBlurCPU(h_image, h_output);

    end = clock();
    double time_spent_cpu_blur = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken for Gaussian Blur on CPU: %f seconds\n", time_spent_cpu_blur);
    if (time_spent_gpu_blur < time_spent_cpu_blur) {
        printf("\nGaussian Blur executed by GPU is %f times faster than CPU\n", time_spent_cpu_blur / time_spent_gpu_blur);
    }
    ////// sobel
    printf("Performing Sobel Edge Detection on GPU\n");

    start = clock();

    sobelEdgeDetectionKernel << <blocks, threads >> > (d_image, d_output);

    cudaDeviceSynchronize();

    end = clock();
    double time_spent_gpu_sobel = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken for Sobel Edge Detection on GPU: %f seconds\n", time_spent_gpu_sobel);

    printf("Performing Sobel Edge Detection on CPU\n");

    start = clock();

    sobelEdgeDetectionCPU(h_image, h_output);

    end = clock();
    double time_spent_cpu_sobel = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken for Sobel Edge Detection on CPU: %f seconds\n", time_spent_cpu_sobel);
    if (time_spent_gpu_sobel < time_spent_cpu_sobel) {
        printf("\nSobel Edge Detection executed by GPU is %f times faster than CPU\n", time_spent_cpu_sobel / time_spent_gpu_sobel);
    }
    ///////

    free(h_image);
    free(h_output);
    cudaFree(d_image);
    cudaFree(d_output);
    return 0;
}
