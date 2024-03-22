# CUDA Accelerated Image Processing

CUDA, which stands for Compute Unified Device Architecture, is a parallel computing platform and programming model developed by NVIDIA. It allows developers to harness the power of NVIDIA GPUs for general-purpose processing, including complex computational tasks like image processing. In this example, CUDA is utilized to accelerate various image processing algorithms compared to their CPU counterparts.


## Results Analysis

1. **Bilinear Interpolation**:
   - GPU Execution Time: 0.378 seconds
   - CPU Execution Time: 8.272 seconds
   - GPU Speedup Over CPU: 21.88 times

   Bilinear Interpolation involves interpolating between pixels in an image. Leveraging parallel processing on the GeForce MX450 graphics card, this operation is significantly faster compared to execution on the Intel Core i7-1165G7 processor.

2. **Gaussian Blur**:
   - GPU Execution Time: 0.739 seconds
   - CPU Execution Time: 13.094 seconds
   - GPU Speedup Over CPU: 17.72 times

   Gaussian Blur is computationally more intensive than bilinear interpolation, and it also achieves substantial speedup thanks to CUDA utilization on the graphics card.

3. **Sobel Edge Detection**:
   - GPU Execution Time: 1.771 seconds
   - CPU Execution Time: 20.822 seconds
   - GPU Speedup Over CPU: 11.76 times

   Sobel Edge Detection involves calculating edge gradients for each pixel. Utilizing the GPU, this operation is nearly twelve times faster than on the CPU.

All three operations, Bilinear Interpolation, Gaussian Blur, and Sobel Edge Detection, demonstrate significant acceleration through the use of CUDA technology on the NVIDIA GeForce MX450 graphics card. This underscores the potential of GPU utilization for image processing tasks, particularly with large images and complex algorithms.

<p float="left">
<img src="https://github.com/knrdsmt/Cuda-Nvidia/blob/main/results.png?raw=true" alt="results" width="55%" height="auto" />
</p>

### Benefits of Using CUDA

1. **Parallel Processing**: CUDA enables parallel execution of computations on GPU cores, significantly speeding up processing tasks compared to traditional sequential execution on CPUs. This parallelism is particularly advantageous for image processing tasks, which often involve large datasets and numerous calculations per pixel.

2. **High Performance**: GPUs are optimized for parallel processing and contain thousands of cores, providing significantly higher computational throughput compared to CPUs. As a result, CUDA-accelerated algorithms can achieve substantial speedups, making them ideal for real-time or large-scale image processing applications.

3. **Optimized Libraries**: NVIDIA provides optimized libraries and tools for common computational tasks, such as linear algebra, signal processing, and image processing. These libraries leverage CUDA to unlock the full potential of GPU hardware, enabling developers to achieve high-performance implementations with minimal effort.

4. **Memory Bandwidth**: GPUs feature high memory bandwidth, allowing for rapid data transfer between the CPU and GPU. This is essential for image processing tasks, where large amounts of image data need to be efficiently transferred and processed.

### How CUDA Works in This Program

1. **Memory Management**: The program allocates memory on both the CPU and GPU using `cudaMalloc` for input and output images. This allows data to be transferred between the CPU and GPU memory spaces efficiently.

2. **Kernel Execution**: Image processing algorithms such as bilinear interpolation, Gaussian blur, and Sobel edge detection are implemented as CUDA kernels. These kernels are executed in parallel by numerous threads organized into blocks and grids on the GPU. Each thread processes a distinct portion of the input image, enabling high-throughput computation.

3. **Synchronization**: The program uses `cudaDeviceSynchronize` to ensure that all CUDA kernels have completed execution before proceeding. This synchronization step is crucial for accurate timing measurements and preventing data race conditions.

4. **Performance Analysis**: The program measures the execution time of CUDA-accelerated algorithms and their CPU counterparts using `clock()` to compare their performance. This analysis demonstrates the significant speedup achieved by utilizing CUDA for image processing tasks.

### Program Operation

1. **Generating Image**: The program starts by generating a random image of size 20000x20000 pixels using the `randomImageCPU` function. This image serves as the input data for subsequent processing stages.

2. **Bilinear Interpolation**: In the first processing stage, the image undergoes bilinear interpolation. Both the CPU and GPU implementations calculate new pixel values using interpolation between neighboring pixel values. The GPU version accelerates this process by utilizing parallel processing of multiple pixels across thousands of threads on the graphics card.

3. **Gaussian Blur**: Next, the image undergoes Gaussian blur to smooth it out. Both the CPU and GPU implementations calculate a new pixel value by averaging the values of neighboring pixels. Due to parallel processing on the GPU, this operation is performed much faster compared to the CPU.

4. **Sobel Edge Detection**: The final processing stage involves Sobel edge detection. Both CPU and GPU implementations compute the edge gradient for each pixel using Sobel masks. The GPU implementation achieves higher performance thanks to parallel processing of multiple pixels simultaneously.

5. **Performance Analysis**: The program measures the execution time of each processing stage for both CPU and GPU versions. Comparing the execution times allows users to understand the significant performance advantage of CUDA implementation over traditional CPU processing, providing insights into the benefits of utilizing CUDA for image processing tasks.
