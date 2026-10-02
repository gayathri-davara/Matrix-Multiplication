Matrix Multiplication – Sequential, OpenMP, MPI and CUDA
Implementation and performance analysis of 4000 × 4000 matrix multiplication using sequential CPU execution, OpenMP shared-memory parallelism, MPI distributed-memory parallelism, and CUDA GPU parallelism.
1. Objective
The objective of this experiment is to implement the same matrix multiplication problem using four different computing models and compare their observed execution performance:
1. Sequential CPU execution
2. OpenMP shared-memory parallelism
3. MPI distributed-memory parallelism
4. CUDA GPU parallelism
The same input size and expected output are used across the implementations.
2. Problem Statement
Two 4000 × 4000 matrices, A and B, are multiplied to produce matrix C.
Matrix Configuration
Parameter	Value
Matrix A	4000 × 4000
Matrix B	4000 × 4000
Matrix C	4000 × 4000
Initial value of A	1.0
Initial value of B	1.0
Expected value of C[i][j]	4000.00


Since every element of A and B is initialized to 1.0, each element of C is the sum of 4000 products:
C[i][j] = (1 × 1) + (1 × 1) + ... + (1 × 1)
          └──────────── 4000 terms ────────────┘

C[i][j] = 4000.00
The result is verified using:
C[0][0] = 4000.00
3. Computing Approaches
Implementation	Computing Model	Configuration
Sequential	Single CPU execution	Baseline
OpenMP	Shared-memory parallelism	16 threads
MPI	Distributed-memory parallelism	4 processes
CUDA	GPU parallelism	NVIDIA GeForce RTX 5060 Ti


4. Experiment Workflow
                    Matrix Multiplication
                            │
             ┌──────────────┼──────────────┐
             │              │              │
        Sequential       OpenMP           MPI
        CPU Baseline   Shared Memory   Distributed Memory
             │              │              │
             └──────────────┼──────────────┘
                            │
                          CUDA
                       GPU Parallelism
                            │
                            ▼
                  Performance Comparison
5. Project Structure
Matrix-Multiplication/
│
├── README.md
│
├── matrix_sequential.c
├── matrix_sequential.jpeg
│
├── matrix_openmp.c
├── matrix_openmp.jpeg
│
├── matrix_mpi.c
├── matrix_mpi.jpeg
│
├── matrix_cuda.c
├── matrix_cuda.jpeg
│
└── graphs/
    ├── execution_time_comparison.png
    ├── speedup_comparison.png
    └── cpu_mpi_execution_time.png
6. Sequential Matrix Multiplication
The sequential implementation performs the complete matrix multiplication using a single CPU execution flow. It provides the baseline execution time for comparison with the parallel implementations.
Source File
matrix_sequential.c
Compilation
gcc -O2 matrix_sequential.c -o matrix_sequential
Execution
./matrix_sequential
Result
Matrix Size = 4000 x 4000
Execution Time = 300.472835 seconds
Verification C[0][0] = 4000.00
Recorded Time
300.472835 seconds
7. OpenMP Matrix Multiplication
OpenMP is used to parallelize matrix multiplication within a shared-memory CPU environment.
Source File
matrix_openmp.c
Configuration
Number of OpenMP Threads = 16
Compilation
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
Execution
./matrix_openmp
Result
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 16
Execution Time = 117.790996 seconds
Verification C[0][0] = 4000.00
Recorded Time
117.790996 seconds
8. MPI Distributed Matrix Multiplication
MPI is used to distribute the matrix multiplication across multiple MPI processes.
The implementation uses:
- MPI_Scatter() to distribute rows of matrix A
- MPI_Bcast() to distribute matrix B
- Local matrix multiplication on each MPI process
- MPI_Gather() to collect partial results
Source File
matrix_mpi.c
Configuration
Matrix Size = 4000 × 4000
MPI Processes = 4
Compilation
mpicc -O2 matrix_mpi.c -o matrix_mpi
Execution
mpirun -np 4 ./matrix_mpi
Result
MPI distribution complete.
Starting 4000x4000 multiplication...

Matrix multiplication completed.
C[0][0] = 4000.00
Execution time = 210.911815 seconds
Recorded Time
210.911815 seconds
9. CUDA Matrix Multiplication
CUDA is used to perform matrix multiplication on an NVIDIA GPU.
Source File
matrix_cuda.c
The CUDA source in this repository uses the .c filename extension even though it contains CUDA code. The compilation command explicitly tells nvcc to treat the file as CUDA source.

GPU
NVIDIA GeForce RTX 5060 Ti
CUDA Toolkit
CUDA 13.4
Configuration
Matrix Size = 4000 × 4000
Block Size = 16 × 16 threads
Grid Size = 250 × 250 blocks
Compilation
nvcc -x cu -O2 matrix_cuda.c -o matrix_cuda
Execution
matrix_cuda.exe
Result
Matrix Size: 4000 x 4000
Block Size: 16 x 16
Grid Size: 250 x 250
C[0][0] = 4000.00
Kernel time: 0.211245 seconds
Recorded Kernel Time
0.211245 seconds
Important: The CUDA value reported here is the GPU kernel execution time. It is not the total end-to-end CUDA execution time, because host-to-device transfers, device-to-host transfers, allocation, initialization, and other overheads are not represented by this kernel-time value.

10. Performance Results
The observed measurements are:
Implementation	Configuration	Measured Time
Sequential	Single CPU execution	300.472835 s
OpenMP	16 CPU threads	117.790996 s
MPI	4 MPI processes	210.911815 s
CUDA	RTX 5060 Ti, kernel time	0.211245 s


All implementations produced the expected verification result:
C[0][0] = 4000.00
10.1 Execution Time Comparison
 
The graph compares the measured execution intervals for all four implementations.
Because the CUDA kernel measurement is much smaller than the CPU and MPI measurements, a separate CPU/MPI comparison is also included below.
10.2 CPU and MPI Execution Time
 
This graph provides a clearer comparison among the sequential, OpenMP, and MPI measurements without the CUDA kernel value dominating the scale.
11. Speedup Analysis
Speedup is calculated relative to the sequential execution time:
Speedup = Sequential Execution Time / Measured Parallel Time
OpenMP
300.472835 / 117.790996 ≈ 2.55×
MPI
300.472835 / 210.911815 ≈ 1.42×
CUDA Kernel Time
300.472835 / 0.211245 ≈ 1422.87×
11.1 Observed Speedup
 
Implementation	Observed Speedup
OpenMP	2.55×
MPI	1.42×
CUDA kernel	1422.87×


The CUDA speedup value is calculated using kernel execution time only. Therefore, it should not be interpreted as an end-to-end application speedup.

12. Performance Interpretation
The measurements show different characteristics for the four computing approaches.
Sequential
The sequential implementation provides the baseline against which the other measurements are compared.
OpenMP
The OpenMP implementation reduces the observed execution time by distributing the computation among 16 CPU threads.
MPI
The MPI implementation distributes the workload among four MPI processes. Its measured execution interval includes the characteristics and overheads of distributed execution.
CUDA
The CUDA kernel measurement is substantially smaller than the measured CPU and MPI execution intervals. However, this value represents kernel execution time only, so it should be interpreted separately from total application execution time.
13. Experimental Limitations
The measurements should be interpreted as observed experiment results, not as a controlled benchmark.
The implementations were executed in different environments and configurations. Differences in hardware, virtualization, CPU resources, memory, operating system environment, and execution overhead can affect the measured times.
In particular:
- Sequential and OpenMP use CPU execution.
- MPI uses multiple processes and distributed-memory communication.
- CUDA uses an NVIDIA GPU.
- The CUDA value reported here is kernel execution time only.
- Therefore, the four measured values are useful for demonstrating the behavior of different computing models but are not a strictly controlled apples-to-apples benchmark.
14. Technologies Used
- C
- GCC
- OpenMP
- MPI / Open MPI
- CUDA
- NVIDIA nvcc
- Linux / WSL
- Windows PowerShell
- VMware virtual machines
15. Learning Outcomes
This experiment provides practical understanding of different approaches to parallel and high-performance computing.
Sequential Computing
- Understand the baseline matrix multiplication algorithm.
- Establish a reference execution time.
OpenMP
- Understand shared-memory parallelism.
- Use multiple CPU threads for parallel computation.
- Compile and execute OpenMP programs using GCC.
MPI
- Understand distributed-memory parallelism.
- Use MPI processes for workload distribution.
- Understand basic MPI communication using scatter, broadcast and gather operations.
CUDA
- Understand GPU-based parallel computation.
- Understand CUDA threads, blocks and grids.
- Measure GPU kernel execution time.
Performance Analysis
- Calculate observed speedup.
- Compare different computing models.
- Understand the importance of execution environment and measurement methodology.
16. Conclusion
The same 4000 × 4000 matrix multiplication problem was implemented using four different computing models:
Sequential CPU
      ↓
OpenMP Shared Memory
      ↓
MPI Distributed Memory
      ↓
CUDA GPU Parallelism
All four implementations produced the expected verification result:
C[0][0] = 4000.00
The experiment demonstrates how a common computational problem can be implemented using sequential execution, shared-memory parallelism, distributed-memory parallelism, and GPU acceleration.
The measured results provide practical experience in implementation, execution, performance measurement, speedup calculation, and interpretation of parallel computing experiments.
17. Execution Screenshots
Sequential Matrix Multiplication
 
OpenMP Matrix Multiplication
 
MPI Matrix Multiplication
 
CUDA Matrix Multiplication
 
Repository Summary
Component	Details
Problem	4000 × 4000 Matrix Multiplication
Sequential	Single CPU execution
OpenMP	16 CPU threads
MPI	4 MPI processes
CUDA	RTX 5060 Ti
Verification	C[0][0] = 4000.00
Graphs	Execution time, CPU/MPI comparison, speedup
