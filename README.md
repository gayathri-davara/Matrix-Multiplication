
````markdown
# Matrix Multiplication – Sequential, OpenMP, MPI and CUDA

## Objective

To implement and compare matrix multiplication using four different computing models:

1. Sequential CPU execution
2. OpenMP shared-memory parallelism
3. MPI distributed-memory parallelism
4. CUDA GPU parallelism

The same matrix multiplication problem is used for all implementations.

---

## Problem Definition

- Matrix A: 4000 × 4000
- Matrix B: 4000 × 4000
- All elements of A and B are initialized to `1.0`
- Operation: `C = A × B`

Since every element is 1.0 and each output element is calculated using 4000 multiplication terms:

```text
C[i][j] = 4000.00
````

Verification:

```text
C[0][0] = 4000.00
```

---

## Experiment Structure

```text
Experiment-1-Matrix-Multiplication/
│
├── matrix_sequential.c
├── matrix_sequential.jpeg
│
├── matrix_openmp.c
├── matrix_openmp.jpeg
│
├── matrix_mpi.c
├── matrix_cuda.cu
│
└── README.md
```

---

# 1. Sequential Matrix Multiplication

The sequential implementation performs matrix multiplication using a single CPU execution flow.

### Source File

```text
matrix_sequential.c
```

### Compilation

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
```

### Execution

```bash
./matrix_sequential
```

### Result

```text
Matrix Size = 4000 x 4000
Execution Time = 300.472835 seconds
Verification C[0][0] = 4000.00
```

### Recorded Time

**300.472835 seconds**

---

# 2. OpenMP Matrix Multiplication

OpenMP is used to parallelize the matrix multiplication across multiple CPU threads.

### Source File

```text
matrix_openmp.c
```

### Threads Used

```text
16 threads
```

### Compilation

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

### Execution

```bash
./matrix_openmp
```

### Result

```text
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 16
Execution Time = 117.790996 seconds
Verification C[0][0] = 4000.00
```

### Recorded Time

**117.790996 seconds**

---

# 3. MPI Distributed Matrix Multiplication

MPI is used to distribute the matrix multiplication across multiple MPI processes.

The implementation uses:

* `MPI_Scatter()` to distribute rows of matrix A
* `MPI_Bcast()` to distribute matrix B
* Local matrix multiplication on each MPI process
* `MPI_Gather()` to collect the partial results

### Source File

```text
matrix_mpi.c
```

### Matrix Size

```text
4000 × 4000
```

### MPI Processes

```text
4 processes
```

### Compilation

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

### Execution

```bash
mpirun -np 4 ./matrix_mpi
```

### Result

```text
MPI distribution complete.
Starting 4000x4000 multiplication...

Matrix multiplication completed.
C[0][0] = 4000.00
Execution time = 210.911815 seconds
```

### Recorded Time

**210.911815 seconds**

---

# 4. CUDA Matrix Multiplication

CUDA is used to perform matrix multiplication on an NVIDIA GPU.

### Source File

```text
matrix_cuda.cu
```

### GPU

```text
NVIDIA GeForce RTX 5060 Ti
```

### CUDA Toolkit

```text
CUDA 13.4
```

### Matrix Size

```text
4000 × 4000
```

### Block Size

```text
16 × 16 threads
```

### Grid Size

```text
250 × 250 blocks
```

### Compilation

```powershell
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

### Execution

```powershell
matrix_cuda.exe
```

### Result

```text
Matrix Size: 4000 x 4000
Block Size: 16 x 16
Grid Size: 250 x 250
C[0][0] = 4000.00
Kernel time: 0.211245 seconds
```

### Recorded Kernel Time

**0.211245 seconds**

> Note: The CUDA value recorded here is the GPU kernel execution time. It should not be directly treated as the total end-to-end CUDA execution time.

---

# 5. Performance Results

The measured execution times from the experiments are:

| Implementation | Configuration                 | Execution Time |
| -------------- | ----------------------------- | -------------: |
| Sequential     | Single CPU execution          |   300.472835 s |
| OpenMP         | 16 CPU threads                |   117.790996 s |
| MPI            | 4 MPI processes               |   210.911815 s |
| CUDA           | RTX 5060 Ti, kernel execution |     0.211245 s |

All implementations produced the expected verification value:

```text
C[0][0] = 4000.00
```

---

## Speedup

Speedup can be calculated using:

```text
Speedup = Sequential Execution Time / Parallel Execution Time
```

For OpenMP:

```text
300.472835 / 117.790996 ≈ 2.55×
```

For MPI:

```text
300.472835 / 210.911815 ≈ 1.42×
```

For CUDA kernel time:

```text
300.472835 / 0.211245 ≈ 1422.87×
```

### Important Note

The implementations were executed in different environments/configurations. Therefore, these values demonstrate the observed execution times rather than a controlled hardware benchmark.

The CUDA measurement is specifically the **kernel execution time**, while the CPU and MPI measurements represent their respective measured execution intervals.

---

# 6. Technologies Used

* C
* GCC
* OpenMP
* MPI / Open MPI
* CUDA
* NVIDIA `nvcc`
* Linux / WSL
* Windows PowerShell
* VMware virtual machines for MPI

---

# 7. Learning Outcomes

This experiment demonstrates four approaches to parallel and high-performance computing:

### Sequential

Provides the baseline implementation and execution time.

### OpenMP

Demonstrates shared-memory parallelism using multiple CPU threads.

### MPI

Demonstrates distributed-memory parallelism using multiple processes and explicit communication.

### CUDA

Demonstrates GPU-based parallel computation using CUDA threads, blocks and grids.

---

# 8. Conclusion

The same 4000 × 4000 matrix multiplication problem was implemented using sequential execution, OpenMP, MPI and CUDA.

The experiment demonstrates how the same computational problem can be approached using:

```text
Sequential CPU
      ↓
OpenMP Shared Memory
      ↓
MPI Distributed Memory
      ↓
CUDA GPU Parallelism
```

The output verification remained consistent across the implementations:

```text
C[0][0] = 4000.00
```

The experiment provides practical understanding of CPU-based parallelism, distributed-memory computing and GPU acceleration.

````
---

# 9. Execution Screenshots

## Sequential Matrix Multiplication

![Sequential Matrix Multiplication](matrix_sequential.jpeg)

## OpenMP Matrix Multiplication

![OpenMP Matrix Multiplication](matrix_openmp.jpeg)

## MPI Matrix Multiplication

![MPI Matrix Multiplication](matrix_mpi.jpeg)

## CUDA Matrix Multiplication

![CUDA Matrix Multiplication](matrix_cuda.jpeg)

