# Session P1 Devlog — C++ Essentials

## Tools
- g++ = GNU C++ compiler (GNU → GCC → g++ for C++)
- Installed via xcode-select --install on Mac
- stdio.h = standard C input/output library, built into the compiler, gives us printf

## Compile and run command
```
g++ -O0 -o cpu_basics cpu_basics.cpp && ./cpu_basics
```
- -O0: no optimizations (code runs exactly as written, good for learning)
- -o cpu_basics: name the output binary
- && ./cpu_basics: run it if compile succeeded

---


## Concepts covered

### RAM, Memory, Address
- RAM is a long strip of slots, each holding 1 byte, each with a unique address
- Address = the number label on a slot (e.g. 0x7fff5a)
- Memory and RAM are the same thing
- When you write x = 42, the value gets stored at some address in RAM

### Variables and Types
- A variable reserves space at some address and gives it a name
- The TYPE determines how many bytes are reserved — not the value
  - int   → always 4 bytes (32 bits), range up to ~2.1 billion
  - float → always 4 bytes
  - double → always 8 bytes
- All 32 bits are used simultaneously — small numbers just have leading zeros
- Python hides this (dynamic sizing). C++ is explicit — fixed sizes, known at compile time

### Pointers
- A pointer is a variable whose value is an address (not a regular value)
- It "points to" where some other data lives in memory
- The address of a variable = the first byte of its reserved slots

### Why pointers exist
- For a single variable, pointers are overkill — you'd just use the variable
- They exist for two real reasons:
  1. Functions need to modify the original data — passing a pointer gives the function
     the actual address so it can write directly. Without it, the function gets a copy.
  2. Large data (e.g. a 1GB matrix) can't be copied on every function call.
     You pass 8 bytes (a pointer) instead of copying gigabytes.
- In CUDA: cudaMalloc gives you back a pointer to GPU memory.
  Every kernel receives that pointer and reads/writes directly into GPU RAM.
  The entire CUDA model is: allocate → get pointer → pass pointer to kernel.

### Pointer syntax
- int* p     →  p is a pointer to an int (declaration)
- &x         →  gives you the address of x
- *p         →  go to the address p holds, get the value there (dereference)
- *p = 100   →  go to that address and write 100 there (modifies the original)
- x and *p refer to the exact same bytes in RAM — no copy, same memory

### Stack vs Heap
- Stack: automatic, managed for you, gone when function returns, small (~8MB)
- Heap: manual, you control lifetime, lives until you free it, as large as RAM allows
- Memory leak: allocating on heap but never freeing it — address is lost, memory stuck
- In CUDA: cudaMalloc allocates on GPU heap, cudaFree releases it
  GPU VRAM is precious — leaking it causes out-of-memory crashes

### 2D arrays as flat 1D memory
- RAM is a straight line — no such thing as 2D in hardware
- Matrices are stored row by row (row-major): row 0 first, then row 1, etc.
- To find element [row][col]: index = row * num_cols + col
- In CUDA: you pass a pointer to a flat array, kernel uses this formula to find any element
- There is no magic 2D indexing — just pointer + arithmetic

### Heap allocation
- new = allocate on the heap, returns a pointer to that space
- delete[] = free the heap memory (equivalent of cudaFree)
- arr = nullptr after freeing — so you don't accidentally use a freed pointer
- arr[i] is shorthand for "go to address arr, move i slots forward, read/write there"
- arr + i is the actual memory address of element i
- int array: addresses increment by 4 bytes each time (1 int = 4 bytes)
- This is exactly how CUDA kernels navigate GPU memory — same pointer arithmetic

### Compilation flow
- cpu_basics.cpp → (g++ compiles) → cpu_basics → (you run it)
- The .cpp file is source code — human readable
- The output file (cpu_basics) is the executable — what the computer actually runs
- -o cpu_basics tells g++ what to name the output file
- Output file is generated after compiling, not before

### printf format specifiers
- %d → integer
- %p → memory address (pointer)
- \n → newline

### main()
- Every C++ program must have a main() — it's where execution starts
- return 0 means "program finished successfully" — signal to the OS
- 0 = success, anything else = error

### 2D indexing on flat 1D array
- Memory is always flat — there is no 2D in hardware
- A 3x3 matrix is just 9 slots in a row: 0 1 2 10 11 12 20 21 22
- To access element [row][col]: index = row * num_cols + col
- You write this arithmetic yourself inside the CUDA kernel — it's not automatic
- You pass a flat pointer to the kernel, the kernel uses r * cols + c to find any element
- %.0f in printf = print float with 0 decimal places

### Key insight
We never pass a matrix to a GPU kernel — we pass a pointer (the address of where
the matrix lives in GPU memory). The kernel receives that pointer and uses the
index formula to read/write any element directly.
