# Session 08 Devlog — CUDA Streams

## Asynchronous execution
- Kernel launches are async — CPU fires kernel and immediately moves on
- GPU runs in background while CPU continues
- cudaMemcpy DeviceToHost implicitly synchronizes — forces CPU to wait for GPU

## What is a stream
- A sequence of GPU operations that execute in order
- Default: everything runs in one stream — kernels and memory copies are sequential
- Multiple streams: overlap compute and memory transfer

## Default vs multi-stream
Default (one stream) — sequential:
  [memcpy H→D] → [kernel] → [memcpy D→H]

Two streams — overlapped:
  Stream 1: [memcpy H→D] → [kernel] → [memcpy D→H]
  Stream 2:               [memcpy H→D] → [kernel] → [memcpy D→H]
  Stream 2 starts while Stream 1 kernel is still running

## Why it matters
- 10-40% latency reduction for fixed-size inference workloads
- vLLM uses CUDA streams heavily for pipelined inference
- GPU is never idle waiting for memory transfers when work is available

## Pinned (page-locked) memory
- Normal CPU memory can be paged — OS moves it to disk and changes its physical address
- DMA (Direct Memory Access) — hardware engine that copies memory without CPU involvement
- DMA needs a fixed physical address to transfer from CPU to GPU in background
- Pinned memory = tell OS "don't move this, lock it at a fixed address forever"
- Required for cudaMemcpyAsync — without it async transfers can't work
- cudaMallocHost = allocate pinned memory, cudaFreeHost = free it

## Paging
- RAM is limited — OS fakes more memory by swapping unused pages to disk
- Moving pages changes their physical address
- Pinned memory opts out of this — stays fixed in physical RAM

## DMA
- Hardware engine that copies memory directly, CPU doesn't have to be involved
- GPU uses DMA to transfer CPU RAM → VRAM in background
- CPU is free to do other work during the transfer

## cudaMallocHost vs new
- new = regular pageable CPU memory, can be paged out
- cudaMallocHost = pinned CPU memory, locked in RAM, required for async transfers
- Use cudaMallocHost only when you need async transfers — pinned memory is limited
- Use new for everything else

## cudaMemcpy vs cudaMemcpyAsync
- cudaMemcpy → blocks CPU until copy is done, CPU waits
- cudaMemcpyAsync → CPU fires copy and moves on, DMA handles it in background
- Using cudaMemcpy with streams destroys the overlap — CPU blocks on each copy
- cudaMemcpyAsync is what makes stream overlap actually happen

## Kernel launch with stream
- <<<grid, block>>> → default stream, no overlap
- <<<grid, block, 0, stream1>>> → specific stream
- 3rd arg = dynamic shared memory bytes (0 = none)
- 4th arg = stream — must specify 3rd to get to 4th

## API
cudaStream_t stream;
cudaStreamCreate(&stream);
kernel<<<grid, block, 0, stream>>>(...);          // launch in specific stream
cudaMemcpyAsync(dst, src, bytes, dir, stream);    // async copy in stream
cudaStreamSynchronize(stream);                    // wait for stream to finish
cudaStreamDestroy(stream);                        // clean up
