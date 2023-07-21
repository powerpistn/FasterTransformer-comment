/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <assert.h>

#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"

namespace fastertransformer {

class BaseLayer {
public:
    BaseLayer(cudaStream_t     stream,
              cublasMMWrapper* cublas_wrapper,
              IAllocator*      allocator,
              bool             is_free_buffer_after_forward,
              cudaDeviceProp*  cuda_device_prop = nullptr,
              bool             sparse           = false):
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        allocator_(allocator),
        cuda_device_prop_(cuda_device_prop),
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        sparse_(sparse){};
    virtual ~BaseLayer() = default;

    virtual cudaStream_t getStream()
    {
        return stream_;
    }

    virtual void setStream(cudaStream_t stream)
    {
        stream_ = stream;
    }

protected:
    virtual void allocateBuffer() = 0;
    virtual void freeBuffer()     = 0;

    // device environments
    cudaStream_t     stream_;// 保存 CUDA 流的句柄，用于指定 GPU 执行操作的上下文
    cublasMMWrapper* cublas_wrapper_; // cublasMMWrapper 类的指针，用于执行矩阵乘法操作。
    IAllocator*      allocator_; // IAllocator 类的指针，用于在 GPU 上分配和释放内存。
    cudaDeviceProp*  cuda_device_prop_ = nullptr;//cudaDeviceProp 结构的指针，用于保存 GPU 设备的属性信息。

    bool is_free_buffer_after_forward_;// 表示在前向推断结束后是否释放缓冲区内存。
    bool is_allocate_buffer_ = false;  // 未来被弃用（deprecated），用于表示是否已经分配缓冲区内存。
    bool sparse_;//表示是否使用稀疏计算。
};

}  // namespace fastertransformer

// BaseLayer 在 FasterTransformer 库中的作用包括：

// 提供 CUDA 流（cudaStream_t）：每个 GPU 设备都有一个或多个 CUDA 流，用于指定 GPU 执行操作的上下文。BaseLayer 通过成员变量 stream_ 保存 CUDA 流的句柄，并提供了接口函数 getStream() 和 setStream()，允许其他层类在 GPU 上指定执行操作的流。

// 提供 cuBLAS 封装（cublasMMWrapper）：cuBLAS 是 NVIDIA 提供的用于执行矩阵乘法和其他线性代数运算的库。BaseLayer 通过成员变量 cublas_wrapper_ 保存 cuBLAS 封装类的指针，可以在其他层类中直接调用该封装类的函数来执行矩阵乘法操作。

// 提供内存管理接口（IAllocator）：GPU 上的内存管理是一个重要的任务，用于在 GPU 上分配和释放内存。BaseLayer 通过成员变量 allocator_ 保存内存分配器类的指针，允许其他层类使用该内存分配器在 GPU 上分配和释放内存。

// GPU 设备属性信息：BaseLayer 可以通过成员变量 cuda_device_prop_ 保存 GPU 设备的属性信息，供其他层类查询和使用。

// 管理缓冲区内存：BaseLayer 中的纯虚函数 allocateBuffer() 和 freeBuffer() 定义了缓冲区内存的分配和释放操作。这些函数需要在具体的层类中实现，用于管理不同层之间的临时缓冲区内存