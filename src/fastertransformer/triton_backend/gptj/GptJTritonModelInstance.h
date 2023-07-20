/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/gptj/GptJ.h"
#include "src/fastertransformer/triton_backend/gptj/GptJTritonModel.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include <memory>

namespace ft = fastertransformer;

template<typename T>
struct GptJTritonModelInstance: AbstractTransformerModelInstance {

    GptJTritonModelInstance(std::unique_ptr<ft::GptJ<T>>                            gpt,
                            std::shared_ptr<ft::GptJWeight<T>>                      gpt_weight,
                            std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                            std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map,
                            std::unique_ptr<std::mutex>                             cublas_wrapper_mutex,
                            std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper,
                            std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr);
    ~GptJTritonModelInstance();

    std::shared_ptr<std::vector<triton::Tensor>>
    forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors) override;

    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors) override;

    static std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors);

private:
    const std::unique_ptr<ft::GptJ<T>>                            gpt_;
    const std::shared_ptr<ft::GptJWeight<T>>                      gpt_weight_;
    const std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;
    const std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map_;
    const std::unique_ptr<std::mutex>                             cublas_wrapper_mutex_;
    const std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper_;
    const std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr_;

    std::unordered_map<std::string, ft::Tensor>
    convert_inputs(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors);

    void allocateBuffer(const size_t request_batch_size,
                        const size_t beam_width,
                        const size_t total_output_len,
                        const size_t max_request_output_len);

    void freeBuffer();

    int*   d_input_ids_                = nullptr;
    int*   d_input_lengths_            = nullptr;
    int*   d_input_bad_words_          = nullptr;
    int*   d_input_stop_words_         = nullptr;
    int*   d_request_prompt_lengths_   = nullptr;
    T*     d_request_prompt_embedding_ = nullptr;
    float* d_top_p_decay_              = nullptr;
    float* d_top_p_min_                = nullptr;
    int*   d_top_p_reset_ids_          = nullptr;

    int*   d_output_ids_       = nullptr;
    int*   d_sequence_lengths_ = nullptr;
    float* d_output_log_probs_ = nullptr;
    float* d_cum_log_probs_    = nullptr;

    uint32_t*          h_total_output_lengths_ = nullptr;
    std::exception_ptr h_exception_            = nullptr;
};
// 这是一个模板类`GptJTritonModelInstance`的定义，继承自`AbstractTransformerModelInstance`。该模板类的目的是作为GPT模型的实例类，用于对输入数据进行前向推断。以下是各个成员的说明：

// 1. 构造函数：接收多个参数，并用于初始化类的成员变量。
//    - `gpt`: 一个指向`ft::GptJ<T>`类型的`unique_ptr`，用于存储GPT模型的实例。
//    - `gpt_weight`: 一个指向`ft::GptJWeight<T>`类型的`shared_ptr`，用于存储GPT模型的权重。
//    - `allocator`: 一个指向`ft::Allocator<ft::AllocatorType::CUDA>`类型的`unique_ptr`，用于管理CUDA内存分配。
//    - `cublas_algo_map`: 一个指向`ft::cublasAlgoMap`类型的`unique_ptr`，用于存储CUBLAS算法映射。
//    - `cublas_wrapper_mutex`: 一个指向`std::mutex`类型的`unique_ptr`，用于在CUBLAS操作中提供互斥锁。
//    - `cublas_wrapper`: 一个指向`ft::cublasMMWrapper`类型的`unique_ptr`，用于封装CUBLAS操作。
//    - `cuda_device_prop_ptr`: 一个指向`cudaDeviceProp`类型的`unique_ptr`，用于存储CUDA设备的属性信息。

// 2. 析构函数：用于在对象被销毁时释放内存资源。

// 3.`forward`函数（重载）：用于进行前向推断，输入是`triton::Tensor`类型的指针。
// 4.`convert_outputs`静态成员函数：将输出的`ft::Tensor`转换为`triton::Tensor`，以便与Triton Inference Server进行交互。
// 5.convert_inputs(): 一个私有成员函数，用于将triton::Tensor类型的输入转换为ft::Tensor类型。
// 6.allocateBuffer(): 一个私有成员函数，用于分配内存缓冲区。
// 7. freeBuffer(): 一个私有成员函数，用于释放内存缓冲区。
// 私有成员变量：
//    - `const std::unique_ptr<ft::GptJ<T>> gpt_;`: 指向`ft::GptJ<T>`类型的`unique_ptr`，用于存储GPT模型的实例。
//    - `const std::shared_ptr<ft::GptJWeight<T>> gpt_weight_;`: 指向`ft::GptJWeight<T>`类型的`shared_ptr`，用于存储GPT模型的权重。
//    - `const std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;`: 一个指向`ft::Allocator<ft::AllocatorType::CUDA>`类型的`unique_ptr`，用于管理CUDA内存分配。
//    - `const std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map_;`: 一个指向`ft::cublasAlgoMap`类型的`unique_ptr`，用于存储CUBLAS算法映射。
//    - `const std::unique_ptr<std::mutex> cublas_wrapper_mutex_;`: 一个指向`std::mutex`类型的`unique_ptr`，用于在CUBLAS操作中提供互斥锁。
//    - `const std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper_;`: 一个指向`ft::cublasMMWrapper`类型的`unique_ptr`，用于封装CUBLAS操作。
//    - `const std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr_;`: 一个指向`cudaDeviceProp`类型的`unique_ptr`，用于存储CUDA设备的属性信息。
//    - `int* d_input_ids_ = nullptr;`: 一个指向`int`类型的指针，用于存储输入数据的ID。
//    - `int* d_input_lengths_ = nullptr;`: 一个指向`int`类型的指针，用于存储输入数据的长度信息。
//    - `int* d_input_bad_words_ = nullptr;`: 一个指向`int`类型的指针，用于存储输入数据中的无效单词信息。
//    - `int* d_input_stop_words_ = nullptr;`: 一个指向`int`类型的指针，用于存储输入数据中的停用词信息。
//    - `int* d_request_prompt_lengths_ = nullptr;`: 一个指向`int`类型的指针，用于存储请求的prompt长度信息。
//    - `T* d_request_prompt_embedding_ = nullptr;`: 一个指向模板类型`T`的指针，用于存储请求的prompt嵌入信息。
//    - `float* d_top_p_decay_ = nullptr;`: 一个指向`float`类型的指针，用于存储top-p采样的衰减信息。
//    - `float* d_top_p_min_ = nullptr;`: 一个指向`float`类型的指针，用于存储top-p采样的最小概率信息。
//    - `int* d_top_p_reset_ids_ = nullptr;`: 一个指向`int`类型的指针，用于存储top-p采样的重置ID信息。
//    - `int* d_output_ids_ = nullptr;`: 一个指向`int`类型的指针，用于存储输出数据的ID。
//    - `int* d_sequence_lengths_ = nullptr;`: 一个指向`int`类型的指针，用于存储输出数据的序列长度信息。
//    - `float* d_output_log_probs_ = nullptr;`: 一个指向`float`类型的指针，用于存储输出数据的对数概率信息。
//    - `float* d_cum_log_probs_ = nullptr;`: 一个指向`float`类型的指针，用于存储输出数据的累积对数概率信息。
//    - `uint32_t* h_total_output_lengths_ = nullptr;`: 一个指向`uint32_t`类型的指针，用于存储输出数据的总长度信息。
//    - `std::exception_ptr h_exception_ = nullptr;`: 一个指向`std::exception_ptr`类型的指针，用于存储异常信息的指针。

// 该模板类的目的是对GPT模型的实例进行封装，提供对输入数据进行前向推断的功能，并将输出结果转换为Triton Inference Server可接受的格式。