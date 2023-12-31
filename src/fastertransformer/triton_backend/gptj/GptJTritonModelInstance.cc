/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/triton_backend/gptj/GptJTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace ft = fastertransformer;

template<typename T>
void triton_stream_callback(std::unordered_map<std::string, ft::Tensor>* output_tensors, void* ctx)
{
    GptJTritonModelInstance<T>* model  = reinterpret_cast<GptJTritonModelInstance<T>*>(ctx);
    auto                        result = GptJTritonModelInstance<T>::convert_outputs(*output_tensors);

    model->stream_cb_(result, model->stream_ctx_);
}

template<typename T>
GptJTritonModelInstance<T>::GptJTritonModelInstance(std::unique_ptr<ft::GptJ<T>>                            gpt,
                                                    std::shared_ptr<ft::GptJWeight<T>>                      gpt_weight,
                                                    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                                                    std::unique_ptr<ft::cublasAlgoMap>   cublas_algo_map,
                                                    std::unique_ptr<std::mutex>          cublas_wrapper_mutex,
                                                    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
                                                    std::unique_ptr<cudaDeviceProp>      cuda_device_prop_ptr):
    gpt_(std::move(gpt)),
    gpt_weight_(gpt_weight),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

// 该函数的主要功能是将输入数据input_tensors转换为ft::Tensor类型的输入数据，并将其中的部分数据从主机（Host）内存移动到设备（Device）内存。
// 它还计算了输出数据的总长度，并存储在h_total_output_lengths_中。最终，函数返回转换后的ft::Tensor类型的输入数据ft_input_tensors。
// 这个函数是GptJTritonModelInstance类中用于前向推理的函数，将输入数据准备好后，可以将其传递给GptJ模型进行处理。
template<typename T>
std::unordered_map<std::string, ft::Tensor> GptJTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    // 1.调试日志输出语句，用于在调试时打印当前函数的名称。
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // 2.将triton::Tensor类型的输入数据input_ids从主机（Host）内存移动到设备（Device）内存，存储在d_input_ids_中。&allocator_是ft::Allocator类型的指针，用于在设备内存中分配内存。
    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_, &allocator_);
    // 3.将triton::Tensor类型的输入数据input_lengths从主机（Host）内存移动到设备（Device）内存，存储在d_input_lengths_中。&allocator_是ft::Allocator类型的指针，用于在设备内存中分配内存。
    move_tensor_H2D(input_tensors->at("input_lengths"), d_input_lengths_, &allocator_);

    // 4.获取输入数据input_ids的批量大小。
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    // 5.获取输入数据input_ids的数据长度。
    const size_t input_data_len     = input_tensors->at("input_ids").shape[1];
    // 6.分配大小为request_batch_size的uint32_t类型的主机内存，用于存储输出数据的总长度信息。
    h_total_output_lengths_         = reinterpret_cast<uint32_t*>(malloc(request_batch_size * sizeof(uint32_t)));
    // 7.对于每个批次的数据，计算输出数据的总长度，并存储在h_total_output_lengths_中。
    for (int i = 0; i < request_batch_size; ++i) {
        h_total_output_lengths_[i] =
            reinterpret_cast<const uint32_t*>(input_tensors->at("request_output_len").data)[i] + input_data_len;
    }
    // 8.初始化一个std::unordered_map对象ft_input_tensors，用于存储转换后的ft::Tensor类型的输入数据。
    std::unordered_map<std::string, ft::Tensor> ft_input_tensors = std::unordered_map<std::string, ft::Tensor>{
        {"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
        {"input_lengths", as_GPU_tensor(input_tensors->at("input_lengths"), d_input_lengths_)},
        {"output_seq_len",
         ft::Tensor{ft::MEMORY_CPU,
                    ft::TYPE_UINT32,
                    {input_tensors->at("request_output_len").shape[0]},
                    h_total_output_lengths_}}};
    // 9.如果输入数据中包含名为"bad_words_list"的数据，则将其从主机内存移动到设备内存，并存储在d_input_bad_words_中。
    if (input_tensors->find("bad_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("bad_words_list"), d_input_bad_words_, &allocator_);
        ft_input_tensors.insert(
            {"bad_words_list", as_GPU_tensor(input_tensors->at("bad_words_list"), d_input_bad_words_)});
    }
    // 10.如果输入数据中包含名为"stop_words_list"的数据，则将其从主机内存移动到设备内存，并存储在d_input_stop_words_中。
    if (input_tensors->find("stop_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("stop_words_list"), d_input_stop_words_, &allocator_);
        ft_input_tensors.insert(
            {"stop_words_list", as_GPU_tensor(input_tensors->at("stop_words_list"), d_input_stop_words_)});
    }
    // 11.如果输入数据中同时包含名为"request_prompt_embedding"、"request_prompt_lengths"和"request_prompt_type"的数据，则将它们从主机内存移动到设备内存，并分别存储在d_request_prompt_embedding_和d_request_prompt_lengths_中。
    if (input_tensors->count("request_prompt_embedding") && input_tensors->count("request_prompt_lengths")
        && input_tensors->count("request_prompt_type")) {

        move_tensor_H2D(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_lengths",
             as_GPU_tensor(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_)});

        move_tensor_H2D(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_embedding",
             as_GPU_tensor(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_)});
    }
    // 12.如果输入数据中包含名为"top_p_decay"的数据，则将其从主机内存移动到设备内存，并存储在d_top_p_decay_中。
    if (input_tensors->find("top_p_decay") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_decay"), d_top_p_decay_, &allocator_);
        ft_input_tensors.insert({"top_p_decay", as_GPU_tensor(input_tensors->at("top_p_decay"), d_top_p_decay_)});
    }
    // 13.如果输入数据中包含名为"top_p_min"的数据，则将其从主机内存移动到设备内存，并存储在d_top_p_min_中。
    if (input_tensors->find("top_p_min") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_min"), d_top_p_min_, &allocator_);
        ft_input_tensors.insert({"top_p_min", as_GPU_tensor(input_tensors->at("top_p_min"), d_top_p_min_)});
    }
    // 14.如果输入数据中包含名为"top_p_reset_ids"的数据，则将其从主机内存移动到设备内存，并存储在d_top_p_reset_ids_中。
    if (input_tensors->find("top_p_reset_ids") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_, &allocator_);
        ft_input_tensors.insert(
            {"top_p_reset_ids", as_GPU_tensor(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_)});
    }
    // 15.遍历输入数据的所有元素，并将不是"input_ids"、"input_lengths"、"output_seq_len"、"prefix_soft_prompt_embedding"和"prefix_soft_prompt_lengths"的数据转换为ft::Tensor类型，并存储在ft_input_tensors中。
    for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
        if (t->first.find("input_ids") == std::string::npos && t->first.find("input_lengths") == std::string::npos
            && t->first.find("output_seq_len") == std::string::npos
            && t->first.find("prefix_soft_prompt_embedding") == std::string::npos
            && t->first.find("prefix_soft_prompt_lengths") == std::string::npos) {
            if (ft_input_tensors.count(t->first) == 0) {
                ft_input_tensors.insert({t->first, t->second.convertTritonTensorToFt()});
            }
        }
    }
    // 16.返回转换后的ft::Tensor类型的输入数据。
    return ft_input_tensors;
}

// 该函数的主要功能是将输入的output_tensors参数（类型为std::unordered_map<std::string, ft::Tensor>）转换为输出类型为std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>的智能指针。
// 在转换过程中，它遍历output_tensors中的每个元素，将每个ft::Tensor对象转换为对应的triton::Tensor对象，并将它们插入到新创建的std::unordered_map<std::string, triton::Tensor>中。最后，将该std::unordered_map指针包装成一个std::shared_ptr智能指针，并返回该智能指针作为函数的输出。
template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
GptJTritonModelInstance<T>::convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors)
{   
    // 1.调试信息
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // 2.创建一个指向unordered_map的指针outputs_mapping，并使用new运算符在堆上分配内存来创建一个空的unordered_map。
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();
    // 3.对于输入参数output_tensors中的每个元素，进行遍历操作。
    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        // 3.1 将遍历到的output_tensors中的元素转换为triton::Tensor类型，并插入到outputs_mapping中。
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }
    // 4.将outputs_mapping包装成shared_ptr智能指针，并作为函数的返回值。由于使用了shared_ptr智能指针，这样做可以在函数结束时自动管理内存的释放，避免内存泄漏。
    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
GptJTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
GptJTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    // 1.日志输出语句，用于在调试模式下记录当前函数的名称。
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // 2.带有信息的断言宏。它检查input_tensors中名为"input_ids"的triton::Tensor对象的形状是否是2维，并在断言失败时输出指定的信息。
    FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape.size() == 2,
                       "input_tensors->at(\"input_ids\").shape.size() == 2");
    // 3.带有信息的断言宏。它检查input_tensors中名为"input_lengths"的triton::Tensor对象的形状是否是1维，并在断言失败时输出指定的信息。                   
    FT_CHECK_WITH_INFO(input_tensors->at("input_lengths").shape.size() == 1,
                       "input_tensors->at(\"input_lengths\").shape.size() == 1");
    
    // 4.获取请求的批处理大小。
    const uint32_t request_batch_size     = input_tensors->at("input_ids").shape[0];
    // 5.获取请求中输出序列的最大长度。
    const uint32_t max_request_output_len = (size_t)*std::max_element(
        (int*)input_tensors->at("request_output_len").data,
        (int*)input_tensors->at("request_output_len").data + input_tensors->at("request_output_len").shape[0]);
    // 6.计算输出序列的总长度。
    const uint32_t total_output_len = max_request_output_len + input_tensors->at("input_ids").shape[1];
    // 7.获取束搜索的宽度。
    const uint32_t beam_width =
        input_tensors->count("beam_width") ? (size_t)(*(uint*)input_tensors->at("beam_width").data) : 1;
    // 8.分配用于推理的缓冲区。
    allocateBuffer(request_batch_size, beam_width, total_output_len, max_request_output_len);
    // 9.将输入input_tensors转换为FastTransformers库中的ft::Tensor格式。
    std::unordered_map<std::string, ft::Tensor> ft_input_tensors = convert_inputs(input_tensors);
    // 10.创建用于存储输出的ft::Tensor的std::unordered_map。
    std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
        {"output_ids",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_UINT32,
                    std::vector<size_t>{request_batch_size, beam_width, total_output_len},
                    d_output_ids_}},
        {"sequence_length",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_INT32,
                    std::vector<size_t>{request_batch_size, beam_width},
                    d_sequence_lengths_}}};
    // 11.创建不同输出结果的ft::Tensor对象，并将它们插入到output_tensors中。
    if (input_tensors->count("is_return_log_probs") && *((bool*)input_tensors->at("is_return_log_probs").data)) {
        output_tensors.insert({"output_log_probs",
                               ft::Tensor{ft::MEMORY_GPU,
                                          ft::TYPE_FP32,
                                          std::vector<size_t>{request_batch_size, beam_width, max_request_output_len},
                                          d_output_log_probs_}});
        output_tensors.insert({"cum_log_probs",
                               ft::Tensor{ft::MEMORY_GPU,
                                          ft::TYPE_FP32,
                                          std::vector<size_t>{request_batch_size, beam_width},
                                          d_cum_log_probs_}});
    }
    // 12.执行forward函数
    try {
        if (stream_cb_ != nullptr) {
            gpt_->registerCallback(triton_stream_callback<T>, this);
        }
        // 调用gpt_对象的forward函数，传入输出张量output_tensors、输入张量ft_input_tensors以及权重gpt_weight_.get()。
        gpt_->forward(&output_tensors, &ft_input_tensors, gpt_weight_.get());

        if (stream_cb_ != nullptr) {
            gpt_->unRegisterCallback();
        }
    }
    catch (...) {
        h_exception_ = std::current_exception();
        output_tensors.insert({"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
    }
    // 13.释放先前分配的输出序列长度缓冲区。
    if (h_total_output_lengths_ != nullptr) {
        free(h_total_output_lengths_);
        h_total_output_lengths_ = nullptr;
    }
    // 14.将ft::Tensor格式的输出结果转换为triton::Tensor格式，并返回包含输出结果的std::unordered_map的智能指针
    return convert_outputs(output_tensors);
}

template<typename T>
GptJTritonModelInstance<T>::~GptJTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void GptJTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                                const size_t beam_width,
                                                const size_t total_output_len,
                                                const size_t max_request_output_len)
{
    d_output_ids_ = (int*)(allocator_->reMalloc(
        d_output_ids_, sizeof(int) * request_batch_size * beam_width * total_output_len, false));
    d_sequence_lengths_ =
        (int*)(allocator_->reMalloc(d_sequence_lengths_, sizeof(int) * request_batch_size * beam_width, false));
    d_output_log_probs_ = (float*)(allocator_->reMalloc(
        d_output_log_probs_, sizeof(float) * request_batch_size * beam_width * max_request_output_len, false));
    d_cum_log_probs_ =
        (float*)(allocator_->reMalloc(d_cum_log_probs_, sizeof(float) * request_batch_size * beam_width, false));
}

template<typename T>
void GptJTritonModelInstance<T>::freeBuffer()
{
    allocator_->free((void**)(&d_output_ids_));
    allocator_->free((void**)(&d_sequence_lengths_));
    allocator_->free((void**)(&d_output_log_probs_));
    allocator_->free((void**)(&d_cum_log_probs_));
}

template struct GptJTritonModelInstance<float>;
template struct GptJTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct GptJTritonModelInstance<__nv_bfloat16>;
#endif