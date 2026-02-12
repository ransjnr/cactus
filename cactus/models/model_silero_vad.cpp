#include "model.h"
#include "../graph/graph.h"
#include "../kernel/kernel.h"
#include <stdexcept>

namespace cactus {
namespace engine {

SileroVADModel::SileroVADModel() : Model() {
    weight_nodes_.encoder_blocks.resize(4);
}

SileroVADModel::SileroVADModel(const Config& config) : Model(config) {
    weight_nodes_.encoder_blocks.resize(4);
}

SileroVADModel::~SileroVADModel() = default;

void SileroVADModel::load_weights_to_graph(CactusGraph* gb) {
    weight_nodes_.stft_basis = gb->mmap_weights(model_folder_path_ + "/stft_basis.weights");

    for (uint32_t i = 0; i < 4; i++) {
        std::string block_prefix = model_folder_path_ + "/encoder_block_" + std::to_string(i) + "_";
        weight_nodes_.encoder_blocks[i].conv_weight = gb->mmap_weights(block_prefix + "conv_weight.weights");
        weight_nodes_.encoder_blocks[i].conv_bias = gb->mmap_weights(block_prefix + "conv_bias.weights");
    }

    weight_nodes_.lstm_weight_ih = gb->mmap_weights(model_folder_path_ + "/lstm_weight_ih.weights");
    weight_nodes_.lstm_weight_hh = gb->mmap_weights(model_folder_path_ + "/lstm_weight_hh.weights");
    weight_nodes_.lstm_bias_ih = gb->mmap_weights(model_folder_path_ + "/lstm_bias_ih.weights");
    weight_nodes_.lstm_bias_hh = gb->mmap_weights(model_folder_path_ + "/lstm_bias_hh.weights");

    weight_nodes_.output_conv_weight = gb->mmap_weights(model_folder_path_ + "/output_conv_weight.weights");
    weight_nodes_.output_conv_bias = gb->mmap_weights(model_folder_path_ + "/output_conv_bias.weights");
}

void SileroVADModel::build_graph() {
    const size_t input_size = CONTEXT_SIZE + CHUNK_SIZE + REFLECT_PAD_SIZE;

    graph_nodes_.input = graph_.input({1, 1, input_size}, Precision::FP16);
    graph_nodes_.h_prev = graph_.input({1, HIDDEN_SIZE}, Precision::FP16);
    graph_nodes_.c_prev = graph_.input({1, HIDDEN_SIZE}, Precision::FP16);

    auto stft_complex = graph_.conv1d(graph_nodes_.input, weight_nodes_.stft_basis, 128);
    auto real_parts = graph_.slice(stft_complex, 1, 0, 129);
    auto imag_parts = graph_.slice(stft_complex, 1, 129, 129);
    auto real_sq = graph_.multiply(real_parts, real_parts);
    auto imag_sq = graph_.multiply(imag_parts, imag_parts);
    auto magnitude_sq = graph_.add(real_sq, imag_sq);
    auto stft = graph_.scalar_sqrt(magnitude_sq);

    const size_t strides[4] = {1, 2, 2, 1};
    auto x = stft;
    for (uint32_t i = 0; i < 4; i++) {
        auto conv = graph_.conv1d_k3(x, weight_nodes_.encoder_blocks[i].conv_weight, strides[i]);
        auto bias_reshaped = graph_.reshape(weight_nodes_.encoder_blocks[i].conv_bias,
            {1, graph_.get_output_buffer(weight_nodes_.encoder_blocks[i].conv_bias).shape[0], 1});
        auto with_bias = graph_.add(conv, bias_reshaped);
        x = graph_.relu(with_bias);
    }

    auto x_squeezed = graph_.reshape(x, {1, HIDDEN_SIZE});

    graph_nodes_.lstm_output = graph_.lstm_cell(
        x_squeezed,
        graph_nodes_.h_prev,
        graph_nodes_.c_prev,
        weight_nodes_.lstm_weight_ih,
        weight_nodes_.lstm_weight_hh,
        weight_nodes_.lstm_bias_ih,
        weight_nodes_.lstm_bias_hh
    );

    auto h_new = graph_.slice(graph_nodes_.lstm_output, 2, 0, 1);
    auto c_new = graph_.slice(graph_nodes_.lstm_output, 2, 1, 1);
    graph_nodes_.h_new = graph_.reshape(h_new, {1, HIDDEN_SIZE});
    graph_nodes_.c_new = graph_.reshape(c_new, {1, HIDDEN_SIZE});

    auto h_relu = graph_.relu(graph_nodes_.h_new);
    auto h_unsqueezed = graph_.reshape(h_relu, {1, HIDDEN_SIZE, 1});
    auto logits = graph_.conv1d(h_unsqueezed, weight_nodes_.output_conv_weight, weight_nodes_.output_conv_bias, 1);
    graph_nodes_.output = graph_.sigmoid(logits);
}

bool SileroVADModel::init(const std::string& model_folder, size_t context_size,
                          const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    (void)do_warmup;

    if (initialized_) {
        return true;
    }

    state_.h.resize(HIDDEN_SIZE, (__fp16)0.0f);
    state_.c.resize(HIDDEN_SIZE, (__fp16)0.0f);
    state_.context.resize(CONTEXT_SIZE, 0.0f);

    model_folder_path_ = model_folder;

    try {
        load_weights_to_graph(&graph_);
        build_graph();
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        initialized_ = false;
        return false;
    }
}

float SileroVADModel::process_chunk(const std::vector<float>& audio_chunk) {
    const float* audio = audio_chunk.data();
    size_t samples = audio_chunk.size();

    if (samples != CHUNK_SIZE) {
        throw std::runtime_error("SileroVAD: Expected 512 samples");
    }
    if (!initialized_) {
        throw std::runtime_error("SileroVAD: Not initialized");
    }

    const size_t input_size = CONTEXT_SIZE + CHUNK_SIZE + REFLECT_PAD_SIZE;

    std::vector<__fp16> input_fp16(input_size);
    std::vector<float> input_with_context_and_pad(input_size);

    std::memcpy(input_with_context_and_pad.data(), state_.context.data(), CONTEXT_SIZE * sizeof(float));
    std::memcpy(input_with_context_and_pad.data() + CONTEXT_SIZE, audio, CHUNK_SIZE * sizeof(float));

    for (size_t i = 0; i < REFLECT_PAD_SIZE; i++) {
        input_with_context_and_pad[CONTEXT_SIZE + CHUNK_SIZE + i] =
            input_with_context_and_pad[CONTEXT_SIZE + CHUNK_SIZE - 2 - i];
    }

    cactus_fp32_to_fp16(input_with_context_and_pad.data(), input_fp16.data(), input_size);

    graph_.set_input(graph_nodes_.input, input_fp16.data(), Precision::FP16);
    graph_.set_input(graph_nodes_.h_prev, state_.h.data(), Precision::FP16);
    graph_.set_input(graph_nodes_.c_prev, state_.c.data(), Precision::FP16);
    graph_.execute();

    void* h_out = graph_.get_output(graph_nodes_.h_new);
    void* c_out = graph_.get_output(graph_nodes_.c_new);
    void* vad_out = graph_.get_output(graph_nodes_.output);

    const __fp16* h_new = static_cast<const __fp16*>(h_out);
    const __fp16* c_new = static_cast<const __fp16*>(c_out);
    const __fp16* vad_score = static_cast<const __fp16*>(vad_out);

    std::memcpy(state_.h.data(), h_new, HIDDEN_SIZE * sizeof(__fp16));
    std::memcpy(state_.c.data(), c_new, HIDDEN_SIZE * sizeof(__fp16));

    float vad_score_f32 = static_cast<float>(vad_score[0]);

    std::memcpy(state_.context.data(), input_with_context_and_pad.data() + CHUNK_SIZE, CONTEXT_SIZE * sizeof(float));

    return vad_score_f32;
}

void SileroVADModel::reset_states() {
    state_.h.assign(HIDDEN_SIZE, (__fp16)0.0f);
    state_.c.assign(HIDDEN_SIZE, (__fp16)0.0f);
    state_.context.assign(CONTEXT_SIZE, 0.0f);
}

size_t SileroVADModel::forward(const std::vector<float>&, const std::vector<uint32_t>&, bool) {
    return 0;
}

}
}
