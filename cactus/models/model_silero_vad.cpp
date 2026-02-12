#include "model.h"
#include <stdexcept>

namespace cactus {
namespace engine {

SileroVAD::SileroVAD() {
    h_state_.resize(128, 0.0f);
    c_state_.resize(128, 0.0f);
    context_.resize(64, 0.0f);
    encoder_blocks_.resize(4);
}

SileroVAD::~SileroVAD() = default;

bool SileroVAD::init(const std::string& weights_path) {
    if (initialized_) {
        return true;
    }

    weights_path_ = weights_path;

    try {
        load_weights();
        build_graph();
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        initialized_ = false;
        return false;
    }
}

void SileroVAD::load_weights() {
    stft_basis_ = graph_.mmap_weights(weights_path_ + "/stft_basis.weights");

    for (size_t i = 0; i < 4; i++) {
        std::string prefix = weights_path_ + "/encoder_block_" + std::to_string(i) + "_";
        encoder_blocks_[i].conv_weight = graph_.mmap_weights(prefix + "conv_weight.weights");
        encoder_blocks_[i].conv_bias = graph_.mmap_weights(prefix + "conv_bias.weights");
    }

    lstm_weight_ih_ = graph_.mmap_weights(weights_path_ + "/lstm_weight_ih.weights");
    lstm_weight_hh_ = graph_.mmap_weights(weights_path_ + "/lstm_weight_hh.weights");
    lstm_bias_ih_ = graph_.mmap_weights(weights_path_ + "/lstm_bias_ih.weights");
    lstm_bias_hh_ = graph_.mmap_weights(weights_path_ + "/lstm_bias_hh.weights");

    output_conv_weight_ = graph_.mmap_weights(weights_path_ + "/output_conv_weight.weights");
    output_conv_bias_ = graph_.mmap_weights(weights_path_ + "/output_conv_bias.weights");
}

void SileroVAD::build_graph() {
    const size_t context_size = 64;
    const size_t chunk_size = 512;
    const size_t input_size = context_size + chunk_size;

    input_node_ = graph_.input({1, 1, input_size}, Precision::FP16);
    h_prev_node_ = graph_.input({1, 128}, Precision::FP16);
    c_prev_node_ = graph_.input({1, 128}, Precision::FP16);

    auto stft_complex = graph_.conv1d(input_node_, stft_basis_, 128);
    auto real_parts = graph_.slice(stft_complex, 1, 0, 129);
    auto imag_parts = graph_.slice(stft_complex, 1, 129, 129);
    auto real_sq = graph_.multiply(real_parts, real_parts);
    auto imag_sq = graph_.multiply(imag_parts, imag_parts);
    auto magnitude_sq = graph_.add(real_sq, imag_sq);
    auto stft = graph_.scalar_sqrt(magnitude_sq);

    const size_t strides[4] = {1, 2, 2, 1};
    auto x = stft;
    for (size_t i = 0; i < 4; i++) {
        auto conv = graph_.conv1d_k3(x, encoder_blocks_[i].conv_weight, strides[i]);
        auto bias_reshaped = graph_.reshape(encoder_blocks_[i].conv_bias,
            {1, graph_.get_output_buffer(encoder_blocks_[i].conv_bias).shape[0], 1});
        auto with_bias = graph_.add(conv, bias_reshaped);
        x = graph_.relu(with_bias);
    }

    auto x_squeezed = graph_.reshape(x, {1, 128});
    auto ih_2d = graph_.reshape(lstm_weight_ih_, {512, 128});
    auto hh_2d = graph_.reshape(lstm_weight_hh_, {512, 128});
    auto ih_bias_2d = graph_.reshape(lstm_bias_ih_, {1, 512});
    auto hh_bias_2d = graph_.reshape(lstm_bias_hh_, {1, 512});

    auto gates_ih = graph_.matmul(x_squeezed, ih_2d, true);
    gates_ih = graph_.add(gates_ih, ih_bias_2d);
    auto gates_hh = graph_.matmul(h_prev_node_, hh_2d, true);
    gates_hh = graph_.add(gates_hh, hh_bias_2d);
    auto gates = graph_.add(gates_ih, gates_hh);

    auto i_gate = graph_.slice(gates, 1, 0, 128);
    auto f_gate = graph_.slice(gates, 1, 128, 128);
    auto g_gate = graph_.slice(gates, 1, 256, 128);
    auto o_gate = graph_.slice(gates, 1, 384, 128);

    i_gate = graph_.sigmoid(i_gate);
    f_gate = graph_.sigmoid(f_gate);
    g_gate = graph_.tanh(g_gate);
    o_gate = graph_.sigmoid(o_gate);

    auto f_c = graph_.multiply(f_gate, c_prev_node_);
    auto i_g = graph_.multiply(i_gate, g_gate);
    c_new_node_ = graph_.add(f_c, i_g);

    auto c_tanh = graph_.tanh(c_new_node_);
    h_new_node_ = graph_.multiply(o_gate, c_tanh);

    auto h_relu = graph_.relu(h_new_node_);
    auto h_unsqueezed = graph_.reshape(h_relu, {1, 128, 1});
    auto logits = graph_.conv1d(h_unsqueezed, output_conv_weight_, output_conv_bias_, 1);
    output_node_ = graph_.sigmoid(logits);
}

float SileroVAD::process_chunk(const float* audio, size_t samples) {
    if (samples != 512) {
        throw std::runtime_error("SileroVAD: Expected 512 samples");
    }
    if (!initialized_) {
        throw std::runtime_error("SileroVAD: Not initialized");
    }

    const size_t context_size = 64;
    const size_t chunk_size = 512;
    const size_t input_size = context_size + chunk_size;

    std::vector<__fp16> input_fp16(input_size);
    std::vector<float> input_with_context(input_size);

    for (size_t i = 0; i < context_size; i++) {
        input_with_context[i] = context_[i];
    }
    for (size_t i = 0; i < chunk_size; i++) {
        input_with_context[context_size + i] = audio[i];
    }

    for (size_t i = 0; i < input_size; i++) {
        input_fp16[i] = static_cast<__fp16>(input_with_context[i]);
    }

    std::vector<__fp16> h_fp16(128), c_fp16(128);
    for (size_t i = 0; i < 128; i++) {
        h_fp16[i] = static_cast<__fp16>(h_state_[i]);
        c_fp16[i] = static_cast<__fp16>(c_state_[i]);
    }

    graph_.set_input(input_node_, input_fp16.data(), Precision::FP16);
    graph_.set_input(h_prev_node_, h_fp16.data(), Precision::FP16);
    graph_.set_input(c_prev_node_, c_fp16.data(), Precision::FP16);
    graph_.execute();

    void* h_out = graph_.get_output(h_new_node_);
    void* c_out = graph_.get_output(c_new_node_);
    void* vad_out = graph_.get_output(output_node_);

    const __fp16* h_new = static_cast<const __fp16*>(h_out);
    const __fp16* c_new = static_cast<const __fp16*>(c_out);
    const __fp16* vad_score = static_cast<const __fp16*>(vad_out);

    for (size_t i = 0; i < 128; i++) {
        h_state_[i] = static_cast<float>(h_new[i]);
        c_state_[i] = static_cast<float>(c_new[i]);
    }

    for (size_t i = 0; i < context_size; i++) {
        context_[i] = input_with_context[input_size - context_size + i];
    }

    return static_cast<float>(vad_score[0]);
}

void SileroVAD::reset() {
    h_state_.assign(128, 0.0f);
    c_state_.assign(128, 0.0f);
    context_.assign(64, 0.0f);
}

SileroVADModel::SileroVADModel() : Model() {
    vad_ = std::make_unique<SileroVAD>();
}

SileroVADModel::SileroVADModel(const Config& config) : Model(config) {
    vad_ = std::make_unique<SileroVAD>();
}

SileroVADModel::~SileroVADModel() = default;

bool SileroVADModel::init(const std::string& model_folder, size_t context_size,
                          const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    (void)do_warmup;

    if (initialized_) {
        return true;
    }

    model_folder_path_ = model_folder;

    if (!vad_->init(model_folder)) {
        return false;
    }

    initialized_ = true;
    return true;
}

float SileroVADModel::process_chunk(const std::vector<float>& audio_chunk) {
    return vad_->process_chunk(audio_chunk.data(), audio_chunk.size());
}

void SileroVADModel::reset_states() {
    vad_->reset();
}

SileroVAD* SileroVADModel::get_vad() {
    return vad_.get();
}

void SileroVADModel::load_weights_to_graph(CactusGraph*) {
}

size_t SileroVADModel::forward(const std::vector<float>&, const std::vector<uint32_t>&, bool) {
    return 0;
}

}
}
