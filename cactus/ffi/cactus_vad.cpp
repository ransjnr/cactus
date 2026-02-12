#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../models/model.h"
#include "../../libs/audio/wav.h"
#include <vector>
#include <sstream>
#include <algorithm>

using namespace cactus::engine;
using namespace cactus::ffi;

struct VADOptions {
    float threshold = 0.5f;
    float neg_threshold = 0.35f;
    int min_speech_duration_ms = 250;
    float max_speech_duration_s = std::numeric_limits<float>::infinity();
    int min_silence_duration_ms = 100;
    int speech_pad_ms = 30;
    int min_silence_at_max_speech = 98;
    bool use_max_poss_sil_at_max_speech = true;
    bool return_seconds = false;
    int sampling_rate = 16000;
    int window_size_samples = 512;
};

static void parse_vad_options(const std::string& json, VADOptions& opts) {
    if (json.empty()) return;

    size_t pos = json.find("\"threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        opts.threshold = std::stof(json.substr(pos));
    }

    pos = json.find("\"min_silence_at_max_speech\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        opts.min_silence_at_max_speech = std::stoi(json.substr(pos));
    }

    pos = json.find("\"use_max_poss_sil_at_max_speech\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        std::string value = json.substr(pos, 5);
        opts.use_max_poss_sil_at_max_speech = (value.find("true") != std::string::npos);
    }

    opts.neg_threshold = std::max(opts.threshold - 0.15f, 0.01f);
}

struct SpeechSegment {
    size_t start;
    size_t end;
};

static std::vector<SpeechSegment> get_speech_timestamps(
    const std::vector<float>& audio,
    SileroVADModel* vad_model,
    const VADOptions& opts
) {
    const size_t audio_length_samples = audio.size();
    const size_t window_size_samples = opts.window_size_samples;
    const float min_speech_samples = opts.sampling_rate * opts.min_speech_duration_ms / 1000.0f;
    const float speech_pad_samples = opts.sampling_rate * opts.speech_pad_ms / 1000.0f;
    const float max_speech_samples = opts.sampling_rate * opts.max_speech_duration_s - window_size_samples - 2 * speech_pad_samples;
    const float min_silence_samples = opts.sampling_rate * opts.min_silence_duration_ms / 1000.0f;

    vad_model->reset_states();

    std::vector<float> speech_probs;
    speech_probs.reserve(audio_length_samples / window_size_samples + 1);

    for (size_t current_start = 0; current_start < audio_length_samples; current_start += window_size_samples) {
        std::vector<float> chunk;
        chunk.reserve(window_size_samples);

        for (size_t i = current_start; i < current_start + window_size_samples && i < audio_length_samples; ++i) {
            chunk.push_back(audio[i]);
        }

        while (chunk.size() < window_size_samples) {
            chunk.push_back(0.0f);
        }

        float speech_prob = vad_model->process_chunk(chunk);
        speech_probs.push_back(speech_prob);
    }

    bool triggered = false;
    std::vector<SpeechSegment> speeches;
    SpeechSegment current_speech = {0, 0};
    size_t temp_end = 0;
    size_t prev_end = 0;
    size_t next_start = 0;
    std::vector<std::pair<size_t, size_t>> possible_ends;

    const float min_silence_samples_at_max_speech = opts.sampling_rate * opts.min_silence_at_max_speech / 1000.0f;

    for (size_t i = 0; i < speech_probs.size(); ++i) {
        float speech_prob = speech_probs[i];
        size_t cur_sample = window_size_samples * i;

        if (speech_prob >= opts.threshold && temp_end) {
            size_t sil_dur = cur_sample - temp_end;
            if (sil_dur > min_silence_samples_at_max_speech) {
                possible_ends.push_back({temp_end, sil_dur});
            }
            temp_end = 0;
            if (next_start < prev_end) {
                next_start = cur_sample;
            }
        }

        if (speech_prob >= opts.threshold && !triggered) {
            triggered = true;
            current_speech.start = cur_sample;
            continue;
        }

        if (triggered && (cur_sample - current_speech.start > max_speech_samples)) {
            if (opts.use_max_poss_sil_at_max_speech && !possible_ends.empty()) {
                auto max_silence = std::max_element(possible_ends.begin(), possible_ends.end(),
                    [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                        return a.second < b.second;
                    });
                prev_end = max_silence->first;
                size_t dur = max_silence->second;
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = {0, 0};
                next_start = prev_end + dur;

                if (next_start < prev_end + cur_sample) {
                    current_speech.start = next_start;
                } else {
                    triggered = false;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else {
                if (prev_end) {
                    current_speech.end = prev_end;
                    speeches.push_back(current_speech);
                    current_speech = {0, 0};
                    if (next_start < prev_end) {
                        triggered = false;
                    } else {
                        current_speech.start = next_start;
                    }
                    prev_end = next_start = temp_end = 0;
                    possible_ends.clear();
                } else {
                    current_speech.end = cur_sample;
                    speeches.push_back(current_speech);
                    current_speech = {0, 0};
                    prev_end = next_start = temp_end = 0;
                    triggered = false;
                    possible_ends.clear();
                    continue;
                }
            }
        }

        if (speech_prob < opts.neg_threshold && triggered) {
            if (!temp_end) {
                temp_end = cur_sample;
            }
            size_t sil_dur_now = cur_sample - temp_end;

            if (!opts.use_max_poss_sil_at_max_speech && sil_dur_now > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }

            if (sil_dur_now < min_silence_samples) {
                continue;
            } else {
                current_speech.end = temp_end;
                if ((current_speech.end - current_speech.start) > min_speech_samples) {
                    speeches.push_back(current_speech);
                }
                current_speech = {0, 0};
                prev_end = next_start = temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }
    }

    if (triggered && (audio_length_samples - current_speech.start) > min_speech_samples) {
        current_speech.end = audio_length_samples;
        speeches.push_back(current_speech);
    }

    for (size_t i = 0; i < speeches.size(); ++i) {
        if (i == 0) {
            speeches[i].start = speeches[i].start > static_cast<size_t>(speech_pad_samples)
                ? speeches[i].start - static_cast<size_t>(speech_pad_samples)
                : 0;
        }
        if (i != speeches.size() - 1) {
            size_t silence_duration = speeches[i + 1].start - speeches[i].end;
            if (silence_duration < 2 * static_cast<size_t>(speech_pad_samples)) {
                speeches[i].end += silence_duration / 2;
                speeches[i + 1].start = speeches[i + 1].start > silence_duration / 2
                    ? speeches[i + 1].start - silence_duration / 2
                    : 0;
            } else {
                speeches[i].end = std::min(audio_length_samples, speeches[i].end + static_cast<size_t>(speech_pad_samples));
                speeches[i + 1].start = speeches[i + 1].start > static_cast<size_t>(speech_pad_samples)
                    ? speeches[i + 1].start - static_cast<size_t>(speech_pad_samples)
                    : 0;
            }
        } else {
            speeches[i].end = std::min(audio_length_samples, speeches[i].end + static_cast<size_t>(speech_pad_samples));
        }
    }

    return speeches;
}

extern "C" {

int cactus_vad(
    cactus_model_t model,
    const char* audio_file_path,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const float* pcm_buffer,
    size_t pcm_sample_count
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ? "Model not initialized." : last_error_message;
        CACTUS_LOG_ERROR("vad", error_msg);
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }

    if (!response_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("vad", "Invalid parameters: response_buffer or buffer_size");
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }

    if (!audio_file_path && (!pcm_buffer || pcm_sample_count == 0)) {
        CACTUS_LOG_ERROR("vad", "No audio input provided");
        handle_error_response("Either audio_file_path or pcm_buffer must be provided", response_buffer, buffer_size);
        return -1;
    }

    if (audio_file_path && pcm_buffer && pcm_sample_count > 0) {
        CACTUS_LOG_ERROR("vad", "Both audio_file_path and pcm_buffer provided");
        handle_error_response("Cannot provide both audio_file_path and pcm_buffer", response_buffer, buffer_size);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* vad_model = dynamic_cast<SileroVADModel*>(handle->model.get());

        if (!vad_model) {
            last_error_message = "Model is not a VAD model";
            CACTUS_LOG_ERROR("vad", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        VADOptions opts;
        parse_vad_options(options_json ? options_json : "", opts);

        std::vector<float> audio;

        if (audio_file_path != nullptr) {
            AudioFP32 wav_audio = load_wav(audio_file_path);
            std::vector<float> waveform_16k = resample_to_16k_fp32(wav_audio.samples, wav_audio.sample_rate);
            audio = waveform_16k;
        } else {
            audio.assign(pcm_buffer, pcm_buffer + pcm_sample_count);
        }

        if (audio.empty()) {
            last_error_message = "Failed to load audio or audio is empty";
            CACTUS_LOG_ERROR("vad", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        auto segments = get_speech_timestamps(audio, vad_model, opts);

        float total_speech_duration = 0.0f;
        for (const auto& seg : segments) {
            total_speech_duration += (seg.end - seg.start);
        }

        std::ostringstream json;
        json << "{";
        json << "\"success\":true,";
        json << "\"error\":null,";
        json << "\"segments\":[";

        for (size_t i = 0; i < segments.size(); ++i) {
            if (i > 0) json << ",";
            json << "{";
            json << "\"start\":" << segments[i].start << ",";
            json << "\"end\":" << segments[i].end;
            json << "}";
        }

        json << "],";
        json << "\"speech_detected\":" << (!segments.empty() ? "true" : "false") << ",";
        json << "\"total_speech_samples\":" << static_cast<size_t>(total_speech_duration) << ",";
        json << "\"total_samples\":" << audio.size();

        json << "}";

        std::string response = json.str();
        if (response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("vad", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, response.c_str());
        return static_cast<int>(segments.size());

    } catch (const std::exception& e) {
        last_error_message = "Exception during VAD processing: " + std::string(e.what());
        CACTUS_LOG_ERROR("vad", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during VAD processing";
        CACTUS_LOG_ERROR("vad", last_error_message);
        handle_error_response(last_error_message, response_buffer, buffer_size);
        return -1;
    }
}

}
