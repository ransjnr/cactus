#include "cactus_ffi.h"
#include "cactus_utils.h"
#include <cstring>

using namespace cactus::ffi;

static std::string extract_json_string_value(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";

    size_t colon_pos = json.find(':', pos);
    if (colon_pos == std::string::npos) return "";

    size_t start_quote = json.find('"', colon_pos);
    if (start_quote == std::string::npos) return "";

    std::string result;
    size_t i = start_quote + 1;
    while (i < json.length()) {
        if (json[i] == '"' && (i == start_quote + 1 || json[i - 1] != '\\')) {
            break;
        }
        if (json[i] == '\\' && i + 1 < json.length()) {
            char next = json[i + 1];
            if (next == '"') {
                result += '"';
                i += 2;
            } else if (next == '\\') {
                result += '\\';
                i += 2;
            } else if (next == 'n') {
                result += '\n';
                i += 2;
            } else if (next == 'r') {
                result += '\r';
                i += 2;
            } else if (next == 't') {
                result += '\t';
                i += 2;
            } else {
                result += json[i];
                i++;
            }
        } else {
            result += json[i];
            i++;
        }
    }
    return result;
}

static bool fuzzy_match(const std::string& a, const std::string& b, size_t n, double threshold) {
    if (!n) return false;
    if (a.size() < n || b.size() < n) return false;

    std::vector<size_t> dp(n + 1);
    size_t dp_im1_jm1;

    for (size_t j = 0; j <= n; ++j) dp[j] = j;

    for (size_t i = 1; i <= n; ++i) {
        dp_im1_jm1 = dp[0];
        dp[0] = i;

        for (size_t j = 1; j <= n; ++j) {
            size_t dp_im1_j = dp[j];

            if (a[i - 1] == b[j - 1]) {
                dp[j] = dp_im1_jm1;
            } else {
                dp[j] = std::min({
                    dp[j] + 1,
                    dp[j - 1] + 1,
                    dp_im1_jm1 + 1
                });
            }
            
            dp_im1_jm1 = dp_im1_j;
        }
    }

    return 1.0 - static_cast<double>(dp[n]) / static_cast<double>(n) >= threshold;
}

static void parse_stream_transcribe_init_options(const std::string& json, double& confirmation_threshold, size_t& min_chunk_size) {
    confirmation_threshold = 0.99;
    min_chunk_size = 32000;

    if (json.empty()) {
        return;
    }

    size_t pos = json.find("\"confirmation_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        confirmation_threshold = std::stod(json.substr(pos));
    }

    pos = json.find("\"min_chunk_size\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        min_chunk_size = static_cast<size_t>(std::stod(json.substr(pos)));
    }
}

struct CactusStreamTranscribeHandle {
    CactusModelHandle* model_handle;

    struct CactusStreamTranscribeOptions {
        double confirmation_threshold;
        size_t min_chunk_size;
    } options;
    
    cactus_stream_transcribe_callback callback;
    void* user_data;

    std::vector<uint8_t> audio_buffer;

    std::string previous_transcription;
    size_t previous_audio_buffer_size;

    char transcribe_response_buffer[8192];
};

extern "C" {

cactus_stream_transcribe_t cactus_stream_transcribe_start(
    cactus_model_t model,
    const char* options_json,
    cactus_stream_transcribe_callback callback,
    void* user_data
) {
    if (!model) {
        last_error_message = "Model not initialized. Check model path and files.";
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }

    try {
        auto* model_handle = static_cast<CactusModelHandle*>(model);
        if (!model_handle->model) {
            last_error_message = "Invalid model handle.";
            CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
            return nullptr;
        }

        auto* stream_handle = new CactusStreamTranscribeHandle();
        stream_handle->model_handle = model_handle;
        stream_handle->previous_audio_buffer_size = 0;
        stream_handle->transcribe_response_buffer[0] = '\0';
        stream_handle->callback = callback;
        stream_handle->user_data = user_data;

        double confirmation_threshold;
        size_t min_chunk_size;
        parse_stream_transcribe_init_options(
            options_json ? options_json : "",
            confirmation_threshold,
            min_chunk_size
        );

        stream_handle->options = { confirmation_threshold, min_chunk_size };

        CACTUS_LOG_INFO("stream_transcribe_start",
            "Stream transcription initialized for model: " << model_handle->model_name);

        return stream_handle;
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_start: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription initialization";
        CACTUS_LOG_ERROR("stream_transcribe_start", last_error_message);
        return nullptr;
    }
}

int cactus_stream_transcribe_process(
    cactus_stream_transcribe_t stream,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size,
    char* response_buffer,
    size_t buffer_size
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);

        if (pcm_buffer && pcm_buffer_size > 0) {
            handle->audio_buffer.insert(
                handle->audio_buffer.end(),
                pcm_buffer,
                pcm_buffer + pcm_buffer_size
            );
            CACTUS_LOG_DEBUG("stream_transcribe_process",
                "Inserted " << pcm_buffer_size << " bytes, buffer size: " << handle->audio_buffer.size());
        }

        if (handle->audio_buffer.size() < handle->options.min_chunk_size * sizeof(int16_t)) {
            std::string json_response = "{\"success\":true,\"confirmed\":\"\",\"pending\":\"\"}";
            if (response_buffer && buffer_size > 0) {
                if (json_response.length() >= buffer_size) {
                    last_error_message = "Response buffer too small";
                    CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                    handle_error_response(last_error_message, response_buffer, buffer_size);
                    return -1;
                }
                std::strcpy(response_buffer, json_response.c_str());
                return static_cast<int>(json_response.length());
            }
            return 0;
        }

        const int result = cactus_transcribe(
            handle->model_handle,
            nullptr,
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            handle->transcribe_response_buffer,
            sizeof(handle->transcribe_response_buffer),
            nullptr,
            nullptr,
            nullptr,
            handle->audio_buffer.data(),
            handle->audio_buffer.size());

        cactus_reset(handle->model_handle);

        if (result < 0) {
            last_error_message = "Transcription failed in stream process.";
            CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        std::string json_str(handle->transcribe_response_buffer);
        std::string response = extract_json_string_value(json_str, "response");

        std::string confirmed;
        const size_t n = std::min(handle->previous_transcription.size(), response.size());
        if (fuzzy_match(handle->previous_transcription, response, n, handle->options.confirmation_threshold)) {
            confirmed = std::move(handle->previous_transcription);

            handle->audio_buffer.erase(
                handle->audio_buffer.begin(),
                handle->audio_buffer.begin() + handle->previous_audio_buffer_size
            );

            handle->previous_transcription.clear();
            handle->previous_audio_buffer_size = 0;
        } else {
            handle->previous_transcription = response;
            handle->previous_audio_buffer_size = handle->audio_buffer.size();
        }

        std::string json_response = "{\"success\":true,\"confirmed\":\"" +
            escape_json_string(confirmed) + "\",\"pending\":\"" +
            escape_json_string(response) + "\"}";

        int bytes_written = 0;
        if (response_buffer && buffer_size > 0) {
            if (json_response.length() >= buffer_size) {
                last_error_message = "Response buffer too small";
                CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
                handle_error_response(last_error_message, response_buffer, buffer_size);
                return -1;
            }
            std::strcpy(response_buffer, json_response.c_str());
            bytes_written = static_cast<int>(json_response.length());
        }

        if (handle->callback) {
            handle->callback(confirmed.c_str(), response.c_str(), handle->user_data);
        }

        return bytes_written;
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_process: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription processing";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response("Unknown error during stream processing", response_buffer, buffer_size);
        return -1;
    }
}

int cactus_stream_transcribe_stop(
    cactus_stream_transcribe_t stream,
    char* response_buffer,
    size_t buffer_size
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        return -1;
    }

    auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);

    try {
        std::string json_response = "{\"success\":true,\"confirmed\":\"" +
            escape_json_string(handle->previous_transcription) + "\"}";

        int bytes_written = 0;
        if (response_buffer && buffer_size > 0) {
            if (json_response.length() >= buffer_size) {
                last_error_message = "Response buffer too small";
                CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
                handle_error_response(last_error_message, response_buffer, buffer_size);
                return -1;
            }
            std::strcpy(response_buffer, json_response.c_str());
            bytes_written = static_cast<int>(json_response.length());
        }

        if (handle->callback) {
            handle->callback(handle->previous_transcription.c_str(), "", handle->user_data);
        }

        delete handle;
        return bytes_written;
    } catch (const std::exception& e) {
        auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);
        last_error_message = "Exception during stream_transcribe_stop: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response(e.what(), response_buffer, buffer_size);
        delete handle;
        return -1;
    } catch (...) {
        auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);
        last_error_message = "Unknown exception during stream transcription stop";
        CACTUS_LOG_ERROR("stream_transcribe_stop", last_error_message);
        handle_error_response("Unknown error during stream stop", response_buffer, buffer_size);
        delete handle;
        return -1;
    }
}

}
