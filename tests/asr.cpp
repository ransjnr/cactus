#include "../cactus/ffi/cactus_ffi.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>

#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#endif

constexpr size_t RESPONSE_BUFFER_SIZE = 65536;

namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string DIM     = "\033[2m";
    const std::string CYAN    = "\033[36m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string RED     = "\033[31m";
    const std::string GRAY    = "\033[90m";
#ifdef HAVE_SDL2
    const std::string CLEAR_LINE = "\033[2K\r";
#endif
}

bool supports_color() {
#ifdef _WIN32
    return false;
#else
    const char* term = std::getenv("TERM");
    return term && std::string(term) != "dumb";
#endif
}

bool use_colors = supports_color();

std::string colored(const std::string& text, const std::string& color) {
    if (!use_colors) return text;
    return color + text + Color::RESET;
}

void print_separator(char ch = '-', int width = 60) {
    std::cout << colored(std::string(width, ch), Color::DIM) << "\n";
}

void print_header_live_mode() {
    std::cout << "\n";
    print_separator('=');
    std::cout << colored("     🌵 CACTUS LIVE TRANSCRIPTION 🌵", Color::GREEN + Color::BOLD) << "\n";
    print_separator('=');
    std::cout << colored("Listening...", Color::YELLOW) << " Press " << colored("Enter", Color::CYAN) << " to stop\n";
    print_separator();
    std::cout << "\n";
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string extract_json_value(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":\"";
    size_t start = json.find(pattern);
    if (start == std::string::npos) return "";
    start += pattern.length();
    size_t end = start;
    while (end < json.length() && json[end] != '"') {
        if (json[end] == '\\' && end + 1 < json.length()) end++;
        end++;
    }
    return json.substr(start, end - start);
}

void print_token(const char* token, uint32_t /*token_id*/, void* /*user_data*/) {
    std::cout << token << std::flush;
}

std::string get_transcribe_prompt(const std::string& model_path) {
    std::string path_lower = model_path;
    std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (path_lower.find("whisper") != std::string::npos) {
        return "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";
    }
    return "";  // Moonshine uses empty prompt
}

int transcribe_file(cactus_model_t model, const std::string& audio_path, const std::string& model_path) {
    if (!file_exists(audio_path)) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "File not found: " << audio_path << "\n";
        return -1;
    }

    std::string prompt = get_transcribe_prompt(model_path);
    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

    auto start_time = std::chrono::steady_clock::now();

    int result = cactus_transcribe(
        model,
        audio_path.c_str(),
        prompt.c_str(),
        response_buffer.data(),
        response_buffer.size(),
        R"({"max_tokens": 500})",
        print_token,
        nullptr,
        nullptr,
        0
    );

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_seconds = duration.count() / 1000.0;

    if (result < 0) {
        std::cerr << "\n" << colored("Error: ", Color::RED + Color::BOLD)
                  << "Transcription failed\n";
        const char* err = cactus_get_last_error();
        if (err) {
            std::cerr << colored("Details: ", Color::RED) << err << "\n";
        }
        return -1;
    }

    std::string json_str(response_buffer.data());

    std::string time_str;
    size_t time_pos = json_str.find("\"total_time_ms\":");
    if (time_pos != std::string::npos) {
        size_t start = time_pos + 16;
        size_t end = json_str.find_first_of(",}", start);
        time_str = json_str.substr(start, end - start);
    }

    std::ostringstream stats;
    stats << std::fixed << std::setprecision(2);
    stats << "\n\n" << colored("[", Color::GRAY);
    stats << colored("processed in: ", Color::GRAY) << total_seconds << "s";
    if (!time_str.empty()) {
        double model_time = std::stod(time_str) / 1000.0;
        stats << colored(" | model time: ", Color::GRAY) << model_time << "s";
    }
    stats << colored("]", Color::GRAY);

    std::cout << stats.str() << "\n";

    return 0;
}

#ifdef HAVE_SDL2

constexpr int SAMPLE_RATE = 16000;
constexpr int AUDIO_BUFFER_MS = 100;

struct AudioState {
    std::mutex mutex;
    std::vector<uint8_t> buffer;
    std::atomic<bool> recording{false};
};

AudioState g_audio_state;

void audio_callback(void* /*userdata*/, Uint8* stream, int len) {
    if (!g_audio_state.recording) return;

    std::lock_guard<std::mutex> lock(g_audio_state.mutex);
    g_audio_state.buffer.insert(g_audio_state.buffer.end(), stream, stream + len);
}

int run_live_transcription(cactus_model_t model) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Failed to initialize SDL: " << SDL_GetError() << "\n";
        return 1;
    }

    int num_devices = SDL_GetNumAudioDevices(1); 
    if (num_devices == 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "No audio capture devices found\n";
        SDL_Quit();
        return 1;
    }

    std::cout << colored("Available microphones:", Color::YELLOW) << "\n";
    for (int i = 0; i < num_devices; i++) {
        std::cout << "  [" << i << "] " << SDL_GetAudioDeviceName(i, 1) << "\n";
    }
    std::cout << "\n";

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = SAMPLE_RATE;
    want.format = AUDIO_S16LSB;
    want.channels = 1;
    want.samples = (SAMPLE_RATE * AUDIO_BUFFER_MS) / 1000;
    want.callback = audio_callback;
    want.userdata = nullptr;

    SDL_AudioDeviceID device = SDL_OpenAudioDevice(nullptr, 1, &want, &have, 0);
    if (device == 0) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Failed to open audio device: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    cactus_stream_transcribe_t stream = cactus_stream_transcribe_start(
        model, R"({"confirmation_threshold": 1.0, "min_chunk_size": 16000})"
    );

    if (!stream) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Failed to initialize streaming transcription\n";
        SDL_CloseAudioDevice(device);
        SDL_Quit();
        return 1;
    }

    print_header_live_mode();

    g_audio_state.recording = true;
    g_audio_state.buffer.clear();
    SDL_PauseAudioDevice(device, 0);

    std::atomic<bool> should_stop{false};
    std::thread input_thread([&should_stop]() {
        std::string line;
        std::getline(std::cin, line);
        should_stop = true;
    });

    std::string confirmed_text;
    std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

    auto last_process_time = std::chrono::steady_clock::now();
    const auto process_interval = std::chrono::milliseconds(500);

    while (!should_stop) {
        auto now = std::chrono::steady_clock::now();

        if (now - last_process_time >= process_interval) {
            last_process_time = now;

            std::vector<uint8_t> audio_chunk;
            {
                std::lock_guard<std::mutex> lock(g_audio_state.mutex);
                if (!g_audio_state.buffer.empty()) {
                    audio_chunk = std::move(g_audio_state.buffer);
                    g_audio_state.buffer.clear();
                }
            }

            if (!audio_chunk.empty()) {
                int process_result = cactus_stream_transcribe_process(
                    stream,
                    audio_chunk.data(),
                    audio_chunk.size(),
                    response_buffer.data(),
                    response_buffer.size()
                );

                if (process_result >= 0) {
                    std::string json_str(response_buffer.data());
                    std::string confirmed = extract_json_value(json_str, "confirmed");
                    std::string pending = extract_json_value(json_str, "pending");

                    // Clear line and show transcription
                    std::cout << Color::CLEAR_LINE;
                    if (!confirmed_text.empty()) {
                        std::cout << colored(confirmed_text, Color::GREEN);
                    }
                    if (!pending.empty()) {
                        std::cout << colored(pending, Color::YELLOW);
                    }
                    std::cout << std::flush;

                    if (!confirmed.empty()) {
                        confirmed_text += confirmed + " ";
                    }
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    g_audio_state.recording = false;
    SDL_PauseAudioDevice(device, 1);

    int stopping_result = cactus_stream_transcribe_stop(
        stream,
        response_buffer.data(),
        response_buffer.size()
    );

    std::cout << "\n\n";
    print_separator();

    if (stopping_result >= 0) {
        std::string json_str(response_buffer.data());
        std::string final_text = extract_json_value(json_str, "confirmed");
        std::string full_transcript = confirmed_text + final_text;

        std::cout << colored("Final transcript:", Color::GREEN + Color::BOLD) << "\n";
        std::cout << full_transcript << "\n";
    } else {
        if (!confirmed_text.empty()) {
            std::cout << colored("Partial transcript:", Color::YELLOW + Color::BOLD) << "\n";
            std::cout << confirmed_text << "\n";
        }
    }

    print_separator();

    if (input_thread.joinable()) {
        input_thread.detach();
    }
    SDL_CloseAudioDevice(device);
    SDL_Quit();

    return 0;
}

#endif // HAVE_SDL2

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Missing model path\n";
        std::cerr << "Usage: " << argv[0] << " <model_path> [audio_file]\n";
        std::cerr << "\nModes:\n";
        std::cerr << "  " << argv[0] << " weights/moonshine-base              # Live microphone transcription\n";
        std::cerr << "  " << argv[0] << " weights/moonshine-base audio.wav    # Transcribe single file\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* audio_file = argc > 2 ? argv[2] : nullptr;

    std::cout << "\n" << colored("Loading model from ", Color::YELLOW)
              << colored(model_path, Color::CYAN) << colored("...", Color::YELLOW) << "\n";

    cactus_model_t model = cactus_init(model_path, nullptr);

    if (!model) {
        std::cerr << colored("Failed to initialize model\n", Color::RED + Color::BOLD);
        const char* err = cactus_get_last_error();
        if (err) {
            std::cerr << colored("Error: ", Color::RED) << err << "\n";
        }
        return 1;
    }

    std::cout << colored("Model loaded successfully!\n", Color::GREEN + Color::BOLD);

    int result = 0;

    if (audio_file) {
        std::cout << "\n" << colored("Transcribing: ", Color::BLUE + Color::BOLD)
                  << audio_file << "\n\n";
        result = transcribe_file(model, audio_file, model_path);
    } else {
#ifdef HAVE_SDL2
        result = run_live_transcription(model);
#else
        std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                  << "Live transcription requires SDL2.\n";
        std::cerr << "Please install SDL2 and rebuild:\n";
        std::cerr << "  macOS:  brew install sdl2\n";
        std::cerr << "  Linux:  sudo apt-get install libsdl2-dev\n";
        result = 1;
#endif
    }

    std::cout << colored("\n👋 Goodbye!\n", Color::MAGENTA + Color::BOLD);
    cactus_destroy(model);
    return result >= 0 ? 0 : 1;
}
