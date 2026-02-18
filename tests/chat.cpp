#include "../cactus/ffi/cactus_ffi.h"
#include "../cactus/telemetry/telemetry.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <thread>
#include <curl/curl.h>
#include <fstream>
#include <signal.h>
#include <unistd.h>

constexpr int MAX_TOKENS = 512;
constexpr size_t MAX_BYTES_PER_TOKEN = 64;
constexpr size_t RESPONSE_BUFFER_SIZE = MAX_TOKENS * MAX_BYTES_PER_TOKEN;

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

void print_header() {
    std::cout << "\n";
    print_separator('=');
    std::cout << colored("           🌵 CACTUS CHAT INTERFACE 🌵", Color::GREEN + Color::BOLD) << "\n";
    print_separator('=');
    std::cout << colored("Commands:", Color::YELLOW) << "\n";
    std::cout << "  • " << colored("reset", Color::CYAN) << " - Clear conversation history\n";
    std::cout << "  • " << colored("exit", Color::CYAN) << " or " << colored("quit", Color::CYAN) << " - Exit the program\n";
    print_separator();
    std::cout << "\n";
}

struct TokenPrinter {
    bool first_token = true;
    int token_count = 0;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point first_token_time;
    double time_to_first_token = 0.0;

    void reset() {
        first_token = true;
        token_count = 0;
        time_to_first_token = 0.0;
        start_time = std::chrono::steady_clock::now();
    }

    void print(const char* token) {
        if (first_token) {
            first_token = false;
            first_token_time = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
            time_to_first_token = latency.count() / 1000.0;
        }
        std::cout << token << std::flush;
        token_count++;
    }

    void print_stats() {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double total_seconds = duration.count() / 1000.0;
        double tokens_per_second = token_count / total_seconds;

        // Format the stats with fixed decimal places
        std::ostringstream stats;
        stats << std::fixed << std::setprecision(3);
        stats << "[" << token_count << " tokens | ";
        stats << "latency: " << time_to_first_token << "s | ";
        stats << "total: " << total_seconds << "s | ";
        stats << std::setprecision(0) << static_cast<int>(tokens_per_second) << " tok/s]";

        std::cout << "\n" << colored(stats.str(), Color::GRAY) << "\n";
    }
};

TokenPrinter* g_printer = nullptr;

void print_token(const char* token, uint32_t /*token_id*/, void* /*user_data*/) {
    if (g_printer) {
        g_printer->print(token);
    }
}

std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {  
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {  
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    o << c;
                }
                break;
        }
    }
    return o.str();
}

std::string unescape_json(const std::string& s) {
    std::string result;
    result.reserve(s.length());

    for (size_t i = 0; i < s.length(); i++) {
        if (s[i] == '\\' && i + 1 < s.length()) {
            switch (s[i + 1]) {
                case '"':  result += '"'; i++; break;
                case '\\': result += '\\'; i++; break;
                case 'b':  result += '\b'; i++; break;
                case 'f':  result += '\f'; i++; break;
                case 'n':  result += '\n'; i++; break;
                case 'r':  result += '\r'; i++; break;
                case 't':  result += '\t'; i++; break;
                case 'u':
                    if (i + 5 < s.length()) {
                        std::string hex = s.substr(i + 2, 4);
                        char* end;
                        int codepoint = std::strtol(hex.c_str(), &end, 16);
                        if (end == hex.c_str() + 4) {
                            result += static_cast<char>(codepoint);
                            i += 5;
                        } else {
                            result += s[i];
                        }
                    } else {
                        result += s[i];
                    }
                    break;
                default:   result += s[i]; break;
            }
        } else {
            result += s[i];
        }
    }
    return result;
}

// Tool execution via HTTP
class ToolExecutor {
private:
    CURL* curl;
    std::string base_url;
    pid_t server_pid;

    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        userp->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

    std::string http_get(const std::string& endpoint) {
        std::string response;
        curl_easy_setopt(curl, CURLOPT_URL, (base_url + endpoint).c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
        CURLcode res = curl_easy_perform(curl);
        return (res == CURLE_OK) ? response : "";
    }

    std::string http_post(const std::string& endpoint, const std::string& data) {
        std::string response;
        curl_easy_setopt(curl, CURLOPT_URL, (base_url + endpoint).c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        CURLcode res = curl_easy_perform(curl);
        curl_slist_free_all(headers);

        return (res == CURLE_OK) ? response : "";
    }

public:
    std::string tools_json;

    ToolExecutor(const char* tools_path, int port = 8765) : server_pid(-1) {
        base_url = "http://127.0.0.1:" + std::to_string(port);
        curl = curl_easy_init();

        // Find tools_server.py in same directory as tools file
        std::string tools_path_str(tools_path);
        size_t last_slash = tools_path_str.rfind('/');
        std::string tools_dir = (last_slash != std::string::npos)
            ? tools_path_str.substr(0, last_slash)
            : ".";
        std::string server_script = tools_dir + "/tools_server.py";

        // Start Python tools server
        server_pid = fork();
        if (server_pid == 0) {
            // Child process: run Python server
            // Use 'python3' from PATH (respects venv) instead of absolute path
            execlp("python3", "python3", server_script.c_str(), tools_path, std::to_string(port).c_str(), nullptr);

            // If execl returns, it failed
            std::cerr << "[Tools] Failed to start server: " << strerror(errno) << std::endl;
            exit(1);
        }

        // Parent: wait for server to start
        std::cout << colored("[Tools] Starting server on port " + std::to_string(port) + "...\n", Color::DIM);

        int attempts = 0;
        while (attempts < 20) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            std::string health = http_get("/health");
            if (!health.empty() && health.find("\"status\":\"ok\"") != std::string::npos) {
                std::cout << colored("[Tools] Server ready\n", Color::DIM);
                break;
            }
            attempts++;
        }

        if (attempts >= 20) {
            std::cerr << colored("[Tools] Failed to start server\n", Color::RED);
            tools_json = "[]";
            return;
        }

        // Fetch tool schemas
        tools_json = http_get("/schemas");
        if (tools_json.empty()) {
            std::cerr << colored("[Tools] Failed to fetch schemas\n", Color::RED);
            tools_json = "[]";
        }
    }

    std::string execute(const std::string& name, const std::string& args_json) {
        std::ostringstream payload;
        payload << "{\"name\":\"" << name << "\",\"arguments\":" << args_json << "}";

        std::string response = http_post("/execute", payload.str());
        return response;
    }

    ~ToolExecutor() {
        if (server_pid > 0) {
            kill(server_pid, SIGTERM);
        }
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << colored("Error: ", Color::RED + Color::BOLD) << "Invalid arguments\n";
        std::cerr << "Usage: " << argv[0] << " <model_path> [tools_file]\n";
        std::cerr << "Example: " << argv[0] << " weights/lfm2-1.2B\n";
        std::cerr << "Example: " << argv[0] << " weights/lfm2-1.2B python/tools/example_tools.py\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* tools_path = (argc == 3) ? argv[2] : nullptr;

    std::cout << "\n" << colored("Loading model from ", Color::YELLOW)
              << colored(model_path, Color::CYAN) << colored("...", Color::YELLOW) << "\n";

    cactus_model_t model = cactus_init(model_path, nullptr, false);

    if (!model) {
        std::cerr << colored("Failed to initialize model\n", Color::RED + Color::BOLD);
        return 1;
    }

    std::cout << colored("Model loaded successfully!\n", Color::GREEN + Color::BOLD);

    // Initialize tool executor if tools provided
    ToolExecutor* executor = nullptr;
    if (tools_path) {
        executor = new ToolExecutor(tools_path);
        if (!executor->tools_json.empty() && executor->tools_json != "[]") {
            std::cout << colored("[Tools] Enabled\n", Color::GREEN);
        } else {
            std::cout << colored("[Tools] Failed to load\n", Color::YELLOW);
            delete executor;
            executor = nullptr;
        }
    }

    print_header();

    std::vector<std::string> history;
    TokenPrinter printer;
    g_printer = &printer;

    while (true) {
        std::cout << colored("You: ", Color::BLUE + Color::BOLD);
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input.empty()) continue;

        if (user_input == "quit" || user_input == "exit") {
            break;
        }

        if (user_input == "reset") {
            history.clear();
            cactus_reset(model);
            std::cout << colored("🔄 Conversation reset.\n", Color::YELLOW);
            print_separator();
            std::cout << "\n";
            continue;
        }

        history.push_back(user_input);

        // Build the messages JSON
        std::ostringstream messages_json;
        messages_json << "[";
        for (size_t i = 0; i < history.size(); i++) {
            if (i > 0) messages_json << ",";
            if (i % 2 == 0) {
                messages_json << "{\"role\":\"user\",\"content\":\""
                             << escape_json(history[i]) << "\"}";
            } else {
                messages_json << "{\"role\":\"assistant\",\"content\":\""
                             << escape_json(history[i]) << "\"}";
            }
        }
        messages_json << "]";

        std::string options = "{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":40,\"max_tokens\":"
                    + std::to_string(MAX_TOKENS)
                    + ",\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\"]}";

        std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

        std::cout << colored("Assistant: ", Color::GREEN + Color::BOLD);

        printer.reset();
        int result = cactus_complete(
            model,
            messages_json.str().c_str(),
            response_buffer.data(),
            response_buffer.size(),
            options.c_str(),
            executor ? executor->tools_json.c_str() : nullptr,
            print_token,
            nullptr
        );

        if (result >= 0) {
            printer.print_stats();
        }

        std::cout << "\n";
        print_separator();
        std::cout << "\n";

        if (result < 0) {
            std::cerr << colored("Error: ", Color::RED + Color::BOLD)
                      << response_buffer.data() << "\n\n";
            history.pop_back();
            continue;
        }

        std::string json_str(response_buffer.data(), response_buffer.size());

        // Check for function calls
        size_t func_calls_pos = json_str.find("\"function_calls\":");
        bool has_function_calls = false;
        if (func_calls_pos != std::string::npos && executor) {
            size_t array_start = json_str.find("[", func_calls_pos);
            size_t array_end = json_str.find("]", array_start);
            if (array_start != std::string::npos && array_end != std::string::npos) {
                std::string calls_array = json_str.substr(array_start, array_end - array_start + 1);
                if (calls_array != "[]" && calls_array.find("{") != std::string::npos) {
                    has_function_calls = true;

                    // Parse and execute function calls
                    size_t pos = 0;
                    while ((pos = calls_array.find("{\"name\":", pos)) != std::string::npos) {
                        size_t name_start = calls_array.find("\"", pos + 8) + 1;
                        size_t name_end = calls_array.find("\"", name_start);
                        std::string tool_name = calls_array.substr(name_start, name_end - name_start);

                        size_t args_start = calls_array.find("\"arguments\":", name_end);
                        size_t args_obj_start = calls_array.find("{", args_start);
                        int brace_count = 1;
                        size_t args_obj_end = args_obj_start + 1;
                        while (args_obj_end < calls_array.length() && brace_count > 0) {
                            if (calls_array[args_obj_end] == '{') brace_count++;
                            else if (calls_array[args_obj_end] == '}') brace_count--;
                            args_obj_end++;
                        }
                        std::string args_json = calls_array.substr(args_obj_start, args_obj_end - args_obj_start);

                        // Execute tool
                        std::string tool_response = executor->execute(tool_name, args_json);

                        // Parse tool response to extract key fields for display
                        std::string display_summary;
                        size_t result_pos = tool_response.find("\"result\":");
                        if (result_pos != std::string::npos) {
                            size_t result_obj_start = tool_response.find("{", result_pos);
                            if (result_obj_start != std::string::npos) {
                                // Try to extract key info based on tool type
                                std::string location, temp, condition, humidity, rolls, total;

                                size_t loc_pos = tool_response.find("\"location\":", result_obj_start);
                                if (loc_pos != std::string::npos) {
                                    size_t loc_start = tool_response.find("\"", loc_pos + 11) + 1;
                                    size_t loc_end = tool_response.find("\"", loc_start);
                                    location = tool_response.substr(loc_start, loc_end - loc_start);
                                }

                                size_t temp_pos = tool_response.find("\"temperature\":", result_obj_start);
                                if (temp_pos != std::string::npos) {
                                    size_t temp_start = temp_pos + 14;
                                    while (temp_start < tool_response.length() &&
                                           (tool_response[temp_start] == ' ' || tool_response[temp_start] == ':')) temp_start++;
                                    size_t temp_end = temp_start;
                                    while (temp_end < tool_response.length() &&
                                           (isdigit(tool_response[temp_end]) || tool_response[temp_end] == '.' || tool_response[temp_end] == '-')) temp_end++;
                                    temp = tool_response.substr(temp_start, temp_end - temp_start);
                                }

                                size_t cond_pos = tool_response.find("\"condition\":", result_obj_start);
                                if (cond_pos != std::string::npos) {
                                    size_t cond_start = tool_response.find("\"", cond_pos + 12) + 1;
                                    size_t cond_end = tool_response.find("\"", cond_start);
                                    condition = tool_response.substr(cond_start, cond_end - cond_start);
                                }

                                size_t hum_pos = tool_response.find("\"humidity\":", result_obj_start);
                                if (hum_pos != std::string::npos) {
                                    size_t hum_start = hum_pos + 11;
                                    while (hum_start < tool_response.length() &&
                                           (tool_response[hum_start] == ' ' || tool_response[hum_start] == ':')) hum_start++;
                                    size_t hum_end = hum_start;
                                    while (hum_end < tool_response.length() && isdigit(tool_response[hum_end])) hum_end++;
                                    humidity = tool_response.substr(hum_start, hum_end - hum_start);
                                }

                                size_t rolls_pos = tool_response.find("\"rolls\":[", result_obj_start);
                                if (rolls_pos != std::string::npos) {
                                    size_t rolls_start = rolls_pos + 9;
                                    size_t rolls_end = tool_response.find("]", rolls_start);
                                    rolls = tool_response.substr(rolls_start, rolls_end - rolls_start);
                                }

                                size_t total_pos = tool_response.find("\"total\":", result_obj_start);
                                if (total_pos != std::string::npos) {
                                    size_t total_start = total_pos + 8;
                                    while (total_start < tool_response.length() &&
                                           (tool_response[total_start] == ' ' || tool_response[total_start] == ':')) total_start++;
                                    size_t total_end = total_start;
                                    while (total_end < tool_response.length() &&
                                           (isdigit(tool_response[total_end]) || tool_response[total_end] == '.')) total_end++;
                                    total = tool_response.substr(total_start, total_end - total_start);
                                }

                                // Build display summary
                                if (!location.empty()) {
                                    display_summary = location;
                                    std::string details = "   → ";
                                    if (!temp.empty()) details += temp + "°C";
                                    if (!condition.empty()) {
                                        if (!temp.empty()) details += ", ";
                                        details += condition;
                                    }
                                    if (!humidity.empty()) details += ", humidity " + humidity + "%";
                                    if (details != "   → ") display_summary += "\n" + details;
                                } else if (!rolls.empty()) {
                                    display_summary = "rolled";
                                    display_summary += "\n   → [" + rolls + "]";
                                    if (!total.empty()) display_summary += " = " + total;
                                }
                            }
                        }

                        // Display in Option A format
                        std::cout << colored("🔧 " + tool_name, Color::CYAN);
                        if (!display_summary.empty()) {
                            std::cout << colored(" → ", Color::DIM) << display_summary << "\n";
                        } else {
                            std::cout << "\n";
                        }

                        // Extract just the result field and stringify it for the model
                        std::string result_content;
                        size_t content_result_pos = tool_response.find("\"result\":");
                        if (content_result_pos != std::string::npos) {
                            size_t content_result_start = tool_response.find("{", content_result_pos);
                            if (content_result_start != std::string::npos) {
                                int brace_count = 1;
                                size_t content_result_end = content_result_start + 1;
                                while (content_result_end < tool_response.length() && brace_count > 0) {
                                    if (tool_response[content_result_end] == '{') brace_count++;
                                    else if (tool_response[content_result_end] == '}') brace_count--;
                                    content_result_end++;
                                }
                                result_content = tool_response.substr(content_result_start, content_result_end - content_result_start);
                            }
                        }

                        // Add tool result to history
                        history.push_back("[TOOL:" + tool_name + "]" + result_content);

                        pos = args_obj_end;
                    }

                    // Continue generation with tool results
                    messages_json.str("");
                    messages_json << "[";
                    for (size_t i = 0; i < history.size(); i++) {
                        if (i > 0) messages_json << ",";
                        std::string msg = history[i];
                        if (msg.find("[TOOL:") == 0) {
                            // Tool result message - content must be escaped JSON string, not nested object
                            size_t tool_end = msg.find("]");
                            std::string tool_name = msg.substr(6, tool_end - 6);
                            std::string tool_result = msg.substr(tool_end + 1);
                            messages_json << "{\"role\":\"tool\",\"name\":\"" << tool_name
                                        << "\",\"content\":\"" << escape_json(tool_result) << "\"}";
                        } else if (i % 2 == 0) {
                            messages_json << "{\"role\":\"user\",\"content\":\""
                                        << escape_json(msg) << "\"}";
                        } else {
                            messages_json << "{\"role\":\"assistant\",\"content\":\""
                                        << escape_json(msg) << "\"}";
                        }
                    }
                    messages_json << "]";

                    std::cout << colored("Assistant (with tool results): ", Color::GREEN + Color::BOLD);
                    printer.reset();
                    result = cactus_complete(
                        model,
                        messages_json.str().c_str(),
                        response_buffer.data(),
                        response_buffer.size(),
                        options.c_str(),
                        nullptr, // Don't pass tools again to avoid loops
                        print_token,
                        nullptr
                    );

                    if (result >= 0) {
                        printer.print_stats();
                    }
                    std::cout << "\n";
                    print_separator();
                    std::cout << "\n";

                    json_str = std::string(response_buffer.data(), response_buffer.size());
                }
            }
        }

        // Extract final response
        const std::string search_str = "\"response\":\"";
        size_t response_start = json_str.find(search_str);
        if (response_start != std::string::npos) {
            response_start += search_str.length();
            size_t response_end = json_str.find("\"", response_start);
            while (response_end != std::string::npos) {
                size_t prior_backslashes = 0;
                for (size_t i = response_end; i > response_start && json_str[i - 1] == '\\'; i--) {
                    prior_backslashes++;
                }
                if (prior_backslashes % 2 == 0) {
                    break;
                }
                response_end = json_str.find("\"", response_end + 1);
            }
            if (response_end != std::string::npos) {
                std::string response = json_str.substr(response_start,
                                                       response_end - response_start);
                history.push_back(unescape_json(response));
            }
        }
    }

    std::cout << colored("\n👋 Goodbye!\n", Color::MAGENTA + Color::BOLD);

    if (executor) {
        delete executor;
    }

    cactus_destroy(model);
    return 0;
}
