#include "../cactus/ffi/cactus_ffi.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <text>\n";
        std::cerr << "Example: " << argv[0] << " weights/lfm2-1.2B \"Hello world\"\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* text = argv[2];

    // Initialize model with a reasonable context size
    cactus_model_t model = cactus_init(model_path, 4096, nullptr);

    if (!model) {
        std::cerr << "Failed to initialize model from: " << model_path << "\n";
        return 1;
    }

    // Allocate buffer for embeddings (most models have embeddings between 256-4096 dimensions)
    constexpr size_t MAX_EMBEDDING_DIM = 8192;
    std::vector<float> embeddings(MAX_EMBEDDING_DIM);
    size_t embedding_dim = 0;

    // Call cactus_embed
    int result = cactus_embed(
        model,
        text,
        embeddings.data(),
        embeddings.size() * sizeof(float),
        &embedding_dim
    );

    if (result < 0) {
        std::cerr << "Error: Failed to generate embedding (error code: " << result << ")\n";
        cactus_destroy(model);
        return 1;
    }

    // Output the embedding as a JSON array
    std::cout << "[";
    for (size_t i = 0; i < embedding_dim; i++) {
        if (i > 0) std::cout << ",";
        std::cout << embeddings[i];
    }
    std::cout << "]\n";

    cactus_destroy(model);
    return 0;
}
