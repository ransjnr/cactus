# Cactus Index FFI Documentation

The Cactus Index provides a clean C FFI (Foreign Function Interface) for integrating a vector database into various applications. This documentation covers all available functions, their parameters, and usage examples.

## Getting Started

The index stores documents with embeddings in memory-mapped files for efficient similarity search:
- `index.bin`: Embeddings (FP16) and metadata pointers
- `data.bin`: Document content and metadata (UTF-8 strings)

All embeddings are automatically normalized to unit length internally. Both document embeddings (at add time) and query embeddings (at query time) are normalized

## Types

### `cactus_index_t`
An opaque pointer type representing an index instance. This handle is used throughout the API to reference a specific index.

```c
typedef void* cactus_index_t;
```

## Core Functions

### `cactus_index_init`
Initializes or opens an index with specified embedding dimension.

```c
cactus_index_t cactus_index_init(
    const char* index_dir,
    size_t embedding_dim
);
```

**Parameters:**
- `index_dir`: Directory path where index files will be stored
- `embedding_dim`: Dimension of embeddings (must match for existing index)

**Returns:** Index handle on success, NULL on failure

**Example:**
```c
cactus_index_t index = cactus_index_init("./my_index", 768);
if (!index) {
    fprintf(stderr, "Failed to initialize index\n");
    return -1;
}
```

### `cactus_index_add`
Adds documents to the index in batches.

```c
int cactus_index_add(
    cactus_index_t index,
    const int* ids,
    const char** documents,
    const char** metadatas,
    const float** embeddings,
    size_t count,
    size_t embedding_dim
);
```

**Parameters:**
- `index`: Index handle from `cactus_index_init` (required)
- `ids`: Array of unique document IDs (required)
- `documents`: Array of document content strings (required, UTF-8)
- `metadatas`: Array of metadata strings (optional, can be NULL; UTF-8)
- `embeddings`: Array of pointers to embedding vectors. (required, none can be NULL)
- `count`: Number of documents to add (must be > 0)
- `embedding_dim`: Dimension of embeddings (must match index, must be > 0)

**Returns:** 0 on success, -1 on error

**Constraints:**
- Document IDs must be unique integers
- Content max 65535 bytes per document
- Metadata max 65535 bytes per document
- Embedding dimension must match index dimension from init
- `ids`, `documents`, and `embeddings` arrays must be non-null
- Each individual `embeddings[i]` pointer must be non-null
- Individual `documents[i]` can be NULL (stored as empty string)
- `metadatas` array can be NULL, or individual `metadatas[i]` can be NULL

**Example:**
```c
int ids[] = {1, 2};
const char* docs[] = {"AI is transforming technology", "Machine learning enables predictions"};
const char* metas[] = {"{\"source\":\"wiki\"}", "{\"source\":\"blog\"}"};

float emb1[768] = {0.1, 0.2, 0.3, /* ... */};
float emb2[768] = {0.4, 0.5, 0.6, /* ... */};
const float* embeddings[] = {emb1, emb2};

int result = cactus_index_add(index, ids, docs, metas, embeddings, 2, 768);
if (result != 0) {
    fprintf(stderr, "Failed to add documents: %s\n", cactus_get_last_error());
}
```

**Example without metadata:**
```c
int ids[] = {1, 2};
const char* docs[] = {"Document one", "Document two"};

float emb1[768] = {0.1, 0.2, 0.3, /* ... */};
float emb2[768] = {0.4, 0.5, 0.6, /* ... */};
const float* embeddings[] = {emb1, emb2};

// Pass NULL for metadatas
int result = cactus_index_add(index, ids, docs, NULL, embeddings, 2, 768);
```

### `cactus_index_delete`
Marks documents as deleted (soft delete). Space reclaimed via `cactus_index_compact`.

```c
int cactus_index_delete(
    cactus_index_t index,
    const int* ids,
    size_t ids_count
);
```

**Parameters:**
- `index`: Index handle from `cactus_index_init` (required)
- `ids`: Array of document IDs to delete (required)
- `ids_count`: Number of document IDs (must be > 0)

**Returns:** 0 on success, -1 on error

**Example:**
```c
int ids[] = {1, 2, 3};
int result = cactus_index_delete(index, ids, 3);
if (result != 0) {
    fprintf(stderr, "Failed to delete documents: %s\n", cactus_get_last_error());
}
```

### `cactus_index_get`
Retrieves documents by IDs. Allows flexible retrieval - you can fetch only the fields you need.

```c
int cactus_index_get(
    cactus_index_t index,
    const int* ids,
    size_t ids_count,
    char** document_buffers,
    size_t* document_buffer_sizes,
    char** metadata_buffers,
    size_t* metadata_buffer_sizes,
    float** embedding_buffers,
    size_t* embedding_buffer_sizes
);
```

**Parameters:**
- `index`: Index handle from `cactus_index_init` (required)
- `ids`: Array of document IDs to retrieve (required)
- `ids_count`: Number of document IDs (must be > 0)
- `document_buffers`: Array of pre-allocated buffers for document content (optional, can be NULL)
- `document_buffer_sizes`: Array serving dual purpose (required if `document_buffers` is not NULL):
  - **Input**: Initial capacity of each buffer in bytes
  - **Output**: Actual size of data written to each buffer (including null terminator)
- `metadata_buffers`: Array of pre-allocated buffers for metadata (optional, can be NULL)
- `metadata_buffer_sizes`: Array for metadata buffer sizes (required if `metadata_buffers` is not NULL)
- `embedding_buffers`: Array of pre-allocated buffers for embeddings (optional, can be NULL)
- `embedding_buffer_sizes`: Array for embedding buffer sizes (required if `embedding_buffers` is not NULL)
  - **Input**: Capacity in number of floats
  - **Output**: Actual number of floats written

**Returns:** 0 on success, -1 on error

**Buffer Requirements:**
- If a buffer array is provided, its corresponding size array must also be provided
- If any buffer is too small, the function returns -1 with no data copied

**Example - Retrieve all fields:**
```c
int ids[] = {1, 2};
char* docs[2];
char* metas[2];
float* embs[2];

// Allocate buffers
for (int i = 0; i < 2; i++) {
    docs[i] = malloc(65536);
    metas[i] = malloc(65536);
    embs[i] = malloc(768 * sizeof(float));
}

size_t doc_sizes[2] = {65536, 65536};
size_t meta_sizes[2] = {65536, 65536};
size_t emb_sizes[2] = {768, 768};  // Number of floats

int result = cactus_index_get(index, ids, 2,
                               docs, doc_sizes,
                               metas, meta_sizes,
                               embs, emb_sizes);

if (result == 0) {
    printf("Retrieved documents successfully\n");
    for (int i = 0; i < 2; i++) {
        printf("Doc %d (%zu bytes): %s\n", ids[i], doc_sizes[i], docs[i]);
        printf("Embedding dim: %zu floats\n", emb_sizes[i]);
    }
}

// Free buffers
for (int i = 0; i < 2; i++) {
    free(docs[i]);
    free(metas[i]);
    free(embs[i]);
}
```

**Example - Retrieve only documents (no metadata or embeddings):**
```c
int ids[] = {1, 2, 3};
char* docs[3];

for (int i = 0; i < 3; i++) {
    docs[i] = malloc(65536);
}

size_t doc_sizes[3] = {65536, 65536, 65536};

// Pass NULL for metadata and embeddings
int result = cactus_index_get(index, ids, 3,
                               docs, doc_sizes,
                               NULL, NULL,  // No metadata
                               NULL, NULL); // No embeddings

if (result == 0) {
    for (int i = 0; i < 3; i++) {
        printf("%s\n", docs[i]);
    }
}

for (int i = 0; i < 3; i++) {
    free(docs[i]);
}
```

**Example - Using sizeof() for buffer sizes:**
```c
int ids[] = {1};

// Static buffers
char doc_buffer[65536];
char meta_buffer[65536];
float emb_buffer[768];

// Buffer arrays
char* docs[1] = {doc_buffer};
char* metas[1] = {meta_buffer};
float* embs[1] = {emb_buffer};

// Use sizeof() for char buffers (bytes), calculate for float buffers
size_t doc_sizes[1] = {sizeof(doc_buffer)};   // bytes
size_t meta_sizes[1] = {sizeof(meta_buffer)}; // bytes
size_t emb_sizes[1] = {sizeof(emb_buffer) / sizeof(float)};  // number of floats

int result = cactus_index_get(index, ids, 1,
                               docs, doc_sizes,
                               metas, meta_sizes,
                               embs, emb_sizes);

if (result == 0) {
    printf("Document: %s\n", doc_buffer);
    printf("Retrieved %zu floats (%zu bytes) of embedding data\n",
           emb_sizes[0], emb_sizes[0] * sizeof(float));
}
```

### `cactus_index_query`
Performs batched similarity search using cosine similarity. Results are sorted by score (highest first).

```c
int cactus_index_query(
    cactus_index_t index,
    const float** embeddings,
    size_t embeddings_count,
    size_t embedding_dim,
    const char* options_json,
    int** id_buffers,
    size_t* id_buffer_sizes,
    float** score_buffers,
    size_t* score_buffer_sizes
);
```

**Parameters:**
- `index`: Index handle from `cactus_index_init` (required)
- `embeddings`: Array of query embedding pointers (required, none can be NULL)
- `embeddings_count`: Number of query embeddings (must be > 0)
- `embedding_dim`: Dimension of embeddings (must match index, must be > 0)
- `options_json`: JSON string with query options (optional, can be NULL for defaults)
- `id_buffers`: Array of pre-allocated buffers for result document IDs (required, one buffer per query)
- `id_buffer_sizes`: Array serving dual purpose (required):
  - **Input**: Maximum capacity of each buffer (number of results)
  - **Output**: Actual number of results per query
- `score_buffers`: Array of pre-allocated buffers for result scores (required, one buffer per query)
- `score_buffer_sizes`: Array serving dual purpose (required):
  - **Input**: Maximum capacity of each buffer (number of results)
  - **Output**: Actual number of results per query

**Options JSON Format:**
```json
{
    "top_k": 10,
    "score_threshold": 0.7
}
```

**Default Options:**
- `top_k`: 10 (maximum results per query)
- `score_threshold`: -1.0 (no filtering)

**Returns:** 0 on success, -1 on error

**Buffer Requirements:**
- Queries are executed first (all similarity computations complete), then buffer sizes are validated
- If any buffer is too small to hold results, the function returns -1 with no data copied
- Both `id_buffers` and `score_buffers` must have capacity for at least `top_k` results per query

**Example - Single query:**
```c
float query[768] = {0.1, 0.2, 0.3, /* ... */};
const float* queries[] = {query};

// Allocate result buffers (1 query, max 10 results)
int* ids[1];
float* scores[1];
ids[0] = (int*)malloc(10 * sizeof(int));
scores[0] = (float*)malloc(10 * sizeof(float));

size_t id_sizes[1] = {10};     // Input: buffer capacity
size_t score_sizes[1] = {10};  // Input: buffer capacity

const char* options = "{\"top_k\":10,\"score_threshold\":0.5}";
int result = cactus_index_query(index, queries, 1, 768, options,
                                 ids, id_sizes,
                                 scores, score_sizes);

if (result == 0) {
    printf("Found %zu results for query:\n", id_sizes[0]);  // Output: actual count
    for (size_t i = 0; i < id_sizes[0]; i++) {
        printf("  ID: %d, Score: %.3f\n", ids[0][i], scores[0][i]);
    }
}

// Free buffers
free(ids[0]);
free(scores[0]);
```

**Example - Batch queries:**
```c
float query1[768] = {/* ... */};
float query2[768] = {/* ... */};
const float* queries[] = {query1, query2};

// Allocate result buffers (2 queries, max 5 results each)
int* ids[2];
float* scores[2];
for (int i = 0; i < 2; i++) {
    ids[i] = (int*)malloc(5 * sizeof(int));
    scores[i] = (float*)malloc(5 * sizeof(float));
}

size_t id_sizes[2] = {5, 5};
size_t score_sizes[2] = {5, 5};

int result = cactus_index_query(index, queries, 2, 768, "{\"top_k\":5}",
                                 ids, id_sizes,
                                 scores, score_sizes);

if (result == 0) {
    for (int q = 0; q < 2; q++) {
        printf("Query %d: %zu results\n", q, id_sizes[q]);
        for (size_t i = 0; i < id_sizes[q]; i++) {
            printf("  ID: %d, Score: %.3f\n", ids[q][i], scores[q][i]);
        }
    }
}

// Free buffers
for (int i = 0; i < 2; i++) {
    free(ids[i]);
    free(scores[i]);
}
```

### `cactus_index_compact`
Removes deleted documents and reclaims disk space.

```c
int cactus_index_compact(cactus_index_t index);
```

**Parameters:**
- `index`: Index handle from `cactus_index_init` (required)

**Returns:** 0 on success, -1 on error

**Example:**
```c
int result = cactus_index_compact(index);
if (result != 0) {
    fprintf(stderr, "Compaction failed: %s\n", cactus_get_last_error());
}
```

### `cactus_index_destroy`
Releases all resources associated with the index.

```c
void cactus_index_destroy(cactus_index_t index);
```

**Important:** Always call this when done with an index to prevent memory leaks.

## Complete Examples

### Creating and Populating an Index
```c
cactus_index_t index = cactus_index_init("./my_index", 768);
if (!index) {
    fprintf(stderr, "Failed to initialize index\n");
    return -1;
}

int ids[] = {1, 2};
const char* docs[] = {
    "AI is transforming technology",
    "Machine learning enables predictions"
};
const char* metas[] = {"{\"source\":\"wiki\"}", "{\"source\":\"blog\"}"};

float emb1[768] = {0.1, 0.2, 0.3, /* ... */};
float emb2[768] = {0.4, 0.5, 0.6, /* ... */};
const float* embeddings[] = {emb1, emb2};

int result = cactus_index_add(index, ids, docs, metas, embeddings, 2, 768);
if (result != 0) {
    fprintf(stderr, "Failed to add documents: %s\n", cactus_get_last_error());
}

cactus_index_destroy(index);
```

### Similarity Search
```c
cactus_index_t index = cactus_index_init("./my_index", 768);
if (!index) {
    fprintf(stderr, "Failed to open index: %s\n", cactus_get_last_error());
    return -1;
}

float query_embedding[768] = {0.1, 0.2, 0.3, /* ... */};
const float* queries[] = {query_embedding};

// Allocate buffers for 1 query with max 10 results
int* ids[1];
float* scores[1];
ids[0] = (int*)malloc(10 * sizeof(int));
scores[0] = (float*)malloc(10 * sizeof(float));

size_t id_sizes[1] = {10};
size_t score_sizes[1] = {10};

int result = cactus_index_query(index, queries, 1, 768,
                                 "{\"top_k\":10,\"score_threshold\":0.7}",
                                 ids, id_sizes,
                                 scores, score_sizes);

if (result == 0) {
    printf("Found %zu results:\n", id_sizes[0]);
    for (size_t i = 0; i < id_sizes[0]; i++) {
        printf("  ID: %d, Score: %.3f\n", ids[0][i], scores[0][i]);
    }
}

free(ids[0]);
free(scores[0]);

cactus_index_destroy(index);
```

### RAG (Retrieval-Augmented Generation)
```c
cactus_index_t index = cactus_index_init("./my_index", 768);
if (!index) return -1;

// Search for relevant documents
float query_embedding[768] = {0.1, 0.2, 0.3, /* ... */};
const float* queries[] = {query_embedding};

// Allocate query result buffers
int* result_ids[1];
float* result_scores[1];
result_ids[0] = (int*)malloc(10 * sizeof(int));
result_scores[0] = (float*)malloc(10 * sizeof(float));

size_t id_sizes[1] = {10};
size_t score_sizes[1] = {10};

int query_result = cactus_index_query(index, queries, 1, 768,
                                       "{\"top_k\":3,\"score_threshold\":0.5}",
                                       result_ids, id_sizes,
                                       result_scores, score_sizes);

if (query_result == 0) {
    // Allocate buffers for retrieving documents
    size_t num_results = id_sizes[0];
    char* docs[10];

    for (size_t i = 0; i < num_results; i++) {
        docs[i] = (char*)malloc(65536);
    }

    size_t doc_sizes[10];
    for (size_t i = 0; i < num_results; i++) {
        doc_sizes[i] = 65536;
    }

    // Retrieve only documents (no metadata or embeddings needed for RAG)
    int get_result = cactus_index_get(index, result_ids[0], num_results,
                                       docs, doc_sizes,
                                       NULL, NULL,  // No metadata
                                       NULL, NULL); // No embeddings

    if (get_result == 0) {
        // Build context from retrieved documents
        char context[32768] = "";
        for (size_t i = 0; i < num_results; i++) {
            strcat(context, docs[i]);
            strcat(context, "\n\n");
        }

        printf("Context: %s\n", context);
    }

    // Free buffers
    for (size_t i = 0; i < num_results; i++) {
        free(docs[i]);
    }
}

free(result_ids[0]);
free(result_scores[0]);

cactus_index_destroy(index);
```

### Deleting Documents and Compacting
```c
cactus_index_t index = cactus_index_init("./my_index", 768);
if (!index) return -1;

int ids[] = {1, 3, 5};
int deleted = cactus_index_delete(index, ids, 3);
if (deleted == 0) {
    printf("Documents deleted successfully\n");
}

// Compact to reclaim space
int compact_result = cactus_index_compact(index);
if (compact_result == 0) {
    printf("Index compacted successfully\n");
}

cactus_index_destroy(index);
```

### Migrating to a Different Embedding Model
```c
// Open old index and create new index with different dimensions
cactus_index_t old_index = cactus_index_init("./old_index", 768);
cactus_index_t new_index = cactus_index_init("./new_index", 1536);

if (!old_index || !new_index) {
    fprintf(stderr, "Failed to open indexes\n");
    return -1;
}

// Get all documents from old index
int all_doc_ids[] = {1, 2, 3, 4, 5};
int num_docs = 5;

// Allocate buffers for old documents
char* old_docs[5];
char* old_metas[5];

for (int i = 0; i < num_docs; i++) {
    old_docs[i] = malloc(65536);
    old_metas[i] = malloc(65536);
}

size_t doc_sizes[5], meta_sizes[5];
for (int i = 0; i < num_docs; i++) {
    doc_sizes[i] = 65536;
    meta_sizes[i] = 65536;
}

int get_result = cactus_index_get(old_index, all_doc_ids, num_docs,
                                   old_docs, doc_sizes,
                                   old_metas, meta_sizes,
                                   NULL, NULL);  // Don't need old embeddings

if (get_result == 0) {
    // Regenerate embeddings with new model (1536 dimensions)
    float new_embs[5][1536];
    const float* new_emb_ptrs[5];

    for (int i = 0; i < num_docs; i++) {
        // Generate new embedding for old_docs[i]
        // ... embedding generation code ...
        new_emb_ptrs[i] = new_embs[i];
    }

    // Add to new index
    const char* doc_ptrs[5];
    const char* meta_ptrs[5];
    for (int i = 0; i < num_docs; i++) {
        doc_ptrs[i] = old_docs[i];
        meta_ptrs[i] = old_metas[i];
    }

    cactus_index_add(new_index, all_doc_ids, doc_ptrs, meta_ptrs,
                     new_emb_ptrs, num_docs, 1536);
}

// Free buffers
for (int i = 0; i < num_docs; i++) {
    free(old_docs[i]);
    free(old_metas[i]);
}

cactus_index_destroy(old_index);
cactus_index_destroy(new_index);
```

## Best Practices

1. **Always Check Return Values**: Most functions return 0 on success, -1 on error
2. **Buffer Management**:
   - Allocate enough bytes for document/metadata buffers
   - Allocate `embedding_dim * sizeof(float)` bytes for embedding buffers
   - For `cactus_index_get`:
     - `document_buffer_sizes` and `metadata_buffer_sizes`: capacity/size in bytes
     - `embedding_buffer_sizes`: capacity/size in number of floats (not bytes)
   - You can use `sizeof(buffer)` for char buffers, use `sizeof(buffer) / sizeof(float)` for float buffers
3. **Memory Management**:
   - Always call `cactus_index_destroy()` when done to prevent memory leaks
   - Free all buffers you allocate
4. **Optional Buffers**:
   - In `cactus_index_get`, you can pass NULL for buffers you don't need
   - This allows flexible retrieval of only the data you need
5. **Thread Safety**: Each index instance should be used from a single thread
6. **Batch Operations**: Add 100-1000 documents per call for best performance
7. **Error Handling**: Always check return values and use `cactus_get_last_error()` for detailed error messages

## Error Handling

All functions return 0 on success, -1 on error (except init functions which return NULL on error).

Common errors: Invalid path, insufficient buffer, dimension mismatch, duplicate IDs, file I/O errors, NULL parameters

**Example:**
```c
cactus_index_t index = cactus_index_init("./index", 768);
if (!index) {
    const char* error = cactus_get_last_error();
    fprintf(stderr, "Error: %s\n", error);
    return -1;
}

int ids[] = {1, 2};
const char* docs[] = {"doc1", "doc2"};
const char* metas[] = {"{}", "{}"};
float emb1[768] = {/* ... */};
float emb2[768] = {/* ... */};
const float* embs[] = {emb1, emb2};

int result = cactus_index_add(index, ids, docs, metas, embs, 2, 768);
if (result != 0) {
    const char* error = cactus_get_last_error();
    fprintf(stderr, "Add failed: %s\n", error);
}

cactus_index_destroy(index);
```
