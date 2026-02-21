import Foundation
import Combine

class TestRunner: ObservableObject {
    @Published var logs: [String] = []
    @Published var finished = false

    func run(modelPath: String, transcribeModelPath: String, assetsPath: String) {
        DispatchQueue.global(qos: .userInitiated).async {
            self.log("=== Cactus Swift Wrapper Test ===")
            self.log("Model:            \(modelPath.isEmpty ? "(none - set CACTUS_TEST_MODEL)" : modelPath)")
            self.log("Transcribe model: \(transcribeModelPath.isEmpty ? "(none)" : transcribeModelPath)")
            self.log("Assets:           \(assetsPath.isEmpty ? "(none)" : assetsPath)")

            guard !modelPath.isEmpty else {
                self.log("[FAIL] No model path provided")
                self.finish()
                return
            }

            // ---------------------------------------------------------------
            self.log("\n--- Test 1: Init ---")
            var model: Cactus?
            do {
                model = try Cactus(modelPath: modelPath)
                self.log("[PASS] Model initialized")
            } catch {
                self.log("[FAIL] \(error)")
                self.finish()
                return
            }

            // ---------------------------------------------------------------
            self.log("\n--- Test 2: Basic Completion ---")
            do {
                let result = try model!.complete(
                    messages: [.user("Say hello in exactly 3 words.")],
                    options: .init(maxTokens: 20)
                )
                let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                self.log("[PASS] \"\(text)\"")
                self.log("       prompt=\(result.promptTokens) completion=\(result.completionTokens) decode=\(String(format: "%.1f", result.decodeTokensPerSecond)) tok/s")
            } catch {
                self.log("[FAIL] \(error)")
            }

            // ---------------------------------------------------------------
            self.log("\n--- Test 3: Streaming Completion ---")
            do {
                var tokenCount = 0
                let result = try model!.complete(
                    messages: [.user("Count from 1 to 5.")],
                    options: .init(maxTokens: 40),
                    onToken: { _, _ in tokenCount += 1 }
                )
                if tokenCount > 0 {
                    self.log("[PASS] Streamed \(tokenCount) tokens: \"\(result.text.trimmingCharacters(in: .whitespacesAndNewlines))\"")
                } else {
                    self.log("[FAIL] onToken callback never fired")
                }
            } catch {
                self.log("[FAIL] \(error)")
            }

            // ---------------------------------------------------------------
            self.log("\n--- Test 4: Tool Calling ---")
            do {
                let tools: [[String: Any]] = [[
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string", "description": "City name"]
                        ],
                        "required": ["location"]
                    ] as [String: Any]
                ]]
                let result = try model!.complete(
                    messages: [.user("What is the weather in Paris? Use the get_weather tool.")],
                    options: .init(maxTokens: 80),
                    tools: tools
                )
                if let calls = result.functionCalls, !calls.isEmpty {
                    let name = calls.first?["name"] as? String ?? "?"
                    let args = calls.first?["arguments"] as? [String: Any] ?? [:]
                    self.log("[PASS] Tool call: \(name)(\(args))")
                } else {
                    self.log("[SKIP] No tool call produced (model may not support function calling)")
                }
            } catch {
                self.log("[FAIL] \(error)")
            }

            // ---------------------------------------------------------------
            self.log("\n--- Test 5: Tokenization ---")
            do {
                let tokens = try model!.tokenize("Hello, world!")
                if tokens.count > 0 {
                    self.log("[PASS] \(tokens.count) tokens")
                } else {
                    self.log("[FAIL] Zero tokens returned")
                }
            } catch {
                self.log("[FAIL] \(error)")
            }

            // ---------------------------------------------------------------
            self.log("\n--- Test 6: Embeddings ---")
            do {
                let embeddings = try model!.embed(text: "Hello world")
                if embeddings.count > 0 {
                    self.log("[PASS] dim=\(embeddings.count) norm=\(String(format: "%.4f", Self.norm(embeddings)))")
                } else {
                    self.log("[SKIP] Empty embeddings (model may not support)")
                }
            } catch {
                self.log("[SKIP] Not supported by this model: \(error)")
            }

            // ---------------------------------------------------------------
            let audioPath = assetsPath.isEmpty ? "" : (assetsPath + "/test.wav")
            let audioExists = !audioPath.isEmpty && FileManager.default.fileExists(atPath: audioPath)

            if !transcribeModelPath.isEmpty {
                var transcribeModel: Cactus?
                do {
                    transcribeModel = try Cactus(modelPath: transcribeModelPath)
                } catch {
                    self.log("\n--- Test 7: Transcription --- [FAIL] Could not init transcribe model: \(error)")
                    self.log("--- Test 8: VAD --- [SKIP]")
                    self.log("\n=== Done ===")
                    self.finish()
                    return
                }

                self.log("\n--- Test 7: Transcription ---")
                if audioExists {
                    do {
                        let modelLower = transcribeModelPath.lowercased()
                        let prompt = modelLower.contains("whisper")
                            ? "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
                            : nil
                        let result = try transcribeModel!.transcribe(audioPath: audioPath, prompt: prompt)
                        self.log("[PASS] \"\(result.text.trimmingCharacters(in: .whitespacesAndNewlines))\"")
                        self.log("       time=\(String(format: "%.0f", result.totalTime))ms")
                    } catch {
                        self.log("[FAIL] \(error)")
                    }
                } else {
                    self.log("[SKIP] test.wav not found at \(audioPath)")
                }

                self.log("\n--- Test 8: VAD ---")
                if audioExists {
                    do {
                        let result = try transcribeModel!.vad(audioPath: audioPath)
                        self.log("[PASS] \(result.segments.count) speech segment(s)")
                        for seg in result.segments {
                            self.log("       segment: \(seg.start)ms – \(seg.end)ms")
                        }
                    } catch {
                        self.log("[FAIL] \(error)")
                    }
                } else {
                    self.log("[SKIP] test.wav not found")
                }
            } else {
                self.log("\n--- Test 7: Transcription --- [SKIP] (no CACTUS_TEST_TRANSCRIBE_MODEL)")
                self.log("--- Test 8: VAD --- [SKIP] (no CACTUS_TEST_TRANSCRIBE_MODEL)")
            }

            self.log("\n=== Done ===")
            self.finish()
        }
    }

    private static func norm(_ v: [Float]) -> Float {
        v.reduce(0) { $0 + $1 * $1 }.squareRoot()
    }

    private func log(_ message: String) {
        print(message)
        DispatchQueue.main.async { self.logs.append(message) }
    }

    private func finish() {
        DispatchQueue.main.async {
            self.finished = true
            exit(0)
        }
    }
}
