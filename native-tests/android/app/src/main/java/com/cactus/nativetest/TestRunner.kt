package com.cactus.nativetest

import android.graphics.Color
import android.os.Handler
import android.os.Looper
import com.cactus.Cactus
import com.cactus.CompletionOptions
import com.cactus.Message
import java.io.File
import kotlin.math.sqrt

class TestRunner(
    private val onLog: (String, Int) -> Unit,
    private val onDone: () -> Unit
) {
    private val mainHandler = Handler(Looper.getMainLooper())

    fun run(modelPath: String, transcribeModelPath: String, assetsPath: String) {
        Thread { runTests(modelPath, transcribeModelPath, assetsPath) }.start()
    }

    private fun runTests(modelPath: String, transcribeModelPath: String, assetsPath: String) {
        log("=== Cactus Kotlin Wrapper Test ===")
        log("Model:            ${if (modelPath.isEmpty()) "(none - pass MODEL_PATH intent extra)" else modelPath}")
        log("Transcribe model: ${if (transcribeModelPath.isEmpty()) "(none)" else transcribeModelPath}")
        log("Assets:           ${if (assetsPath.isEmpty()) "(none)" else assetsPath}")

        if (modelPath.isEmpty()) {
            log("[FAIL] No model path provided")
            done(); return
        }

        // ---------------------------------------------------------------
        log("\n--- Test 1: Init ---")
        val model = try {
            Cactus.create(modelPath).also { log("[PASS] Model initialized") }
        } catch (e: Exception) {
            log("[FAIL] $e"); done(); return
        }

        // ---------------------------------------------------------------
        log("\n--- Test 2: Basic Completion ---")
        try {
            val result = model.complete(
                listOf(Message.user("Say hello in exactly 3 words.")),
                CompletionOptions(maxTokens = 20)
            )
            val text = result.text.trim()
            log("[PASS] \"$text\"")
            log("       prompt=${result.promptTokens} completion=${result.completionTokens} decode=${"%.1f".format(result.decodeTokensPerSecond)} tok/s")
        } catch (e: Exception) {
            log("[FAIL] $e")
        }

        // ---------------------------------------------------------------
        log("\n--- Test 3: Streaming Completion ---")
        try {
            var tokenCount = 0
            val result = model.complete(
                listOf(Message.user("Count from 1 to 5.")),
                CompletionOptions(maxTokens = 40),
                callback = { _, _ -> tokenCount++ }
            )
            if (tokenCount > 0) {
                log("[PASS] Streamed $tokenCount tokens: \"${result.text.trim()}\"")
            } else {
                log("[FAIL] onToken callback never fired")
            }
        } catch (e: Exception) {
            log("[FAIL] $e")
        }

        // ---------------------------------------------------------------
        log("\n--- Test 4: Tool Calling ---")
        try {
            val tools = listOf(mapOf(
                "name" to "get_weather",
                "description" to "Get the current weather for a location",
                "parameters" to mapOf(
                    "type" to "object",
                    "properties" to mapOf(
                        "location" to mapOf("type" to "string", "description" to "City name")
                    ),
                    "required" to listOf("location")
                )
            ))
            val result = model.complete(
                listOf(Message.user("What is the weather in Paris? Use the get_weather tool.")),
                CompletionOptions(maxTokens = 80),
                tools = tools
            )
            val calls = result.functionCalls
            if (!calls.isNullOrEmpty()) {
                val name = calls.first()["name"] as? String ?: "?"
                val args = calls.first()["arguments"] ?: emptyMap<String, Any>()
                log("[PASS] Tool call: $name($args)")
            } else {
                log("[SKIP] No tool call produced (model may not support function calling)")
            }
        } catch (e: Exception) {
            log("[FAIL] $e")
        }

        // ---------------------------------------------------------------
        log("\n--- Test 5: Tokenization ---")
        try {
            val tokens = model.tokenize("Hello, world!")
            if (tokens.isNotEmpty()) {
                log("[PASS] ${tokens.size} tokens")
            } else {
                log("[FAIL] Zero tokens returned")
            }
        } catch (e: Exception) {
            log("[FAIL] $e")
        }

        // ---------------------------------------------------------------
        log("\n--- Test 6: Embeddings ---")
        try {
            val embeddings = model.embed("Hello world")
            if (embeddings.isNotEmpty()) {
                val norm = sqrt(embeddings.fold(0f) { s, x -> s + x * x }).toDouble()
                log("[PASS] dim=${embeddings.size} norm=${"%.4f".format(norm)}")
            } else {
                log("[SKIP] Empty embeddings (model may not support)")
            }
        } catch (e: Exception) {
            log("[SKIP] Not supported by this model: $e")
        }

        // ---------------------------------------------------------------
        val audioPath = if (assetsPath.isEmpty()) "" else "$assetsPath/test.wav"
        val audioExists = audioPath.isNotEmpty() && File(audioPath).exists()

        if (transcribeModelPath.isNotEmpty()) {
            val transcribeModel = try {
                Cactus.create(transcribeModelPath)
            } catch (e: Exception) {
                log("\n--- Test 7: Transcription --- [FAIL] Could not init transcribe model: $e")
                log("--- Test 8: VAD --- [SKIP]")
                log("\n=== Done ===")
                model.close(); done(); return
            }

            log("\n--- Test 7: Transcription ---")
            if (audioExists) {
                try {
                    val isWhisper = transcribeModelPath.lowercase().contains("whisper")
                    val prompt = if (isWhisper) "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" else null
                    val result = transcribeModel.transcribe(audioPath, prompt = prompt)
                    log("[PASS] \"${result.text.trim()}\"")
                    log("       time=${"%.0f".format(result.totalTime)}ms")
                } catch (e: Exception) {
                    log("[FAIL] $e")
                }
            } else {
                log("[SKIP] test.wav not found at $audioPath")
            }

            log("\n--- Test 8: VAD ---")
            if (audioExists) {
                try {
                    val result = transcribeModel.vad(audioPath)
                    log("[PASS] ${result.segments.size} speech segment(s)")
                    for (seg in result.segments) {
                        log("       segment: ${seg.start}ms – ${seg.end}ms")
                    }
                } catch (e: Exception) {
                    log("[FAIL] $e")
                }
            } else {
                log("[SKIP] test.wav not found")
            }

            transcribeModel.close()
        } else {
            log("\n--- Test 7: Transcription --- [SKIP] (no TRANSCRIBE_PATH intent extra)")
            log("--- Test 8: VAD --- [SKIP] (no TRANSCRIBE_PATH intent extra)")
        }

        model.close()
        log("\n=== Done ===")
        done()
    }

    private fun log(message: String) {
        println(message)
        mainHandler.post { onLog(message, colorFor(message)) }
    }

    private fun done() {
        mainHandler.post { onDone() }
    }

    private fun colorFor(log: String) = when {
        log.startsWith("[PASS]") -> Color.parseColor("#4CAF50")
        log.startsWith("[FAIL]") -> Color.parseColor("#F44336")
        log.startsWith("[SKIP]") -> Color.parseColor("#FF9800")
        log.startsWith("===") -> Color.BLACK
        else -> Color.DKGRAY
    }
}
