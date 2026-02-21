import 'dart:io';
import 'package:flutter/material.dart';
import 'cactus.dart';

void main() {
  runApp(const CactusTestApp());
}

class CactusTestApp extends StatelessWidget {
  const CactusTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus Flutter Test',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal)),
      home: const TestPage(),
    );
  }
}

class TestPage extends StatefulWidget {
  const TestPage({super.key});

  @override
  State<TestPage> createState() => _TestPageState();
}

class _TestPageState extends State<TestPage> {
  final List<String> _logs = [];
  bool _finished = false;
  final ScrollController _scroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _runTests();
  }

  void _log(String message) {
    debugPrint(message);
    setState(() => _logs.add(message));
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.jumpTo(_scroll.position.maxScrollExtent);
      }
    });
  }

  Future<void> _runTests() async {
    final modelPath = Platform.environment['CACTUS_TEST_MODEL']
        ?? const String.fromEnvironment('CACTUS_TEST_MODEL');
    final transcribePath = Platform.environment['CACTUS_TEST_TRANSCRIBE_MODEL']
        ?? const String.fromEnvironment('CACTUS_TEST_TRANSCRIBE_MODEL');
    final assetsPath = Platform.environment['CACTUS_TEST_ASSETS']
        ?? const String.fromEnvironment('CACTUS_TEST_ASSETS');
    final audioPath = assetsPath.isEmpty ? '' : '$assetsPath/test.wav';

    _log('=== Cactus Flutter Wrapper Test ===');
    _log('Model:            ${modelPath.isEmpty ? "(none - set CACTUS_TEST_MODEL)" : modelPath}');
    _log('Transcribe model: ${transcribePath.isEmpty ? "(none)" : transcribePath}');
    _log('Assets:           ${assetsPath.isEmpty ? "(none)" : assetsPath}');

    if (modelPath.isEmpty) {
      _log('[FAIL] No model path provided');
      setState(() => _finished = true);
      exit(1);
    }

    // ---------------------------------------------------------------
    _log('\n--- Test 1: Init ---');
    Cactus? model;
    try {
      model = Cactus.create(modelPath);
      _log('[PASS] Model initialized');
    } catch (e) {
      _log('[FAIL] $e');
      setState(() => _finished = true);
      exit(1);
    }

    // ---------------------------------------------------------------
    _log('\n--- Test 2: Basic Completion ---');
    try {
      final result = model.completeMessages(
        [Message.user('Say hello in exactly 3 words.')],
        options: const CompletionOptions(maxTokens: 20),
      );
      final text = result.text.trim();
      _log('[PASS] "$text"');
      _log('       prompt=${result.promptTokens} completion=${result.completionTokens} decode=${result.decodeTokensPerSecond.toStringAsFixed(1)} tok/s');
    } catch (e) {
      _log('[FAIL] $e');
    }

    // ---------------------------------------------------------------
    _log('\n--- Test 3: Streaming Completion ---');
    try {
      var tokenCount = 0;
      final result = model.completeMessages(
        [Message.user('Count from 1 to 5.')],
        options: const CompletionOptions(maxTokens: 40),
        onToken: (_, __) => tokenCount++,
      );
      if (tokenCount > 0) {
        _log('[PASS] Streamed $tokenCount tokens: "${result.text.trim()}"');
      } else {
        _log('[FAIL] onToken callback never fired');
      }
    } catch (e) {
      _log('[FAIL] $e');
    }

    // ---------------------------------------------------------------
    _log('\n--- Test 4: Tool Calling ---');
    try {
      final tools = [
        {
          'name': 'get_weather',
          'description': 'Get the current weather for a location',
          'parameters': {
            'type': 'object',
            'properties': {
              'location': {'type': 'string', 'description': 'City name'}
            },
            'required': ['location']
          }
        }
      ];
      final result = model.completeMessages(
        [Message.user('What is the weather in Paris? Use the get_weather tool.')],
        options: const CompletionOptions(maxTokens: 80),
        tools: tools,
      );
      final calls = result.functionCalls;
      if (calls != null && calls.isNotEmpty) {
        final name = calls.first['name'] ?? '?';
        final args = calls.first['arguments'] ?? {};
        _log('[PASS] Tool call: $name($args)');
      } else {
        _log('[SKIP] No tool call produced (model may not support function calling)');
      }
    } catch (e) {
      _log('[FAIL] $e');
    }

    // ---------------------------------------------------------------
    _log('\n--- Test 5: Tokenization ---');
    try {
      final tokens = model.tokenize('Hello, world!');
      if (tokens.isNotEmpty) {
        _log('[PASS] ${tokens.length} tokens');
      } else {
        _log('[FAIL] Zero tokens returned');
      }
    } catch (e) {
      _log('[FAIL] $e');
    }

    // ---------------------------------------------------------------
    _log('\n--- Test 6: Embeddings ---');
    try {
      final embeddings = model.embed('Hello world');
      if (embeddings.isNotEmpty) {
        final norm = embeddings.fold(0.0, (s, x) => s + x * x);
        _log('[PASS] dim=${embeddings.length} norm=${norm.toStringAsFixed(4)}');
      } else {
        _log('[SKIP] Empty embeddings (model may not support)');
      }
    } catch (e) {
      _log('[SKIP] Not supported by this model: $e');
    }

    // ---------------------------------------------------------------
    if (transcribePath.isNotEmpty) {
      Cactus? transcribeModel;
      try {
        transcribeModel = Cactus.create(transcribePath);
      } catch (e) {
        _log('\n--- Test 7: Transcription --- [FAIL] Could not init transcribe model: $e');
        _log('--- Test 8: VAD --- [SKIP]');
        _log('\n=== Done ===');
        setState(() => _finished = true);
        exit(1);
      }

      _log('\n--- Test 7: Transcription ---');
      final audioExists = audioPath.isNotEmpty && File(audioPath).existsSync();
      if (audioExists) {
        try {
          final isWhisper = transcribePath.toLowerCase().contains('whisper');
          final prompt = isWhisper
              ? '<|startoftranscript|><|en|><|transcribe|><|notimestamps|>'
              : null;
          final result = transcribeModel.transcribe(audioPath, prompt: prompt);
          _log('[PASS] "${result.text.trim()}"');
          _log('       time=${result.totalTime.toStringAsFixed(0)}ms');
        } catch (e) {
          _log('[FAIL] $e');
        }
      } else {
        _log('[SKIP] test.wav not found at $audioPath');
      }

      _log('\n--- Test 8: VAD ---');
      if (audioExists) {
        try {
          final result = transcribeModel.vad(audioPath);
          _log('[PASS] ${result.segments.length} speech segment(s)');
          for (final seg in result.segments) {
            _log('       segment: ${seg.start}ms – ${seg.end}ms');
          }
        } catch (e) {
          _log('[FAIL] $e');
        }
      } else {
        _log('[SKIP] test.wav not found');
      }

      transcribeModel.dispose();
    } else {
      _log('\n--- Test 7: Transcription --- [SKIP] (no CACTUS_TEST_TRANSCRIBE_MODEL)');
      _log('--- Test 8: VAD --- [SKIP] (no CACTUS_TEST_TRANSCRIBE_MODEL)');
    }

    model.dispose();
    _log('\n=== Done ===');
    setState(() => _finished = true);
    exit(0);
  }

  Color _colorFor(String log) {
    if (log.startsWith('[PASS]')) return Colors.green;
    if (log.startsWith('[FAIL]')) return Colors.red;
    if (log.startsWith('[SKIP]')) return Colors.orange;
    if (log.startsWith('===')) return Colors.black;
    return Colors.grey.shade700;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cactus Flutter Test'),
        actions: [
          if (!_finished)
            const Padding(
              padding: EdgeInsets.all(16),
              child: SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
        ],
      ),
      body: ListView.builder(
        controller: _scroll,
        padding: const EdgeInsets.all(12),
        itemCount: _logs.length,
        itemBuilder: (_, i) => Text(
          _logs[i],
          style: TextStyle(
            fontFamily: 'monospace',
            fontSize: 12,
            color: _colorFor(_logs[i]),
          ),
        ),
      ),
    );
  }
}
