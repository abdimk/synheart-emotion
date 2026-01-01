import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:synheart_emotion/synheart_emotion.dart';

void main() {
  runApp(const EmotionExampleApp());
}

class EmotionExampleApp extends StatelessWidget {
  const EmotionExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Synheart Emotion Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const EmotionDemoPage(),
    );
  }
}

class EmotionDemoPage extends StatefulWidget {
  const EmotionDemoPage({super.key});

  @override
  State<EmotionDemoPage> createState() => _EmotionDemoPageState();
}

class _EmotionDemoPageState extends State<EmotionDemoPage> {
  EmotionEngine? _engine;
  Timer? _dataTimer;
  Timer? _inferenceTimer;
  
  final List<EmotionResult> _results = [];
  final List<String> _logs = [];
  
  bool _isRunning = false;
  bool _isInitialized = false;
  String _currentEmotion = 'Unknown';
  double _currentConfidence = 0.0;
  Map<String, double> _currentProbabilities = {};
  String? _currentModelPath;
  Duration _currentStepSize = const Duration(seconds: 5);
  
  // Available models from assets
  final List<Map<String, String>> _availableModels = [
    {
      'name': 'ExtraTrees 60s/5s',
      'path': 'assets/ml/ExtraTrees_60_5_nozipmap.onnx',
      'description': '60s window, 5s step',
    },
    {
      'name': 'ExtraTrees 120s/5s',
      'path': 'assets/ml/ExtraTrees_120_5_nozipmap.onnx',
      'description': '120s window, 5s step',
    },
    {
      'name': 'ExtraTrees 120s/60s',
      'path': 'assets/ml/ExtraTrees_120_60_nozipmap.onnx',
      'description': '120s window, 60s step',
    },
  ];

  @override
  void initState() {
    super.initState();
    // Load default model
    _loadModel('assets/ml/ExtraTrees_120_60_nozipmap.onnx');
  }

  Future<void> _loadModel(String modelPath) async {
    setState(() {
      _isInitialized = false;
      _logs.add('[INFO] Loading model: $modelPath');
    });

    try {
      // Determine window and step size from model path
      int windowSize = 120;
      int stepSize = 60;
      
      if (modelPath.contains('120_5')) {
        windowSize = 120;
        stepSize = 5;
      } else if (modelPath.contains('120_60')) {
        windowSize = 120;
        stepSize = 60;
      } else if (modelPath.contains('60_5')) {
        windowSize = 60;
        stepSize = 5;
      }

      // Load ONNX model
      final onnxModel = await OnnxEmotionModel.loadFromAsset(
        modelAssetPath: modelPath,
      );
      
      // Create config with appropriate window and step sizes
      final config = EmotionConfig(
        window: Duration(seconds: windowSize),
        step: Duration(seconds: stepSize),
        modelId: onnxModel.modelId,
      );
      
      // Create engine with loaded model
      _engine = EmotionEngine.fromPretrained(
        config,
        model: onnxModel,
        onLog: (level, message, {context}) {
          setState(() {
            _logs.add('[$level.toUpperCase()] $message');
            if (_logs.length > 50) {
              _logs.removeAt(0);
            }
          });
        },
      );
      
      setState(() {
        _isInitialized = true;
        _currentModelPath = modelPath;
        _currentStepSize = Duration(seconds: stepSize);
        _logs.add('[INFO] Engine initialized with 14-feature extraction');
        _logs.add('[INFO] Model: ${modelPath.split('/').last}');
        _logs.add('[INFO] Window: ${windowSize}s, Step: ${stepSize}s');
      });
    } catch (e) {
      setState(() {
        _isInitialized = false;
        _currentModelPath = null;
        _logs.add('[ERROR] Failed to load model: $e');
      });
    }
  }

  Future<void> _showModelSelectionDialog() async {
    final selectedModel = await showDialog<String>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Select Model'),
          content: SizedBox(
            width: double.maxFinite,
            child: ListView(
              shrinkWrap: true,
              children: [
                ..._availableModels.map((model) {
                  return ListTile(
                    title: Text(model['name']!),
                    subtitle: Text(model['description']!),
                    trailing: _currentModelPath == model['path']
                        ? const Icon(Icons.check, color: Colors.green)
                        : null,
                    onTap: () {
                      Navigator.of(context).pop(model['path']);
                    },
                  );
                }),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
          ],
        );
      },
    );

    if (selectedModel != null) {
      await _loadModel(selectedModel);
    }
  }

  void _startSimulation() {
    if (_isRunning || !_isInitialized || _engine == null) return;
    
    setState(() {
      _isRunning = true;
      _results.clear();
    });

    // Simulate data every 500ms
    _dataTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
      _simulateDataPoint();
    });

    // Run inference based on step size
    _inferenceTimer = Timer.periodic(_currentStepSize, (_) {
      _runInference();
    });
  }

  void _stopSimulation() {
    _dataTimer?.cancel();
    _inferenceTimer?.cancel();
    
    setState(() {
      _isRunning = false;
    });
  }

  void _simulateDataPoint() {
    final random = Random();
    
    // Simulate realistic HR and RR intervals
    final baseHr = 70 + (random.nextDouble() - 0.5) * 20; // ~70 BPM ± 10
    final hr = baseHr.clamp(50.0, 120.0);
    
    // Generate RR intervals (time between heartbeats in ms)
    final rrIntervals = <double>[];
    for (int i = 0; i < 60; i++) {
      final baseRr = 60000 / hr; // Convert HR to RR
      final rr = baseRr + (random.nextDouble() - 0.5) * 40; // Add some variability
      rrIntervals.add(rr.clamp(400.0, 1200.0));
    }

    _engine!.push(
      hr: hr,
      rrIntervalsMs: rrIntervals,
      timestamp: DateTime.now().toUtc(),
    );
  }

  void _runInference() async {
    if (_engine == null) return;
    
    try {
      final results = await _engine!.consumeReadyAsync();
      
      for (final result in results) {
        setState(() {
          _results.add(result);
          _currentEmotion = result.emotion;
          _currentConfidence = result.confidence;
          _currentProbabilities = result.probabilities;
          _logs.add('[INFO] Inference: ${result.emotion} (${(result.confidence * 100).toStringAsFixed(1)}%)');
          if (_logs.length > 50) {
            _logs.removeAt(0);
          }
        });
      }
    } catch (e) {
      setState(() {
        _logs.add('[ERROR] Inference failed: $e');
        if (_logs.length > 50) {
          _logs.removeAt(0);
        }
      });
    }
  }

  void _clearResults() {
    setState(() {
      _results.clear();
      _logs.clear();
      _currentEmotion = 'Unknown';
      _currentConfidence = 0.0;
      _currentProbabilities.clear();
    });
    _engine?.clear();
  }

  @override
  void dispose() {
    _stopSimulation();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Synheart Emotion Demo'),
        actions: [
          IconButton(
            icon: const Icon(Icons.model_training),
            onPressed: _showModelSelectionDialog,
            tooltip: 'Select Model',
          ),
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: _showInfo,
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status indicator
            if (!_isInitialized)
              Card(
                color: Colors.orange[100],
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text(
                        'No model loaded',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 8),
                      ElevatedButton.icon(
                        onPressed: _showModelSelectionDialog,
                        icon: const Icon(Icons.model_training),
                        label: const Text('Select Model'),
                      ),
                    ],
                  ),
                ),
              ),
            
            // Current model indicator
            if (_isInitialized && _currentModelPath != null)
              Card(
                color: Colors.green[50],
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Model Loaded',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Colors.green,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              _currentModelPath!.split('/').last,
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.swap_horiz),
                        onPressed: _showModelSelectionDialog,
                        tooltip: 'Change Model',
                      ),
                    ],
                  ),
                ),
              ),
            
            // Control buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: (_isRunning || !_isInitialized) ? null : _startSimulation,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Start'),
                ),
                ElevatedButton.icon(
                  onPressed: _isRunning ? _stopSimulation : null,
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop'),
                ),
                ElevatedButton.icon(
                  onPressed: _clearResults,
                  icon: const Icon(Icons.clear),
                  label: const Text('Clear'),
                ),
              ],
            ),
            
            const SizedBox(height: 20),
            
            // Current emotion display
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Text(
                      'Current Emotion',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _currentEmotion,
                      style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                        color: _getEmotionColor(_currentEmotion),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      '${(_currentConfidence * 100).toStringAsFixed(1)}% confidence',
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Probability bars
            if (_currentProbabilities.isNotEmpty) ...[
              Text(
                'Probabilities',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              ..._currentProbabilities.entries.map((entry) {
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2.0),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 80,
                        child: Text(entry.key),
                      ),
                      Expanded(
                        child: LinearProgressIndicator(
                          value: entry.value,
                          backgroundColor: Colors.grey[300],
                          valueColor: AlwaysStoppedAnimation<Color>(
                            _getEmotionColor(entry.key),
                          ),
                        ),
                      ),
                      Text('${(entry.value * 100).toStringAsFixed(1)}%'),
                    ],
                  ),
                );
              }),
              const SizedBox(height: 16),
            ],
            
            // Results list
            Expanded(
              child: Card(
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'Results (${_results.length})',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          Text(
                            'Buffer: ${_engine?.getBufferStats()['count'] ?? 0} points',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                    Expanded(
                      child: ListView.builder(
                        itemCount: _results.length,
                        itemBuilder: (context, index) {
                          final result = _results[_results.length - 1 - index];
                          return ListTile(
                            leading: CircleAvatar(
                              backgroundColor: _getEmotionColor(result.emotion),
                              child: Text(
                                result.emotion[0],
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                            title: Text(result.emotion),
                            subtitle: Text(
                              '${(result.confidence * 100).toStringAsFixed(1)}% • '
                              '${result.timestamp.toLocal().toString().substring(11, 19)}',
                            ),
                            trailing: Text(
                              'Features: ${result.features.length}',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getEmotionColor(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'baseline':
        return Colors.blue;
      case 'stress':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  void _showInfo() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About This Demo'),
        content: const Text(
          'This demo performs real-time emotion inference from simulated heart rate and '
          'RR interval data using 14 HRV features.\n\n'
          'The app generates realistic biometric data and runs emotion inference '
          'using a sliding window approach.\n\n'
          'Models: ExtraTrees (14 HRV features)\n'
          'Features:\n'
          '• Time-domain: RMSSD, Mean_RR, HRV_SDNN, pNN50\n'
          '• Frequency-domain: HF, LF, HF_nu, LF_nu, LFHF, TP\n'
          '• Non-linear: SD1SD2, SampEn, DFA_alpha1\n'
          '• Heart Rate\n\n'
          'Powered by synheart_emotion package.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }
}
