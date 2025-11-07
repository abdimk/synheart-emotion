import 'package:flutter_test/flutter_test.dart';
import 'package:synheart_emotion/synheart_emotion.dart';

void main() {
  // Note: JsonLinearModel tests removed in v0.2.0
  // JsonLinearModel has been removed in favor of OnnxEmotionModel
  // See CHANGELOG.md for migration guide

  group('Package Structure', () {
    test('package exports are available', () {
      // Verify that core classes are exported
      expect(() => EmotionConfig(), returnsNormally);
      expect(() => EmotionEngine.fromPretrained(const EmotionConfig()),
          returnsNormally);
    });
  });
}
