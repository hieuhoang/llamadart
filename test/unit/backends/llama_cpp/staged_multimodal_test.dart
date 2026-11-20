@TestOn('vm')
library;

import 'dart:mirrors';
import 'dart:typed_data';

import 'package:llamadart/src/backends/llama_cpp/llama_cpp_service.dart';
import 'package:llamadart/src/core/exceptions.dart';
import 'package:llamadart/src/core/models/chat/content_part.dart';
import 'package:test/test.dart';

void main() {
  final library = reflectClass(LlamaCppService).owner as LibraryMirror;
  final check = MirrorSystem.getSymbol('_checkStagedMedia', library);

  test('staged image requests accept images but reject audio', () {
    const image = LlamaImageContent(path: '/image.png');
    final audio = LlamaAudioContent(samples: Float32List(16));
    expect(
      () => library.invoke(check, [
        [image],
      ]),
      returnsNormally,
    );
    for (final parts in [
      [audio],
      [image, audio],
    ]) {
      expect(
        () => library.invoke(check, [parts]),
        throwsA(
          isA<LlamaUnsupportedException>().having(
            (error) => error.message,
            'message',
            contains('Disable staged media processing'),
          ),
        ),
      );
    }
  });
}
