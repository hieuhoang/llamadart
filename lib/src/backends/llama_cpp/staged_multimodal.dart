part of 'llama_cpp_service.dart';

void _checkStagedMedia(List<LlamaContentPart> media) {
  if (media.any((part) => part is LlamaAudioContent)) {
    throw LlamaUnsupportedException(
      'Staged audio is not supported. Disable staged media processing '
      'to send audio.',
    );
  }
}

class _StagedChunk {
  const _StagedChunk({
    required this.tokens,
    required this.embeddings,
    required this.positions,
    required this.positionCount,
  });

  final Int32List? tokens;
  final Float32List? embeddings;
  final Int32List? positions;
  final int positionCount;
}

class _StagedCancelled implements Exception {}

class _StagedPrompt {
  const _StagedPrompt(this.chunks, this.width);

  final List<_StagedChunk> chunks;
  final int width;
}

extension _StagedMultimodal on LlamaCppService {
  _StagedPrompt _stageMultimodalPrompt(
    int contextHandle,
    int modelHandle,
    int projectorHandle,
    String prompt,
    List<LlamaContentPart> media,
    String? chatMediaMarker,
    Pointer<Int8> cancelToken,
  ) {
    final oldContext = _contexts[contextHandle]!;
    final oldModel = _models[modelHandle]!;
    final modelPath = oldModel.sourcePath;
    final projectorPath = _mtmdProjectorPaths[projectorHandle];
    final params = _modelLoadParams[modelHandle]!;
    if (modelPath == null || projectorPath == null) {
      throw LlamaStateException('Staged media requires local model files.');
    }
    if (_contextToModel.values.where((value) => value == modelHandle).length !=
            1 ||
        (_activeLoras[contextHandle]?.isNotEmpty ?? false) ||
        (_loraAdapters[modelHandle]?.isNotEmpty ?? false) ||
        (_modelToSpeculativeDraftModelKeys[modelHandle]?.isNotEmpty ?? false) ||
        (_generatingContexts[contextHandle] ?? 0) != 1 ||
        _decisionHeads.isNotEmpty) {
      throw LlamaUnsupportedException(
        'Staged media requires one context without LoRA adapters or decision heads.',
      );
    }
    final capacity = llama_n_ctx(oldContext.pointer);
    _calls.synchronize(oldContext.pointer);
    _contexts.remove(contextHandle);
    oldContext.dispose();
    _models.remove(modelHandle);
    oldModel.dispose();
    try {
      return _encodeStagedPrompt(
        modelPath,
        projectorPath,
        prompt,
        media,
        chatMediaMarker,
        capacity,
        cancelToken,
      );
    } finally {
      _restoreStagedModel(
        contextHandle,
        modelHandle,
        oldContext,
        modelPath,
        params,
      );
    }
  }

  _StagedPrompt _encodeStagedPrompt(
    String modelPath,
    String projectorPath,
    String prompt,
    List<LlamaContentPart> media,
    String? chatMediaMarker,
    int capacity,
    Pointer<Int8> cancelToken,
  ) {
    void checkCancelled() {
      if (cancelToken.value != 0) {
        throw _StagedCancelled();
      }
    }

    if (_calls.mixedBatch == null || _mtmdExitTeardown == null) {
      throw LlamaUnsupportedException(
        'Staged media requires the guarded native mixed-batch API.',
      );
    }
    checkCancelled();
    final modelParams = llama_model_default_params()
      ..no_alloc = true
      ..n_gpu_layers = 0
      ..load_mode = llama_load_mode.LLAMA_LOAD_MODE_NONE;
    final modelPathPtr = modelPath.toNativeUtf8();
    Pointer<llama_model> model = nullptr;
    Pointer<mtmd_context> projector = nullptr;
    _PreparedEmbedding? prepared;
    try {
      model = _loadOwnedModel(modelPathPtr.cast(), modelParams);
      if (model == nullptr) {
        throw LlamaModelException('Could not load model metadata for media.');
      }
      IsolateShutdownReleases.current.hold(
        ShutdownStage.model,
        _calls.modelFreeAddress,
        model,
      );
      checkCancelled();
      final projectorParams = _mtmdContextParamsFor(-1)..use_gpu = false;
      final projectorPathPtr = projectorPath.toNativeUtf8();
      try {
        projector = _mtmdInitFromFile(
          projectorPathPtr.cast(),
          model,
          projectorParams,
        );
      } finally {
        malloc.free(projectorPathPtr);
      }
      if (projector == nullptr) {
        throw LlamaModelException('Could not load the media projector.');
      }
      for (final part in media) {
        final audio = part is LlamaAudioContent;
        if (_mtmdHasEncoder(projector, audio: audio) != true) {
          throw LlamaUnsupportedException(
            'The projector does not report ${audio ? 'audio' : 'vision'} support.',
          );
        }
      }
      checkCancelled();
      final vocab = llama_model_get_vocab(model);
      final normalized = _normalizeMtmdPromptMarkers(
        prompt,
        media.length,
        chatMediaMarker: chatMediaMarker,
      );
      final marker = _mtmdDefaultMarker().cast<Utf8>().toDartString();
      final textParts = normalized.split(marker);
      if (textParts.length != media.length + 1) {
        throw LlamaInferenceException(
          'Media markers do not match the attachments.',
        );
      }
      final input = LlamaEmbeddingInput([
        for (var i = 0; i < media.length; i++) ...[
          LlamaTextContent(textParts[i]),
          media[i],
        ],
        LlamaTextContent(textParts.last),
      ]);
      final bos = llama_vocab_bos(vocab);
      final eos = llama_vocab_eos(vocab);
      prepared = _prepareEmbedding(
        input,
        projector,
        vocab,
        addSpecial:
            bos != eos &&
            bos != -1 &&
            !_promptStartsWithBosToken(vocab, normalized),
      );
      final api = _calls.mixedBatch!;
      final count = api.tokenCount(prepared.chunks);
      if (count > capacity) {
        throw LlamaInferenceException(
          'The prompt and media need $count tokens, exceeding the $capacity-token context.',
        );
      }
      if (!api.projectorUsesMrope(projector)) {
        throw LlamaUnsupportedException(
          'Staged media currently requires an M-RoPE projector.',
        );
      }
      final width = _calls.embeddingModelInfo(model).inputWidth;
      if (width <= 0) {
        throw LlamaInferenceException(
          'The media embedding width is unavailable.',
        );
      }
      final chunks = <_StagedChunk>[];
      for (var i = 0; i < api.chunksSize(prepared.chunks); i++) {
        checkCancelled();
        final chunk = api.chunksGet(prepared.chunks, i);
        final type = api.chunkType(chunk);
        final n = api.chunkTokenCount(chunk);
        final nPos = mtmd_input_chunk_get_n_pos(chunk);
        if (n == 0) {
          if (nPos != 0) {
            throw LlamaInferenceException('Invalid empty staged media chunk.');
          }
          continue;
        }
        if (type == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT.value) {
          final nPtr = calloc<Size>();
          try {
            final ptr = api.chunkTokens(chunk, nPtr);
            if (nPtr.value != n || (n > 0 && ptr == nullptr)) {
              throw LlamaInferenceException('Invalid staged text chunk.');
            }
            chunks.add(
              _StagedChunk(
                tokens: Int32List.fromList(ptr.asTypedList(n)),
                embeddings: null,
                positions: null,
                positionCount: nPos,
              ),
            );
          } finally {
            calloc.free(nPtr);
          }
          continue;
        }
        if (mtmd_decode_use_non_causal(projector, chunk)) {
          throw LlamaUnsupportedException(
            'Staged media does not support noncausal media chunks.',
          );
        }
        final image =
            type == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE.value;
        if (!image) {
          throw LlamaUnsupportedException('Unsupported staged media chunk.');
        }
        _calls.encodeEmbeddingChunk(projector, chunk);
        checkCancelled();
        final output = api.mediaOutput(projector);
        if (output == nullptr) {
          throw LlamaInferenceException(
            'The media encoder returned no embeddings.',
          );
        }
        final positions = Int32List(n * 4);
        final imageTokens = mtmd_input_chunk_get_tokens_image(chunk);
        if (imageTokens == nullptr) {
          throw LlamaInferenceException('Image position data is unavailable.');
        }
        for (var row = 0; row < n; row++) {
          final pos = mtmd_image_tokens_get_decoder_pos(imageTokens, 0, row);
          positions.setRange(row * 4, row * 4 + 4, [
            pos.t,
            pos.y,
            pos.x,
            pos.z,
          ]);
        }
        chunks.add(
          _StagedChunk(
            tokens: null,
            embeddings: Float32List.fromList(output.asTypedList(n * width)),
            positions: positions,
            positionCount: nPos,
          ),
        );
      }
      if (chunks.isEmpty) {
        throw LlamaInferenceException('The staged prompt contains no tokens.');
      }
      return _StagedPrompt(chunks, width);
    } finally {
      prepared?.dispose();
      if (projector != nullptr) _mtmdFree(projector);
      if (model != nullptr) {
        IsolateShutdownReleases.current.release(model);
        _calls.freeModel(model);
      }
      malloc.free(modelPathPtr);
    }
  }

  void _restoreStagedModel(
    int contextHandle,
    int modelHandle,
    _LlamaContextWrapper oldContext,
    String path,
    ModelParams params,
  ) {
    final newHandle = loadModel(path, params);
    final model = _models.remove(newHandle)!;
    _models[modelHandle] = model;
    _loraAdapters.remove(newHandle);
    _modelToMtmdUseGpu.remove(newHandle);
    _modelResolvedBackends.remove(newHandle);
    _modelArchitectures.remove(newHandle);
    _modelBackendNames.remove(newHandle);
    _modelResolvedGpuLayers.remove(newHandle);
    _modelsOnNoGpu.remove(newHandle);
    _modelLoadParams.remove(newHandle);
    _modelPlacements.remove(newHandle);
    _modelGpuMemoryFitReports.remove(newHandle);
    _modelFittedContextSizes.remove(newHandle);
    final context = _createOwnedContext(
      model.pointer,
      _contextParams[contextHandle]!,
    );
    if (context == nullptr) {
      throw LlamaContextException(
        'Could not restore the model context after media preparation.',
      );
    }
    IsolateShutdownReleases.current.hold(
      ShutdownStage.context,
      _calls.contextFreeAddress,
      context,
    );
    _contexts[contextHandle] = _LlamaContextWrapper(
      context,
      model,
      _calls,
      promptMicroBatchLimit: oldContext.promptMicroBatchLimit,
      qwen35VulkanRestricted: oldContext.qwen35VulkanRestricted,
    );
  }

  ({int nPast, int promptTokenCount, int evaluatedPromptTokens})
  _ingestStagedPrompt(
    _LlamaContextWrapper context,
    _StagedPrompt prompt,
    int batchSize,
    Pointer<Int8> cancelToken,
  ) {
    final api = _calls.mixedBatch!;
    final batch = _calls.createEmbeddingBatch(context.pointer);
    var past = 0;
    var evaluated = 0;
    try {
      using((arena) {
        final pos = arena<llama_pos>(4);
        final embd = arena<llama_embd>();
        for (var c = 0; c < prompt.chunks.length; c++) {
          final chunk = prompt.chunks[c];
          final tokens = chunk.tokens;
          final vectors = chunk.embeddings;
          final positions = chunk.positions;
          final count = tokens?.length ?? positions!.length ~/ 4;
          final nativeVectors = vectors == null
              ? nullptr
              : arena<Float>(vectors.length);
          if (vectors != null) {
            nativeVectors.asTypedList(vectors.length).setAll(0, vectors);
          }
          for (var start = 0; start < count; start += batchSize) {
            if (cancelToken.value != 0) return;
            llama_batch_ext_clear(batch);
            final end = math.min(start + batchSize, count);
            for (var row = start; row < end; row++) {
              final int index;
              if (tokens != null) {
                index = api.addToken(batch, 0, tokens[row]);
                pos[0] = past + row;
                pos[1] = 0;
                pos[2] = 0;
                pos[3] = 0;
              } else {
                embd.ref
                  ..data = nativeVectors + row * prompt.width
                  ..n_rows = 1
                  ..n_embd = prompt.width;
                index = api.addEmbedding(batch, 0, embd.ref);
                for (var j = 0; j < 4; j++) {
                  pos[j] = positions![row * 4 + j] + (j < 3 ? past : 0);
                }
              }
              if (index < 0 || !api.setPosition(batch, index, pos)) {
                throw LlamaInferenceException(
                  'Staged media batch rejected a token.',
                );
              }
              if (c == prompt.chunks.length - 1 &&
                  row == count - 1 &&
                  !llama_batch_ext_set_output_logits(batch, index, true)) {
                throw LlamaInferenceException(
                  'Could not request generation logits.',
                );
              }
            }
            _calls.processEmbeddingBatch(context.pointer, batch);
            evaluated += end - start;
          }
          past += chunk.positionCount;
        }
      });
      _calls.synchronize(context.pointer);
      return (
        nPast: past,
        promptTokenCount: 0,
        evaluatedPromptTokens: evaluated,
      );
    } finally {
      api.free(batch);
    }
  }
}
