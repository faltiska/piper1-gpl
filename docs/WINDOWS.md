### Prerequisites
Download CUDA Toolkit 12 x64: https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network
Download CUDNN 9 x64: https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
```
uv pip install piper-tts
uv pip install onnxruntime-gpu
python -m piper.download_voices --data-dir .voices en_GB-cori-high en_US-lessac-high en_US-libritts-high en_US-ljspeech-high en_US-ryan-high
python -m piper -m en_US-ryan-high --data-dir .voices -f test.wav -- 'Are you listening?'
```
These are required for running with CUDA, but you can also run on CPU with them.

### Test synthesis
```
python -m piper -m en_US-ryan-high --data-dir .voices --cuda -f test.wav -- 'Are you listening?'
```
To run on CPU, just remove --cuda from the command line.

### Start a development web server
```
uv pip install flask
python -m piper.http_server -m en_US-ryan-high --data-dir .voices --cuda
```

### Train
Install Visual Studio 2022 build tools first.

It is highly recommended to fine-tune from a pre-trained checkpoint, as this significantly speeds up training and improves quality.
Pre-trained checkpoints are available at https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main

```
uv pip install -e .[train]
cd src/piper/train/vits/monotonic_align
mkdir monotonic_align
cythonize -i core.pyx
move core*.pyd monotonic_align\
setup_visual_studio_env.bat
uv pip install cmake
mkdir build && cd build && cmake .. && cmake --build . --config Release
copy build\Release\espeakbridge.pyd src\piper\espeakbridge.pyd
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

python -m piper.train fit ^
  --data.voice_name "your-voice-name" ^
  --data.csv_path .training/corpus/ARU_v1_0/metadata.csv ^
  --data.audio_dir .training/corpus/ARU_v1_0/wav ^
  --model.sample_rate 22050 ^
  --data.espeak_voice "en-gb" ^
  --data.cache_dir .training/cache ^
  --data.config_path .training/model.onnx.json ^
  --trainer.log_every_n_steps 3 ^
  --data.num_workers 16 ^
  --data.batch_size 48 ^
  --data.validation_split 0.0 ^
  --ckpt_path /path/to/last/checkpoint
```

Parameter --ckpt_path will be something like "lightning_logs/version_0/checkpoints/epoch=39-step=160.ckpt"
Set --data.batch_size as large as your GPU memory allows.
Set --data.num_workers to at least half of the CPU cores you have.
Monitor GPU performance in Task Manager.  

See more [here](TRAINING.md).

When done, export the model:
```
python -m piper.train.export_onnx ^
  --checkpoint lightning_logs/version_0/checkpoints/epoch=45-step=184.ckpt ^
  --output-file .training/model.onnx
```

## Command line parameters:
```
usage: __main__.py [options] fit [-h] [-c CONFIG] [--print_config[=flags]] [--seed_everything SEED_EVERYTHING] [--trainer CONFIG] [--trainer.accelerator.help CLASS_PATH_OR_NAME] [--trainer.accelerator ACCELERATOR]
     [--trainer.strategy.help CLASS_PATH_OR_NAME] [--trainer.strategy STRATEGY] [--trainer.devices DEVICES] [--trainer.num_nodes NUM_NODES] [--trainer.precision PRECISION] [--trainer.logger.help CLASS_PATH_OR_NAME]
     [--trainer.logger LOGGER] [--trainer.callbacks.help CLASS_PATH_OR_NAME] [--trainer.callbacks CALLBACKS] [--trainer.fast_dev_run FAST_DEV_RUN] [--trainer.max_epochs MAX_EPOCHS] [--trainer.min_epochs MIN_EPOCHS]
     [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS] [--trainer.max_time MAX_TIME] [--trainer.limit_train_batches LIMIT_TRAIN_BATCHES] [--trainer.limit_val_batches LIMIT_VAL_BATCHES]
     [--trainer.limit_test_batches LIMIT_TEST_BATCHES] [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES] [--trainer.overfit_batches OVERFIT_BATCHES] [--trainer.val_check_interval VAL_CHECK_INTERVAL]
     [--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--trainer.log_every_n_steps LOG_EVERY_N_STEPS] [--trainer.enable_checkpointing {true,false,null}]
     [--trainer.enable_progress_bar {true,false,null}] [--trainer.enable_model_summary {true,false,null}] [--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--trainer.gradient_clip_val GRADIENT_CLIP_VAL]
     [--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.deterministic DETERMINISTIC] [--trainer.benchmark {true,false,null}] [--trainer.inference_mode {true,false}]
     [--trainer.use_distributed_sampler {true,false}] [--trainer.profiler.help CLASS_PATH_OR_NAME] [--trainer.profiler PROFILER] [--trainer.detect_anomaly {true,false}] [--trainer.barebones {true,false}]
     [--trainer.plugins.help CLASS_PATH_OR_NAME] [--trainer.plugins PLUGINS] [--trainer.sync_batchnorm {true,false}] [--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
     [--trainer.default_root_dir DEFAULT_ROOT_DIR] [--trainer.model_registry MODEL_REGISTRY] [--model CONFIG] [--model.sample_rate SAMPLE_RATE] [--model.num_speakers NUM_SPEAKERS] [--model.resblock RESBLOCK]
     [--model.resblock_kernel_sizes RESBLOCK_KERNEL_SIZES] [--model.resblock_dilation_sizes RESBLOCK_DILATION_SIZES] [--model.upsample_rates UPSAMPLE_RATES] [--model.upsample_initial_channel UPSAMPLE_INITIAL_CHANNEL]
     [--model.upsample_kernel_sizes UPSAMPLE_KERNEL_SIZES] [--model.filter_length FILTER_LENGTH] [--model.hop_length HOP_LENGTH] [--model.win_length WIN_LENGTH] [--model.mel_channels MEL_CHANNELS] [--model.mel_fmin MEL_FMIN]
     [--model.mel_fmax MEL_FMAX] [--model.inter_channels INTER_CHANNELS] [--model.hidden_channels HIDDEN_CHANNELS] [--model.filter_channels FILTER_CHANNELS] [--model.n_heads N_HEADS] [--model.n_layers N_LAYERS]
     [--model.kernel_size KERNEL_SIZE] [--model.p_dropout P_DROPOUT] [--model.n_layers_q N_LAYERS_Q] [--model.use_spectral_norm {true,false}] [--model.gin_channels GIN_CHANNELS] [--model.use_sdp {true,false}]
     [--model.segment_size SEGMENT_SIZE] [--model.learning_rate LEARNING_RATE] [--model.learning_rate_d LEARNING_RATE_D] [--model.betas [ITEM,...]] [--model.betas_d [ITEM,...]] [--model.eps EPS] [--model.lr_decay LR_DECAY]
     [--model.lr_decay_d LR_DECAY_D] [--model.init_lr_ratio INIT_LR_RATIO] [--model.warmup_epochs WARMUP_EPOCHS] [--model.c_mel C_MEL] [--model.c_kl C_KL] [--model.grad_clip GRAD_CLIP]
     [--model.dataset.help [CLASS_PATH_OR_NAME]] [--model.dataset DATASET] [--data CONFIG] --data.csv_path CSV_PATH --data.cache_dir CACHE_DIR --data.espeak_voice ESPEAK_VOICE --data.config_path CONFIG_PATH
     --data.voice_name VOICE_NAME [--data.audio_dir AUDIO_DIR] [--data.alignments_dir ALIGNMENTS_DIR] [--data.num_symbols NUM_SYMBOLS] [--data.batch_size BATCH_SIZE] [--data.validation_split VALIDATION_SPLIT]
     [--data.num_test_examples NUM_TEST_EXAMPLES] [--data.num_workers NUM_WORKERS] [--data.trim_silence {true,false}] [--data.keep_seconds_before_silence KEEP_SECONDS_BEFORE_SILENCE]
     [--data.keep_seconds_after_silence KEEP_SECONDS_AFTER_SILENCE] [--optimizer.help [CLASS_PATH_OR_NAME]] [--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE] [--lr_scheduler.help CLASS_PATH_OR_NAME]
     [--lr_scheduler CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE] [--ckpt_path CKPT_PATH]
```

