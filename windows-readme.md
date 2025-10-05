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
python -m piper -m en_US-ryan-high --data-dir .voices --cuda -f test.wav -- 'Are you listenting?'
```
To run on CPU, just remove --cuda from the command line.

### Start a development web server
```
uv pip install flask
python -m piper.http_server --data-dir .voices --cuda
```

