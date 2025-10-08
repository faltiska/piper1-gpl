python3 -m piper.train.export_onnx \
  --checkpoint lightning_logs/version_10/checkpoints/epoch=14494-step=57982.ckpt \
  --output-file .training/model.onnx
  
python3 -m piper -m model --data-dir .training -f test.wav -- "The soft cushion broke the man's fall."  