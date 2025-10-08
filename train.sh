python -m piper.train fit \
  --data.voice_name "model" \
  --data.csv_path .training/corpus/ARU_v1_0/metadata.csv \
  --data.audio_dir .training/corpus/ARU_v1_0/wav \
  --model.sample_rate 22050 \
  --data.espeak_voice "en" \
  --data.cache_dir .training/cache \
  --data.config_path .training/model.onnx.json \
  --trainer.log_every_n_steps 3 \
  --data.num_workers 16 \
  --data.batch_size 45 \
  --trainer.check_val_every_n_epoch 5 \
  --ckpt_path lightning_logs/version_10/checkpoints/epoch=14494-step=57982.ckpt
  