#!/bin/bash
# Voxtral-4B-TTS int4 quantized inference
# Usage:
#   ./run.sh "Hello, how are you today?"
#   ./run.sh "Your text here" output.wav
#   ./run.sh                              # runs benchmark

cd "$(dirname "$0")/src"
PYTHON="../venv/bin/python3"
MODEL_DIR="../models/original"

if [ -z "$1" ]; then
    echo "Running benchmark..."
    exec $PYTHON torchao_inference.py
fi

TEXT="$1"
OUTPUT="${2:-/tmp/voxtral_output.wav}"

exec $PYTHON -c "
import torch, soundfile as sf
from torchao_inference import load_model_int4
from generate_fast import TekkenTokenizer, generate_speech_fast

model = load_model_int4('$MODEL_DIR', device='cuda')
tok = TekkenTokenizer('$MODEL_DIR/tekken.json')
voice_dir = '$MODEL_DIR/voice_embedding'

with torch.inference_mode():
    # Warmup
    generate_speech_fast(model, tok, 'Hi.', voice_name='neutral_female',
        voice_dir=voice_dir, max_frames=5, device='cuda', flow_steps=3, cfg_alpha=1.0)

    audio, gen_time = generate_speech_fast(
        model, tok, '$TEXT',
        voice_name='neutral_female', voice_dir=voice_dir,
        max_frames=300, device='cuda', flow_steps=3, cfg_alpha=1.0)

dur = len(audio) / 24000
fps = int(dur * 12.5) / gen_time
print(f'Generated {dur:.1f}s audio in {gen_time:.2f}s ({fps:.0f} fps, RTF={gen_time/dur:.3f})')
sf.write('$OUTPUT', audio, 24000)
print(f'Saved to $OUTPUT')
"
