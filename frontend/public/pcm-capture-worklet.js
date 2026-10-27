// AudioWorklet: microphone float32 at the context rate -> 16 kHz mono Int16 frames of 512 samples (32 ms).
// Downsampling averages the input samples covering each output sample (box low-pass + decimation),
// which is adequate for speech recognition and avoids aliasing from naive sample dropping.
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.frameSize = (options.processorOptions && options.processorOptions.frameSize) || 512;
    this.ratio = sampleRate / this.targetRate; // `sampleRate` is a worklet global
    this.acc = 0;
    this.accCount = 0;
    this.pos = 0; // fractional input position relative to the next output boundary
    this.out = new Int16Array(this.frameSize);
    this.outIdx = 0;
    this.level = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    let peak = 0;
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i];
      const a = s < 0 ? -s : s;
      if (a > peak) peak = a;
      this.acc += s;
      this.accCount++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.accCount));
        this.acc = 0;
        this.accCount = 0;
        this.out[this.outIdx++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.outIdx === this.frameSize) {
          this.port.postMessage({ pcm: this.out.buffer, level: this.level }, [this.out.buffer]);
          this.out = new Int16Array(this.frameSize);
          this.outIdx = 0;
        }
      }
    }
    this.level = Math.max(peak, this.level * 0.9);
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
