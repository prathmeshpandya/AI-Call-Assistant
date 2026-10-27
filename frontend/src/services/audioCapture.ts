// Microphone capture: getUserMedia -> AudioWorklet (downsample to 16 kHz Int16) -> callback per 32 ms frame.

export interface CaptureCallbacks {
  onFrame: (pcm: ArrayBuffer) => void;
  onLevel?: (level: number) => void;
}

export class MicPermissionError extends Error {}

export class AudioCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;

  async start(cb: CaptureCallbacks): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new MicPermissionError("This browser does not support microphone capture (needs HTTPS or localhost).");
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // Echo cancellation reduces the assistant's own voice leaking into the mic (headphones are still best).
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      const name = (err as DOMException)?.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        throw new MicPermissionError("Microphone permission was denied. Allow it in the browser's site settings.");
      }
      if (name === "NotFoundError") throw new MicPermissionError("No microphone was found.");
      throw new MicPermissionError(`Could not open the microphone: ${(err as Error).message}`);
    }
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/pcm-capture-worklet.js");
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-capture", {
      processorOptions: { targetRate: 16000, frameSize: 512 },
    });
    this.node.port.onmessage = (e: MessageEvent<{ pcm: ArrayBuffer; level: number }>) => {
      cb.onFrame(e.data.pcm);
      cb.onLevel?.(e.data.level);
    };
    this.source.connect(this.node);
    // The worklet must be pulled by the graph; route it to a muted gain so nothing is audible.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.node.connect(mute).connect(this.ctx.destination);
  }

  async stop(): Promise<void> {
    this.node?.port.close();
    this.node?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close().catch(() => undefined);
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.source = null;
  }
}
