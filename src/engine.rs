//! Clean Vocal processing engine.
//!
//! Chain, per channel at the host sample rate:
//!   hum cut (2-pole high-pass) -> resample to 48 kHz -> DeepFilterNet3 -> resample back
//!
//! Everything runs synchronously on the calling thread, so offline renders are always fully
//! processed (no "dry fallback" when the CPU is busy). The total delay is fixed and reported via
//! `latency_samples()` so the host can line the vocal back up with the beat.

use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::Array2;
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;

/// DeepFilterNet3 runs at 48 kHz with a 480-sample (10 ms) hop.
const MODEL_SR: usize = 48_000;

/// User-facing settings.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// 0.0 ..= 1.0. 1.0 = full removal, lower keeps some of the room/air.
    pub strength: f32,
    /// Extra-clean post-filter (stronger cleanup in the gaps).
    pub extra_clean: bool,
    /// Hum cut on/off.
    pub hum_cut: bool,
    /// Hum cut corner frequency in Hz.
    pub hum_cut_hz: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self { strength: 1.0, extra_clean: true, hum_cut: true, hum_cut_hz: 75.0 }
    }
}

/// Strength (0..1) -> DeepFilterNet attenuation limit in dB. 100 dB means unlimited.
pub fn strength_to_atten_db(strength: f32) -> f32 {
    let s = strength.clamp(0.0, 1.0);
    if s >= 0.995 {
        100.0
    } else {
        // Curved so the whole knob is useful on real recordings:
        // 0% -> 1 dB (basically untouched), 50% -> ~13 dB, 75% -> ~24 dB, 99% -> ~35 dB
        1.0 + 35.0 * s.powf(1.5)
    }
}

const PF_BETA: f32 = 0.02; // same as the deep-filter CLI default for --pf

/// 2-pole Butterworth high-pass (RBJ cookbook, Q = 0.707), transposed direct form II.
#[derive(Clone, Copy, Default)]
struct HighPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl HighPass {
    fn set(&mut self, sr: f32, hz: f32) {
        let hz = hz.clamp(10.0, sr * 0.45);
        let w0 = 2.0 * std::f32::consts::PI * hz / sr;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        self.b0 = (1.0 + cos) / 2.0 / a0;
        self.b1 = -(1.0 + cos) / a0;
        self.b2 = (1.0 + cos) / 2.0 / a0;
        self.a1 = -2.0 * cos / a0;
        self.a2 = (1.0 - alpha) / a0;
    }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
    fn clear(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

pub struct Engine {
    host_sr: usize,
    channels: usize,
    /// Host-rate samples per DeepFilterNet hop (441 at 44.1 kHz, 480 at 48 kHz, ...).
    chunk_in: usize,
    hop: usize,

    df_pristine: DfTract,
    df: DfTract,
    up: Option<FftFixedInOut<f32>>,
    down: Option<FftFixedInOut<f32>>,

    hp: Vec<HighPass>,
    settings: Settings,
    applied: Option<Settings>,

    in_fifo: Vec<VecDeque<f32>>,
    out_fifo: Vec<VecDeque<f32>>,
    prime: usize,
    latency: usize,

    // Scratch buffers (allocated once).
    host_chunk: Vec<Vec<f32>>,
    model_chunk: Vec<Vec<f32>>,
    host_out: Vec<Vec<f32>>,
    df_in: Array2<f32>,
    df_out: Array2<f32>,
}

// SAFETY: DfTract holds `Rc`s internally (tract runtime state), which makes it !Send. Every one
// of those `Rc`s, including the ones shared between `df` and `df_pristine` via `clone()`, is
// owned exclusively by this `Engine` and never handed out. The engine is only ever used through
// `&mut self`, so moving the whole engine to another thread moves all clones of each `Rc`
// together and no two threads can touch a refcount at the same time.
unsafe impl Send for Engine {}

impl Engine {
    /// Supported host rates are those where sr/100 is a whole number of samples
    /// (44.1k, 48k, 88.2k, 96k, 176.4k, 192k, 32k, 16k, ...).
    pub fn supports_rate(sr: usize) -> bool {
        sr >= 8_000 && sr % 100 == 0
    }

    pub fn new(host_sr: usize, channels: usize) -> Result<Self, String> {
        if !Self::supports_rate(host_sr) {
            return Err(format!("Unsupported sample rate {host_sr}"));
        }
        let channels = channels.max(1);
        let rp = RuntimeParams::default_with_ch(channels)
            .with_post_filter(PF_BETA)
            .with_atten_lim(100.0)
            .with_thresholds(-15.0, 35.0, 35.0) // same as the deep-filter CLI defaults
            .with_mask_reduce(ReduceMask::MAX);
        let df = DfTract::new(DfParams::default(), &rp).map_err(|e| e.to_string())?;
        if df.sr != MODEL_SR {
            return Err(format!("Unexpected model rate {}", df.sr));
        }
        let hop = df.hop_size;
        let chunk_in = host_sr * hop / MODEL_SR; // exact for supported rates

        let (up, down) = if host_sr == MODEL_SR {
            (None, None)
        } else {
            // Note: rubato 0.14 sizes FftFixedInOut from the *output* side, so we pass the
            // desired output length (verified exactly below).
            let up = FftFixedInOut::<f32>::new(host_sr, MODEL_SR, hop, channels)
                .map_err(|e| e.to_string())?;
            let down = FftFixedInOut::<f32>::new(MODEL_SR, host_sr, chunk_in, channels)
                .map_err(|e| e.to_string())?;
            if up.input_frames_next() != chunk_in
                || up.output_frames_next() != hop
                || down.input_frames_next() != hop
                || down.output_frames_next() != chunk_in
            {
                return Err("Resampler chunk sizes do not line up".into());
            }
            (Some(up), Some(down))
        };

        // ---- Latency (in host samples) ----
        // 1) waiting for a full chunk: we output one chunk behind (prime = chunk_in - 1)
        let prime = chunk_in - 1;
        // 2) DeepFilterNet: STFT delay + model lookahead (at 48 kHz)
        let df_delay_48k = (df.fft_size - df.hop_size) + df.lookahead * df.hop_size;
        // 3) resampler delays
        let up_delay_48k = up.as_ref().map(|r| r.output_delay()).unwrap_or(0);
        let down_delay_host = down.as_ref().map(|r| r.output_delay()).unwrap_or(0);
        let model_side = (df_delay_48k + up_delay_48k) as f64 * host_sr as f64 / MODEL_SR as f64;
        let latency = prime + model_side.round() as usize + down_delay_host;

        let mut hp = vec![HighPass::default(); channels];
        for f in hp.iter_mut() {
            f.set(host_sr as f32, 75.0);
        }

        let mut e = Self {
            host_sr,
            channels,
            chunk_in,
            hop,
            df_pristine: df.clone(),
            df,
            up,
            down,
            hp,
            settings: Settings::default(),
            applied: None,
            in_fifo: vec![VecDeque::with_capacity(chunk_in * 4); channels],
            out_fifo: vec![VecDeque::with_capacity(chunk_in * 4); channels],
            prime,
            latency,
            host_chunk: vec![vec![0.0; chunk_in]; channels],
            model_chunk: vec![vec![0.0; hop]; channels],
            host_out: vec![vec![0.0; chunk_in]; channels],
            df_in: Array2::zeros((channels, hop)),
            df_out: Array2::zeros((channels, hop)),
        };
        e.reset();
        Ok(e)
    }

    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn set_settings(&mut self, s: Settings) {
        self.settings = s;
    }

    /// Clear all state (called when playback restarts).
    pub fn reset(&mut self) {
        self.df = self.df_pristine.clone();
        self.applied = None;
        for f in self.hp.iter_mut() {
            f.clear();
        }
        if let Some(r) = self.up.as_mut() {
            r.reset();
        }
        if let Some(r) = self.down.as_mut() {
            r.reset();
        }
        for q in self.in_fifo.iter_mut() {
            q.clear();
        }
        for q in self.out_fifo.iter_mut() {
            q.clear();
            q.extend(std::iter::repeat(0.0).take(self.prime));
        }
    }

    fn apply_settings(&mut self) {
        let s = self.settings;
        let changed = match self.applied {
            None => true,
            Some(a) => {
                a.strength != s.strength
                    || a.extra_clean != s.extra_clean
                    || a.hum_cut_hz != s.hum_cut_hz
            }
        };
        if !changed {
            return;
        }
        self.df.set_atten_lim(strength_to_atten_db(s.strength));
        self.df.set_pf_beta(if s.extra_clean { PF_BETA } else { 0.0 });
        for f in self.hp.iter_mut() {
            f.set(self.host_sr as f32, s.hum_cut_hz);
        }
        self.applied = Some(s);
    }

    /// Process one chunk_in-sized block per channel from `host_chunk` into `host_out`.
    fn process_chunk(&mut self) {
        // Up-sample to 48 kHz (or copy).
        match self.up.as_mut() {
            Some(r) => {
                let _ = r.process_into_buffer(&self.host_chunk, &mut self.model_chunk, None);
            }
            None => {
                for (dst, src) in self.model_chunk.iter_mut().zip(self.host_chunk.iter()) {
                    dst.copy_from_slice(src);
                }
            }
        }
        for (ch, buf) in self.model_chunk.iter().enumerate() {
            for (i, v) in buf.iter().enumerate() {
                self.df_in[[ch, i]] = *v;
            }
        }
        if self.df.process(self.df_in.view(), self.df_out.view_mut()).is_err() {
            // On a model error, pass audio through rather than going silent.
            self.df_out.assign(&self.df_in);
        }
        for (ch, buf) in self.model_chunk.iter_mut().enumerate() {
            for (i, v) in buf.iter_mut().enumerate() {
                *v = self.df_out[[ch, i]];
            }
        }
        // Down-sample back (or copy).
        match self.down.as_mut() {
            Some(r) => {
                let _ = r.process_into_buffer(&self.model_chunk, &mut self.host_out, None);
            }
            None => {
                for (dst, src) in self.host_out.iter_mut().zip(self.model_chunk.iter()) {
                    dst.copy_from_slice(src);
                }
            }
        }
    }

    /// Process audio in place. `buffers` holds one slice per channel, all the same length.
    pub fn process(&mut self, buffers: &mut [&mut [f32]]) {
        self.apply_settings();
        let n = buffers.first().map(|b| b.len()).unwrap_or(0);
        let nch = buffers.len().min(self.channels);
        let hum = self.settings.hum_cut;
        for i in 0..n {
            for ch in 0..self.channels {
                // If the host gives fewer channels than configured, duplicate the first one.
                let x = buffers[ch.min(nch.saturating_sub(1))][i];
                let x = if hum { self.hp[ch].run(x) } else { x };
                self.in_fifo[ch].push_back(x);
            }
            if self.in_fifo[0].len() >= self.chunk_in {
                for ch in 0..self.channels {
                    for (k, v) in self.in_fifo[ch].drain(..self.chunk_in).enumerate() {
                        self.host_chunk[ch][k] = v;
                    }
                }
                self.process_chunk();
                for ch in 0..self.channels {
                    self.out_fifo[ch].extend(self.host_out[ch].iter().copied());
                }
            }
            for ch in 0..nch {
                buffers[ch][i] = self.out_fifo[ch].pop_front().unwrap_or(0.0);
            }
            for ch in nch..self.channels {
                let _ = self.out_fifo[ch].pop_front();
            }
        }
    }
}
