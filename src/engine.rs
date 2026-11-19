//! Clean Vocal v3 processing engine.
//!
//! Signal flow, per channel at the host sample rate:
//!
//! ```text
//!  in -> hum cut (always running, crossfaded on/off) -> dry
//!  dry -> [mono sum if Mono] -> resample to 48 kHz -> denoiser -> resample back -> wet
//!  out = mix( dry_delayed , wet + release fill )
//! ```
//!
//! Two denoisers:
//! - LIVE: RNNoise (nnnoiseless). Light enough to run right on the audio thread, ~20 ms delay.
//! - STANDARD / HQ / GENTLE: DeepFilterNet3. Runs on its own worker thread so the audio thread
//!   never waits on the neural net. If the worker is ever late, the output crossfades to the
//!   (delayed, hum-cut) dry signal for that moment instead of dropping to silence, and lines back
//!   up exactly when the worker catches up.
//!
//! Renders: when the host renders offline (or faster than real time), the audio thread waits
//! for the worker, so bounces are always fully processed.

use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::Array2;
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Both models run at 48 kHz with a 480-sample (10 ms) frame.
const MODEL_SR: usize = 48_000;
const FRAME_48K: usize = 480;
/// DeepFilterNet post-filter amount used for "Extra Clean" (deep-filter CLI default).
const PF_BETA: f32 = 0.02;
/// How fast the cut amount may move per 10 ms frame (dB), so knob moves never step.
const CUT_SLEW_DB: f32 = 1.5;
/// nnnoiseless output delay at 48 kHz (one frame). Verified by the alignment test.
const LIVE_DELAY_48K: usize = FRAME_48K;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Low-latency RNNoise: for hearing yourself while recording.
    Live,
    /// DeepFilterNet3, everyday mixing.
    Standard,
    /// DeepFilterNet3 at its strongest: for bouncing.
    Hq,
    /// DeepFilterNet3 with a shallow cut: natural, keeps air and breaths.
    Gentle,
}

impl Mode {
    pub fn uses_df(self) -> bool {
        !matches!(self, Mode::Live)
    }
}

/// User-facing settings.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub mode: Mode,
    /// Most the cleaner may turn noise down, in dB (60 = no limit).
    pub cut_db: f32,
    /// Extra post-filter in the gaps (STANDARD only; HQ always on, others off).
    pub extra_clean: bool,
    /// How slowly the cleaning closes in after a phrase ends (ms). Keeps word tails.
    pub release_ms: f32,
    /// 0..1 dry/wet.
    pub mix: f32,
    pub hum_cut: bool,
    pub hum_cut_hz: f32,
    /// Run the model once on L+R instead of per channel (half the CPU).
    pub mono: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            mode: Mode::Standard,
            cut_db: 18.0,
            extra_clean: false,
            release_ms: 100.0,
            mix: 1.0,
            hum_cut: true,
            hum_cut_hz: 75.0,
            mono: true,
        }
    }
}

/// What each mode actually runs with.
#[derive(Clone, Copy, Debug)]
pub struct Effective {
    pub cut_db: f32,
    pub pf_beta: f32,
    pub release_ms: f32,
}

impl Settings {
    pub fn effective(&self) -> Effective {
        let cut = self.cut_db.clamp(0.0, 60.0);
        match self.mode {
            Mode::Live => Effective { cut_db: cut, pf_beta: 0.0, release_ms: self.release_ms },
            Mode::Standard => Effective {
                cut_db: cut,
                pf_beta: if self.extra_clean { PF_BETA } else { 0.0 },
                release_ms: self.release_ms,
            },
            Mode::Hq => Effective {
                cut_db: cut.max(30.0),
                pf_beta: PF_BETA,
                release_ms: self.release_ms.min(80.0),
            },
            Mode::Gentle => Effective {
                cut_db: cut.min(8.0),
                pf_beta: 0.0,
                release_ms: self.release_ms.max(250.0),
            },
        }
    }
}

/// Numbers the panel shows. Shared with the editor.
#[derive(Default)]
pub struct Status {
    pub latency_samples: AtomicU32,
    pub sample_rate: AtomicU32,
    /// Times the AI was late and the dry signal covered for it.
    pub dropouts: AtomicU64,
    /// true while the AI model is (re)loading or catching up after a mode change.
    pub loading: AtomicBool,
}

// ---------------------------------------------------------------------------------------------
// Small DSP pieces
// ---------------------------------------------------------------------------------------------

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
}

fn db_to_lim(cut_db: f32) -> f32 {
    if cut_db >= 59.5 {
        0.0
    } else {
        10f32.powf(-cut_db.max(0.0) / 20.0)
    }
}

#[inline]
fn coef(ms: f32, sr: f32) -> f32 {
    1.0 - (-1.0 / (ms.max(0.1) * 0.001 * sr)).exp()
}

/// host <-> 48 kHz resampling around a 10 ms model frame.
struct Resample {
    up: Option<FftFixedInOut<f32>>,
    down: Option<FftFixedInOut<f32>>,
    host_in: Vec<Vec<f32>>,
    model_in: Vec<Vec<f32>>,
    model_out: Vec<Vec<f32>>,
    host_out: Vec<Vec<f32>>,
    /// delay added on the 48 kHz side (up-sampler), in 48 kHz samples
    up_delay_48k: usize,
    /// delay added by the down-sampler, in host samples
    down_delay_host: usize,
}

impl Resample {
    fn new(host_sr: usize, ch: usize) -> Result<Self, String> {
        let chunk_in = host_sr * FRAME_48K / MODEL_SR;
        let (up, down) = if host_sr == MODEL_SR {
            (None, None)
        } else {
            // rubato 0.14 sizes FftFixedInOut from the output side; verified below.
            let up = FftFixedInOut::<f32>::new(host_sr, MODEL_SR, FRAME_48K, ch)
                .map_err(|e| e.to_string())?;
            let down = FftFixedInOut::<f32>::new(MODEL_SR, host_sr, chunk_in, ch)
                .map_err(|e| e.to_string())?;
            if up.input_frames_next() != chunk_in
                || up.output_frames_next() != FRAME_48K
                || down.input_frames_next() != FRAME_48K
                || down.output_frames_next() != chunk_in
            {
                return Err("Resampler chunk sizes do not line up".into());
            }
            (Some(up), Some(down))
        };
        Ok(Self {
            up_delay_48k: up.as_ref().map(|r| r.output_delay()).unwrap_or(0),
            down_delay_host: down.as_ref().map(|r| r.output_delay()).unwrap_or(0),
            up,
            down,
            host_in: vec![vec![0.0; chunk_in]; ch],
            model_in: vec![vec![0.0; FRAME_48K]; ch],
            model_out: vec![vec![0.0; FRAME_48K]; ch],
            host_out: vec![vec![0.0; chunk_in]; ch],
        })
    }

    /// host_in -> 48k -> f -> host_out
    fn run(&mut self, f: &mut dyn FnMut(&[Vec<f32>], &mut [Vec<f32>])) {
        match self.up.as_mut() {
            Some(r) => {
                let _ = r.process_into_buffer(&self.host_in, &mut self.model_in, None);
            }
            None => {
                for (d, s) in self.model_in.iter_mut().zip(self.host_in.iter()) {
                    d.copy_from_slice(s);
                }
            }
        }
        f(&self.model_in, &mut self.model_out);
        match self.down.as_mut() {
            Some(r) => {
                let _ = r.process_into_buffer(&self.model_out, &mut self.host_out, None);
            }
            None => {
                for (d, s) in self.host_out.iter_mut().zip(self.model_out.iter()) {
                    d.copy_from_slice(s);
                }
            }
        }
    }

    /// Total delay in host samples for a model with `model_delay_48k`.
    fn delay_host(&self, host_sr: usize, model_delay_48k: usize) -> usize {
        let side = (model_delay_48k + self.up_delay_48k) as f64 * host_sr as f64 / MODEL_SR as f64;
        side.round() as usize + self.down_delay_host
    }
}

// ---------------------------------------------------------------------------------------------
// Denoisers (both work on one 48 kHz frame per channel)
// ---------------------------------------------------------------------------------------------

struct DfModel {
    df: DfTract,
    inp: Array2<f32>,
    out: Array2<f32>,
    cur_cut: f32,
    cur_pf: f32,
    rng: u32,
}

impl DfModel {
    fn new(ch: usize) -> Result<Self, String> {
        let rp = RuntimeParams::default_with_ch(ch)
            .with_post_filter(PF_BETA)
            .with_atten_lim(18.0)
            // v2 used (-15, 35, 35): frames under -15 dB SNR were hard-muted (the "gate") and
            // frames over 35 dB skipped processing (flip-flopping). Now every frame is cleaned
            // and the cut amount alone decides how deep it goes.
            .with_thresholds(-100.0, 100.0, 100.0)
            .with_mask_reduce(ReduceMask::MAX);
        let df = DfTract::new(DfParams::default(), &rp).map_err(|e| e.to_string())?;
        if df.sr != MODEL_SR || df.hop_size != FRAME_48K {
            return Err(format!("Unexpected model format {} Hz / {}", df.sr, df.hop_size));
        }
        let mut m = Self {
            df,
            inp: Array2::zeros((ch, FRAME_48K)),
            out: Array2::zeros((ch, FRAME_48K)),
            cur_cut: 18.0,
            cur_pf: PF_BETA,
            rng: 0x1234_5678,
        };
        m.df.set_atten_lim(m.cur_cut);
        Ok(m)
    }

    fn delay_48k(&self) -> usize {
        (self.df.fft_size - self.df.hop_size) + self.df.lookahead * self.df.hop_size
    }

    fn process(&mut self, inp: &[Vec<f32>], out: &mut [Vec<f32>], cut_db: f32, pf: f32) {
        // Smooth cut changes.
        let target = cut_db.clamp(0.0, 100.0);
        let step = (target - self.cur_cut).clamp(-CUT_SLEW_DB, CUT_SLEW_DB);
        if step != 0.0 {
            self.cur_cut += step;
            // >= 59.5 dB means "no limit"
            self.df.set_atten_lim(if self.cur_cut >= 59.5 { 100.0 } else { self.cur_cut.max(0.02) });
        }
        if pf != self.cur_pf {
            self.cur_pf = pf;
            self.df.set_pf_beta(pf);
        }

        let mut ms = 0.0f32;
        for (c, buf) in inp.iter().enumerate() {
            for (i, v) in buf.iter().enumerate() {
                self.inp[[c, i]] = *v;
                ms += v * v;
            }
        }
        ms /= (inp.len() * FRAME_48K) as f32;
        // DeepFilterNet zeroes any frame under ~-70 dBFS and freezes its history, which chops
        // quiet tails and pops when sound comes back. Keep very quiet frames just above that
        // with -66 dBFS dither; the model removes it as noise.
        if ms < 2e-7 {
            for v in self.inp.iter_mut() {
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 17;
                self.rng ^= self.rng << 5;
                *v += (self.rng as f32 / u32::MAX as f32 - 0.5) * 2.0e-3;
            }
        }

        let ok = self.df.process(self.inp.view(), self.out.view_mut()).is_ok();
        for (c, buf) in out.iter_mut().enumerate() {
            for (i, v) in buf.iter_mut().enumerate() {
                let y = if ok { self.out[[c, i]] } else { inp[c][i] };
                *v = if y.is_finite() { y } else { 0.0 };
            }
        }
    }
}

struct LiveModel {
    st: Vec<Box<nnnoiseless::DenoiseState<'static>>>,
    /// input history so the cut limit mixes back perfectly aligned noisy audio
    hist: Vec<Vec<f32>>,
    tmp_in: Vec<f32>,
    tmp_out: Vec<f32>,
    cur_cut: f32,
}

impl LiveModel {
    fn new(ch: usize) -> Self {
        Self {
            st: (0..ch).map(|_| nnnoiseless::DenoiseState::new()).collect(),
            hist: vec![vec![0.0; LIVE_DELAY_48K]; ch],
            tmp_in: vec![0.0; FRAME_48K],
            tmp_out: vec![0.0; FRAME_48K],
            cur_cut: 18.0,
        }
    }

    fn process(&mut self, inp: &[Vec<f32>], out: &mut [Vec<f32>], cut_db: f32) {
        let step = (cut_db - self.cur_cut).clamp(-CUT_SLEW_DB, CUT_SLEW_DB);
        self.cur_cut += step;
        let lim = db_to_lim(self.cur_cut);
        for c in 0..inp.len() {
            for (d, s) in self.tmp_in.iter_mut().zip(inp[c].iter()) {
                *d = s * 32768.0;
            }
            self.st[c].process_frame(&mut self.tmp_out, &self.tmp_in);
            // With a 480-sample delay, the aligned noisy input is exactly the previous frame.
            for i in 0..FRAME_48K {
                let wet = self.tmp_out[i] / 32768.0;
                let wet = if wet.is_finite() { wet } else { 0.0 };
                out[c][i] = wet * (1.0 - lim) + self.hist[c][i] * lim;
            }
            self.hist[c].copy_from_slice(&inp[c]);
        }
    }
}

/// LIVE path: runs synchronously on the audio thread.
struct LivePath {
    rs: Resample,
    model: LiveModel,
    in_fifo: Vec<VecDeque<f32>>,
    out_fifo: Vec<VecDeque<f32>>,
    chunk_in: usize,
    prime: usize,
    latency: usize,
}

impl LivePath {
    fn new(host_sr: usize, ch: usize) -> Result<Self, String> {
        let rs = Resample::new(host_sr, ch)?;
        let chunk_in = host_sr * FRAME_48K / MODEL_SR;
        let prime = chunk_in - 1;
        let latency = prime + rs.delay_host(host_sr, LIVE_DELAY_48K);
        let mut p = Self {
            rs,
            model: LiveModel::new(ch),
            in_fifo: vec![VecDeque::with_capacity(chunk_in * 4); ch],
            out_fifo: vec![VecDeque::with_capacity(chunk_in * 4); ch],
            chunk_in,
            prime,
            latency,
        };
        p.restart();
        Ok(p)
    }

    /// Re-prime the queues (no allocation).
    fn restart(&mut self) {
        for q in self.in_fifo.iter_mut() {
            q.clear();
        }
        for q in self.out_fifo.iter_mut() {
            q.clear();
            q.extend(std::iter::repeat(0.0).take(self.prime));
        }
    }

    /// Push one frame of model channels, get one frame back (always available).
    #[inline]
    fn tick(&mut self, x: &[f32], y: &mut [f32], cut_db: f32) {
        for (c, q) in self.in_fifo.iter_mut().enumerate() {
            q.push_back(x[c]);
        }
        if self.in_fifo[0].len() >= self.chunk_in {
            for c in 0..self.in_fifo.len() {
                for (k, v) in self.in_fifo[c].drain(..self.chunk_in).enumerate() {
                    self.rs.host_in[c][k] = v;
                }
            }
            let model = &mut self.model;
            self.rs.run(&mut |i, o| model.process(i, o, cut_db));
            for c in 0..self.out_fifo.len() {
                self.out_fifo[c].extend(self.rs.host_out[c].iter().copied());
            }
        }
        for (c, q) in self.out_fifo.iter_mut().enumerate() {
            y[c] = q.pop_front().unwrap_or(0.0);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// DeepFilterNet worker thread
// ---------------------------------------------------------------------------------------------

struct Shared {
    flush_req: AtomicU64,
    flush_ack: AtomicU64,
    /// model channel count wanted after the next flush
    mch_req: AtomicU32,
    cut_db: AtomicU32,
    pf_beta: AtomicU32,
    quit: AtomicBool,
    failed: AtomicBool,
}

fn worker_main(
    host_sr: usize,
    mut mch: usize,
    sh: Arc<Shared>,
    mut rx: rtrb::Consumer<f32>,
    mut tx: rtrb::Producer<f32>,
    ready: std::sync::mpsc::Sender<Result<usize, String>>,
) {
    let build = |ch: usize| -> Result<(DfModel, Resample), String> {
        Ok((DfModel::new(ch)?, Resample::new(host_sr, ch)?))
    };
    let (mut model, mut rs) = match build(mch) {
        Ok(v) => v,
        Err(e) => {
            sh.failed.store(true, Ordering::Release);
            let _ = ready.send(Err(e));
            return;
        }
    };
    let delay = rs.delay_host(host_sr, model.delay_48k());
    let _ = ready.send(Ok(delay));
    drop(ready);

    let chunk_in = host_sr * FRAME_48K / MODEL_SR;
    let mut filled = 0usize; // frames gathered into rs.host_in
    let mut gen = 0u64;
    loop {
        if sh.quit.load(Ordering::Acquire) {
            return;
        }
        let req = sh.flush_req.load(Ordering::Acquire);
        if req != gen {
            while rx.pop().is_ok() {}
            let want = sh.mch_req.load(Ordering::Acquire).max(1) as usize;
            if want != mch {
                match build(want) {
                    Ok((m, r)) => {
                        model = m;
                        rs = r;
                        mch = want;
                    }
                    Err(_) => {
                        sh.failed.store(true, Ordering::Release);
                        return;
                    }
                }
            }
            filled = 0;
            gen = req;
            sh.flush_ack.store(gen, Ordering::Release);
            continue;
        }

        // Gather one chunk.
        while filled < chunk_in && rx.slots() >= mch {
            for c in 0..mch {
                rs.host_in[c][filled] = rx.pop().unwrap_or(0.0);
            }
            filled += 1;
        }
        if filled < chunk_in {
            std::thread::park_timeout(Duration::from_millis(2));
            continue;
        }
        filled = 0;
        let cut = f32::from_bits(sh.cut_db.load(Ordering::Relaxed));
        let pf = f32::from_bits(sh.pf_beta.load(Ordering::Relaxed));
        rs.run(&mut |i, o| model.process(i, o, cut, pf));
        for k in 0..chunk_in {
            for c in 0..mch {
                // Ring is sized for seconds of audio; if it's ever full the audio thread
                // isn't reading, so dropping is harmless (it resyncs on its side).
                let _ = tx.push(rs.host_out[c][k]);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DfState {
    /// LIVE mode active, worker idle
    Idle,
    /// waiting for the worker to acknowledge a flush
    Flushing,
    Running,
}

// ---------------------------------------------------------------------------------------------
// Engine (audio-thread side)
// ---------------------------------------------------------------------------------------------

pub struct Engine {
    host_sr: usize,
    channels: usize,
    settings: Settings,
    status: Arc<Status>,

    // hum cut
    hp: Vec<HighPass>,
    hp_hz: f32,
    hum_mix: f32,

    // dry delay line
    dry: Vec<Vec<f32>>,
    dry_pos: usize,
    /// delay before the last mode switch, crossfaded out over 10 ms
    lat_prev: usize,
    lat_fade: f32,

    // LIVE
    live_mono: LivePath,
    live_multi: LivePath,

    // DeepFilterNet worker
    sh: Arc<Shared>,
    worker: Option<std::thread::JoinHandle<()>>,
    worker_thread: std::thread::Thread,
    to_worker: rtrb::Producer<f32>,
    from_worker: rtrb::Consumer<f32>,
    df_state: DfState,
    df_gen: u64,
    df_mch: usize,
    df_wait: usize, // samples between input and when its processed output is due
    df_latency: usize,
    pushed: usize,
    debt: usize,
    since_unpark: usize,

    // timing / render detection
    always_block: bool,
    fast_run: u32,
    slow_run: u32,
    blocking: bool,
    last_call: Option<Instant>,

    // output stage
    active_mode: Mode,
    active_mono: bool,
    xf: f32,
    env_d: f32,
    env_p: f32,
    g: f32,
    mix_s: f32,

    // scratch (allocated once)
    x_model: Vec<f32>,
    y_model: Vec<f32>,
    dry_in: Vec<f32>,
    dry_out: Vec<f32>,
    proc: Vec<f32>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.sh.quit.store(true, Ordering::Release);
        self.worker_thread.unpark();
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

impl Engine {
    /// Supported host rates are those where sr/100 is a whole number of samples
    /// (44.1k, 48k, 88.2k, 96k, 176.4k, 192k, 32k, 16k, ...).
    pub fn supports_rate(sr: usize) -> bool {
        sr >= 8_000 && sr % 100 == 0
    }

    /// `max_block`: the host's largest buffer. `offline`: host says it is rendering.
    pub fn new(
        host_sr: usize,
        channels: usize,
        max_block: usize,
        offline: bool,
        settings: Settings,
        status: Arc<Status>,
    ) -> Result<Self, String> {
        if !Self::supports_rate(host_sr) {
            return Err(format!("Unsupported sample rate {host_sr}"));
        }
        let channels = channels.max(1);
        let chunk_in = host_sr * FRAME_48K / MODEL_SR;

        let sh = Arc::new(Shared {
            flush_req: AtomicU64::new(0),
            flush_ack: AtomicU64::new(0),
            mch_req: AtomicU32::new(1),
            cut_db: AtomicU32::new(settings.effective().cut_db.to_bits()),
            pf_beta: AtomicU32::new(settings.effective().pf_beta.to_bits()),
            quit: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        });
        let ring = host_sr * 2 * channels;
        let (to_worker, rx) = rtrb::RingBuffer::<f32>::new(ring);
        let (tx, from_worker) = rtrb::RingBuffer::<f32>::new(ring);
        let df_mch = if settings.mono { 1 } else { channels };
        sh.mch_req.store(df_mch as u32, Ordering::Release);
        let (rtx, rrx) = std::sync::mpsc::channel();
        let sh2 = sh.clone();
        let worker = std::thread::Builder::new()
            .name("Clean Vocal AI".into())
            .spawn(move || worker_main(host_sr, df_mch, sh2, rx, tx, rtx))
            .map_err(|e| e.to_string())?;
        let df_delay = rrx
            .recv_timeout(Duration::from_secs(30))
            .map_err(|_| "AI model did not start".to_string())??;
        let worker_thread = worker.thread().clone();

        // Live playback: room for the host's block plus one model frame of compute, plus 2 ms
        // of jitter. Offline renders wait for the worker, so they need no extra room.
        let margin = if offline {
            0
        } else {
            max_block.clamp(32, 1024) + chunk_in + host_sr / 500
        };
        let df_wait = chunk_in - 1 + margin;
        let df_latency = df_wait + df_delay;

        let live_mono = LivePath::new(host_sr, 1)?;
        let live_multi = LivePath::new(host_sr, channels)?;
        let max_lat = df_latency.max(live_multi.latency).max(live_mono.latency);

        let mut hp = vec![HighPass::default(); channels];
        for f in hp.iter_mut() {
            f.set(host_sr as f32, settings.hum_cut_hz);
        }

        let mut e = Self {
            host_sr,
            channels,
            settings,
            status,
            hp,
            hp_hz: settings.hum_cut_hz,
            hum_mix: if settings.hum_cut { 1.0 } else { 0.0 },
            dry: vec![vec![0.0; max_lat + 1]; channels],
            dry_pos: 0,
            lat_prev: 0,
            lat_fade: 1.0,
            live_mono,
            live_multi,
            sh,
            worker: Some(worker),
            worker_thread,
            to_worker,
            from_worker,
            df_state: DfState::Running,
            df_gen: 0,
            df_mch,
            df_wait,
            df_latency,
            pushed: 0,
            debt: 0,
            since_unpark: 0,
            always_block: offline,
            fast_run: 0,
            slow_run: 0,
            blocking: offline,
            last_call: None,
            active_mode: settings.mode,
            active_mono: settings.mono,
            xf: 0.0,
            env_d: 0.0,
            env_p: 0.0,
            g: 1.0,
            mix_s: settings.mix,
            x_model: vec![0.0; channels],
            y_model: vec![0.0; channels],
            dry_in: vec![0.0; channels],
            dry_out: vec![0.0; channels],
            proc: vec![0.0; channels],
        };
        if !settings.mode.uses_df() {
            e.df_state = DfState::Idle;
        }
        e.publish_status();
        Ok(e)
    }

    /// Current total delay in host samples (depends on mode and Mono).
    pub fn latency_samples(&self) -> usize {
        if self.active_mode.uses_df() {
            self.df_latency
        } else if self.active_mono {
            self.live_mono.latency
        } else {
            self.live_multi.latency
        }
    }

    pub fn set_settings(&mut self, s: Settings) {
        self.settings = s;
    }

    /// Host restarted playback. The models keep running (no reload, no hitch); only the LIVE
    /// queues are re-primed. Nothing here allocates.
    pub fn reset(&mut self) {
        self.live_mono.restart();
        self.live_multi.restart();
    }

    fn publish_status(&self) {
        self.status.latency_samples.store(self.latency_samples() as u32, Ordering::Relaxed);
        self.status.sample_rate.store(self.host_sr as u32, Ordering::Relaxed);
        self.status
            .loading
            .store(self.active_mode.uses_df() && self.df_state == DfState::Flushing, Ordering::Relaxed);
    }

    fn start_flush(&mut self, mch: usize) {
        self.df_gen += 1;
        self.df_mch = mch;
        self.sh.mch_req.store(mch as u32, Ordering::Release);
        self.sh.flush_req.store(self.df_gen, Ordering::Release);
        self.worker_thread.unpark();
        self.df_state = DfState::Flushing;
        self.pushed = 0;
        self.debt = 0;
    }

    fn poll_flush(&mut self) {
        if self.df_state == DfState::Flushing
            && self.sh.flush_ack.load(Ordering::Acquire) == self.df_gen
        {
            while self.from_worker.pop().is_ok() {}
            self.df_state = DfState::Running;
        }
    }

    /// Apply mode / mono switches at the start of a block.
    fn apply_structure(&mut self) {
        let s = self.settings;
        if s.mode == self.active_mode && s.mono == self.active_mono {
            return;
        }
        let was_df = self.active_mode.uses_df();
        self.lat_prev = self.latency_samples();
        self.lat_fade = 0.0;
        self.active_mode = s.mode;
        self.active_mono = s.mono;
        let mch = if s.mono { 1 } else { self.channels };
        if s.mode.uses_df() {
            if !was_df || mch != self.df_mch || self.df_state == DfState::Idle {
                self.start_flush(mch);
            }
        } else {
            self.df_state = DfState::Idle;
            self.live_mono.restart();
            self.live_multi.restart();
        }
        // Fresh start for the crossfade so the change fades in from dry.
        self.xf = 0.0;
    }

    /// Decide whether to wait for the worker (renders) or never wait (live playback).
    fn update_timing(&mut self, n: usize) {
        if self.always_block {
            self.blocking = true;
            return;
        }
        let now = Instant::now();
        if let Some(prev) = self.last_call {
            let dt = now.duration_since(prev).as_secs_f64();
            let block = n as f64 / self.host_sr as f64;
            if dt < block * 0.3 {
                self.fast_run += 1;
                self.slow_run = 0;
            } else if dt > block * 0.6 {
                self.slow_run += 1;
                self.fast_run = 0;
            }
            if self.fast_run >= 3 {
                self.blocking = true;
            }
            if self.slow_run >= 8 {
                self.blocking = false;
            }
        }
        self.last_call = Some(now);
    }

    /// Process audio in place. `buffers` holds one slice per channel, all the same length.
    pub fn process(&mut self, buffers: &mut [&mut [f32]]) {
        let n = buffers.first().map(|b| b.len()).unwrap_or(0);
        if n == 0 {
            return;
        }
        let nch = buffers.len().min(self.channels).max(1);
        self.update_timing(n);
        self.apply_structure();
        self.poll_flush();

        let sr = self.host_sr as f32;
        let eff = self.settings.effective();
        self.sh.cut_db.store(eff.cut_db.to_bits(), Ordering::Relaxed);
        self.sh.pf_beta.store(eff.pf_beta.to_bits(), Ordering::Relaxed);

        // Hum cut corner: glide toward the knob (no zipper), filters never stop running.
        if (self.hp_hz - self.settings.hum_cut_hz).abs() > 0.05 {
            let t = self.settings.hum_cut_hz;
            self.hp_hz += (t - self.hp_hz) * (1.0 - (-(n as f32) / (0.03 * sr)).exp());
            for f in self.hp.iter_mut() {
                f.set(sr, self.hp_hz);
            }
        }
        let hum_target = if self.settings.hum_cut { 1.0 } else { 0.0 };
        let a_hum = coef(10.0, sr);
        let a_xf = coef(5.0, sr);
        let a_env = coef(10.0, sr);
        let a_att = coef(3.0, sr);
        let a_rel = coef(eff.release_ms, sr);
        let a_mix = coef(20.0, sr);
        let a_lat = 1.0 / (0.010 * sr); // linear 10 ms
        let mix_target = self.settings.mix.clamp(0.0, 1.0);

        let lat = self.latency_samples();
        let dry_len = self.dry[0].len();
        let mono = self.active_mono;
        let mch = if mono { 1 } else { self.channels };
        let use_df = self.active_mode.uses_df();
        let chunk_in = self.host_sr * FRAME_48K / MODEL_SR;
        let mut dropouts = 0u64;

        for i in 0..n {
            // ---- input, hum cut, dry delay ----
            self.hum_mix += (hum_target - self.hum_mix) * a_hum;
            let rd = (self.dry_pos + dry_len - lat) % dry_len;
            let rd_prev = (self.dry_pos + dry_len - self.lat_prev) % dry_len;
            let lf = self.lat_fade;
            if lf < 1.0 {
                self.lat_fade = (lf + a_lat).min(1.0);
            }
            for ch in 0..self.channels {
                let x = buffers[ch.min(nch - 1)][i];
                let y = self.hp[ch].run(x);
                let v = x + (y - x) * self.hum_mix;
                self.dry_in[ch] = v;
                self.dry[ch][self.dry_pos] = v;
                self.dry_out[ch] = if lf < 1.0 {
                    self.dry[ch][rd_prev] * (1.0 - lf) + self.dry[ch][rd] * lf
                } else {
                    self.dry[ch][rd]
                };
            }
            self.dry_pos = (self.dry_pos + 1) % dry_len;

            if mono {
                let s: f32 = self.dry_in.iter().sum();
                self.x_model[0] = s / self.channels as f32;
            } else {
                self.x_model[..self.channels].copy_from_slice(&self.dry_in);
            }

            // ---- denoiser ----
            let mut avail = false;
            if !use_df {
                let path = if mono { &mut self.live_mono } else { &mut self.live_multi };
                path.tick(&self.x_model[..mch], &mut self.y_model[..mch], eff.cut_db);
                avail = true;
            } else {
                if self.df_state == DfState::Flushing {
                    self.poll_flush();
                    if self.df_state == DfState::Flushing && self.blocking {
                        let t0 = Instant::now();
                        while self.df_state == DfState::Flushing
                            && t0.elapsed() < Duration::from_secs(10)
                            && !self.sh.failed.load(Ordering::Acquire)
                        {
                            std::thread::sleep(Duration::from_micros(200));
                            self.poll_flush();
                        }
                    }
                }
                if self.df_state == DfState::Running {
                    let mut ok = true;
                    if self.to_worker.slots() >= mch {
                        for c in 0..mch {
                            let _ = self.to_worker.push(self.x_model[c]);
                        }
                    } else {
                        ok = false;
                    }
                    self.since_unpark += 1;
                    if self.since_unpark >= chunk_in {
                        self.since_unpark = 0;
                        self.worker_thread.unpark();
                    }
                    if !ok {
                        // Worker hopelessly behind: start over cleanly.
                        self.start_flush(mch);
                    } else {
                        self.pushed += 1;
                        if self.pushed > self.df_wait {
                            if self.blocking && self.from_worker.slots() < mch {
                                self.worker_thread.unpark();
                                let t0 = Instant::now();
                                while self.from_worker.slots() < mch
                                    && t0.elapsed() < Duration::from_secs(5)
                                    && !self.sh.failed.load(Ordering::Acquire)
                                {
                                    std::thread::sleep(Duration::from_micros(100));
                                }
                            }
                            // Skip samples that arrived too late to be used.
                            while self.debt > 0 && self.from_worker.slots() >= mch {
                                for _ in 0..mch {
                                    let _ = self.from_worker.pop();
                                }
                                self.debt -= 1;
                            }
                            if self.debt == 0 && self.from_worker.slots() >= mch {
                                for c in 0..mch {
                                    self.y_model[c] = self.from_worker.pop().unwrap_or(0.0);
                                }
                                avail = true;
                            } else {
                                self.debt += 1;
                                dropouts += 1;
                            }
                        }
                    }
                }
            }

            // ---- output stage: crossfade, release, mix ----
            self.xf += ((if avail { 1.0 } else { 0.0 }) - self.xf) * a_xf;
            self.mix_s += (mix_target - self.mix_s) * a_mix;
            let mut ad = 0.0f32;
            let mut ap = 0.0f32;
            for ch in 0..self.channels {
                let d = self.dry_out[ch];
                let w = if avail { self.y_model[if mono { 0 } else { ch }] } else { d };
                let w = if w.is_finite() { w } else { d };
                let p = d + (w - d) * self.xf;
                self.proc[ch] = p;
                ad += d.abs();
                ap += p.abs();
            }
            self.env_d += (ad - self.env_d) * a_env;
            self.env_p += (ap - self.env_p) * a_env;
            let r = if self.env_d > 1e-7 { (self.env_p / self.env_d).min(1.0) } else { 1.0 };
            let a = if r > self.g { a_att } else { a_rel };
            self.g += (r - self.g) * a;
            let fill = (self.g - r).max(0.0);
            for ch in 0..self.channels {
                let d = self.dry_out[ch];
                let p = self.proc[ch] + d * fill;
                let out = d + (p - d) * self.mix_s;
                if ch < buffers.len() {
                    buffers[ch][i] = out;
                }
            }
        }
        if dropouts > 0 {
            // Count events, not samples: one late chunk = one dropout.
            self.status.dropouts.fetch_add(1, Ordering::Relaxed);
        }
        self.publish_status();
    }
}
