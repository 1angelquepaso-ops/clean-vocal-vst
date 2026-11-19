//! Offline test harness: runs a WAV through the exact plugin engine, then removes the reported
//! latency the same way a DAW's delay compensation does.
//!
//! Usage: cv_offline <in.wav> <out.wav> [options]
//!   --mode live|standard|hq|gentle   (default standard)
//!   --cut <dB>  --release <ms>  --mix <0-100>  --extra 0|1  --humcut 0|1  --hum <Hz>  --mono 0|1
//!   --host render|fast|paced         render: host says "offline" (default)
//!                                    fast:   host says "realtime" but calls as fast as it can
//!                                            (checks render detection)
//!                                    paced:  real-time playback, one block per block-duration
//!   --block <n>                      paced/fast block size (default 128)
//!   --switch <ms>:<mode>             change mode part-way through (repeatable)

use clean_vocal::engine::{Engine, Mode, Settings, Status};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn mode(s: &str) -> Mode {
    match s {
        "live" => Mode::Live,
        "hq" => Mode::Hq,
        "gentle" => Mode::Gentle,
        _ => Mode::Standard,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: cv_offline in.wav out.wav [--mode ..] [--cut ..] [--host render|fast|paced] ...");
        std::process::exit(2);
    }
    let mut s = Settings::default();
    let mut host = "render".to_string();
    let mut block = 128usize;
    let mut switches: Vec<(f64, Mode)> = vec![];
    let mut i = 3;
    while i + 1 < a.len() {
        let v = &a[i + 1];
        match a[i].as_str() {
            "--mode" => s.mode = mode(v),
            "--cut" => s.cut_db = v.parse().unwrap(),
            "--release" => s.release_ms = v.parse().unwrap(),
            "--mix" => s.mix = v.parse::<f32>().unwrap() / 100.0,
            "--extra" => s.extra_clean = v == "1",
            "--humcut" => s.hum_cut = v == "1",
            "--hum" => s.hum_cut_hz = v.parse().unwrap(),
            "--mono" => s.mono = v == "1",
            "--host" => host = v.clone(),
            "--block" => block = v.parse().unwrap(),
            "--switch" => {
                let (t, m) = v.split_once(':').unwrap();
                switches.push((t.parse().unwrap(), mode(m)));
            }
            o => panic!("unknown option {o}"),
        }
        i += 2;
    }

    let mut r = hound::WavReader::open(&a[1]).expect("open input");
    let spec = r.spec();
    let ch = spec.channels as usize;
    let sr = spec.sample_rate as usize;
    let inter: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|x| x.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|x| x.unwrap() as f32 * scale).collect()
        }
    };
    let frames = inter.len() / ch;
    let mut chans: Vec<Vec<f32>> =
        (0..ch).map(|c| (0..frames).map(|i| inter[i * ch + c]).collect()).collect();

    let status = Arc::new(Status::default());
    let t0 = Instant::now();
    let max_block = if host == "render" { 2048 } else { block };
    let mut eng = Engine::new(sr, ch, max_block, host == "render", s, status.clone()).expect("engine");
    let init_ms = t0.elapsed().as_millis();
    eng.set_settings(s);
    let lat0 = eng.latency_samples();

    // Pad so the tail comes out, like a DAW render.
    let pad = sr; // 1 s covers any mode
    for c in chans.iter_mut() {
        c.extend(std::iter::repeat(0.0).take(pad));
    }
    let total = frames + pad;

    // Irregular block sizes for renders (like a DAW); fixed for paced/fast.
    let sizes = [512usize, 64, 1024, 333, 128, 2048, 17, 256];
    let t1 = Instant::now();
    let mut pos = 0;
    let mut k = 0;
    let mut late_blocks = 0usize;
    let mut worst_block_ms = 0f64;
    // Latency in effect for each output sample (changes on mode switch).
    let mut lat_at: Vec<usize> = Vec::with_capacity(total);
    while pos < total {
        let n = if host == "render" { sizes[k % sizes.len()] } else { block }.min(total - pos);
        k += 1;
        let tsec = pos as f64 / sr as f64 * 1000.0;
        for (t, m) in switches.iter() {
            if (tsec - t).abs() < (n as f64 / sr as f64 * 1000.0) && tsec >= *t {
                s.mode = *m;
            }
        }
        eng.set_settings(s);
        let tb = Instant::now();
        let mut views: Vec<&mut [f32]> = chans.iter_mut().map(|c| &mut c[pos..pos + n]).collect();
        eng.process(&mut views);
        let el = tb.elapsed().as_secs_f64();
        let block_s = n as f64 / sr as f64;
        worst_block_ms = worst_block_ms.max(el * 1000.0);
        if el > block_s {
            late_blocks += 1;
        }
        lat_at.extend(std::iter::repeat(eng.latency_samples()).take(n));
        pos += n;
        if host == "paced" {
            let due = t1 + Duration::from_secs_f64(pos as f64 / sr as f64);
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
        }
    }
    let proc_s = t1.elapsed().as_secs_f64();

    // Delay compensation (uses the latency the engine reported at the start, like FL).
    let out_spec = hound::WavSpec {
        channels: ch as u16,
        sample_rate: sr as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(&a[2], out_spec).expect("create output");
    for i in 0..frames {
        for c in 0..ch {
            w.write_sample(chans[c][i + lat0]).unwrap();
        }
    }
    w.finalize().unwrap();
    let audio_s = frames as f64 / sr as f64;
    let lat_end = *lat_at.last().unwrap_or(&lat0);
    eprintln!(
        "{{\"init_ms\":{init_ms},\"latency\":{lat0},\"latency_ms\":{:.1},\"latency_end\":{lat_end},\
         \"audio_s\":{audio_s:.2},\"proc_s\":{proc_s:.2},\"speed_x\":{:.2},\"dropouts\":{},\
         \"late_blocks\":{late_blocks},\"worst_block_ms\":{worst_block_ms:.2}}}",
        lat0 as f64 * 1000.0 / sr as f64,
        audio_s / proc_s,
        status.dropouts.load(Ordering::Relaxed)
    );
}
