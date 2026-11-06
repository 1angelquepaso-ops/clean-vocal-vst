//! Offline test harness: runs a WAV through the exact plugin engine, feeding it in
//! irregular block sizes like a DAW would, then removes the reported latency the same way
//! a DAW's delay compensation does.
//!
//! Usage: cv_offline <in.wav> <out.wav> [strength 0-100] [extra 0|1] [humcut 0|1] [hum_hz]

use clean_vocal::engine::{Engine, Settings};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: cv_offline in.wav out.wav [strength] [extra] [humcut] [hum_hz]");
        std::process::exit(2);
    }
    let mut s = Settings::default();
    if let Some(v) = a.get(3) { s.strength = v.parse::<f32>().unwrap() / 100.0; }
    if let Some(v) = a.get(4) { s.extra_clean = v == "1"; }
    if let Some(v) = a.get(5) { s.hum_cut = v == "1"; }
    if let Some(v) = a.get(6) { s.hum_cut_hz = v.parse().unwrap(); }

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
    let mut chans: Vec<Vec<f32>> = (0..ch).map(|c| (0..frames).map(|i| inter[i * ch + c]).collect()).collect();

    let t0 = std::time::Instant::now();
    let mut eng = Engine::new(sr, ch).expect("engine");
    let init_ms = t0.elapsed().as_millis();
    eng.set_settings(s);
    let lat = eng.latency_samples();

    // Append latency worth of silence so the tail comes out, like a DAW render.
    for c in chans.iter_mut() { c.extend(std::iter::repeat(0.0).take(lat)); }
    let total = frames + lat;

    // Irregular block sizes, like a DAW.
    let sizes = [512usize, 64, 1024, 333, 128, 2048, 17, 256];
    let t1 = std::time::Instant::now();
    let mut pos = 0;
    let mut k = 0;
    while pos < total {
        let n = sizes[k % sizes.len()].min(total - pos);
        k += 1;
        let mut views: Vec<&mut [f32]> = chans.iter_mut().map(|c| &mut c[pos..pos + n]).collect();
        eng.process(&mut views);
        pos += n;
    }
    let proc_s = t1.elapsed().as_secs_f64();

    // Delay compensation: drop the first `lat` samples.
    let out_spec = hound::WavSpec { channels: ch as u16, sample_rate: sr as u32, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut w = hound::WavWriter::create(&a[2], out_spec).expect("create output");
    for i in 0..frames {
        for c in 0..ch { w.write_sample(chans[c][i + lat]).unwrap(); }
    }
    w.finalize().unwrap();
    let audio_s = frames as f64 / sr as f64;
    eprintln!(
        "sr {sr} ch {ch} latency {lat} samples ({:.1} ms) | init {init_ms} ms | processed {:.1}s of audio in {:.1}s (realtime x{:.1})",
        lat as f64 * 1000.0 / sr as f64, audio_s, proc_s, audio_s / proc_s
    );
}
