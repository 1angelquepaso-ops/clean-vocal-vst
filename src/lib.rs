//! Clean Vocal: AI vocal cleaner plugin (VST3 / CLAP), powered by DeepFilterNet3 and RNNoise.

pub mod editor;
pub mod engine;

use engine::{Engine, Mode, Settings, Status};
use nih_plug::prelude::*;
use std::num::NonZeroU32;
use std::sync::Arc;

#[derive(Enum, Debug, PartialEq, Eq, Clone, Copy)]
pub enum ModeParam {
    #[name = "LIVE"]
    Live,
    #[name = "STANDARD"]
    Standard,
    #[name = "HQ"]
    Hq,
    #[name = "GENTLE"]
    Gentle,
}

impl From<ModeParam> for Mode {
    fn from(m: ModeParam) -> Self {
        match m {
            ModeParam::Live => Mode::Live,
            ModeParam::Standard => Mode::Standard,
            ModeParam::Hq => Mode::Hq,
            ModeParam::Gentle => Mode::Gentle,
        }
    }
}

pub struct CleanVocal {
    params: Arc<CleanVocalParams>,
    status: Arc<Status>,
    engine: Option<Engine>,
    engine_cfg: (usize, usize, usize, bool), // (sample rate, channels, max block, offline)
    reported_latency: u32,
}

#[derive(Params)]
pub struct CleanVocalParams {
    /// Remembers the panel size/state with the project.
    #[persist = "editor-state"]
    pub editor_state: Arc<nih_plug_egui::EguiState>,

    /// LIVE (low latency, for tracking), STANDARD, HQ (strongest, for bouncing), GENTLE.
    #[id = "mode"]
    pub mode: EnumParam<ModeParam>,

    /// Most the cleaner may turn noise down. 60 dB = no limit.
    #[id = "cut"]
    pub cut: FloatParam,

    /// How slowly the cleaning closes after a phrase. Keeps word tails and breaths natural.
    #[id = "release"]
    pub release: FloatParam,

    /// Dry/wet.
    #[id = "mix"]
    pub mix: FloatParam,

    /// Extra cleanup pass on what's left between phrases (STANDARD).
    #[id = "extra"]
    pub extra_clean: BoolParam,

    /// Cut rumble and electrical hum below the frequency below.
    #[id = "humcut"]
    pub hum_cut: BoolParam,

    #[id = "humhz"]
    pub hum_cut_hz: FloatParam,

    /// Clean L+R together (half the CPU). Right for a mono mic.
    #[id = "mono"]
    pub mono: BoolParam,
}

impl Default for CleanVocalParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),
            mode: EnumParam::new("Mode", ModeParam::Standard),
            cut: FloatParam::new(
                "Max Cut",
                18.0,
                FloatRange::Skewed { min: 0.0, max: 60.0, factor: FloatRange::skew_factor(-0.8) },
            )
            .with_unit(" dB")
            .with_step_size(0.5)
            .with_value_to_string(Arc::new(|v| {
                if v >= 59.5 { "MAX".into() } else { format!("{v:.1}") }
            })),
            release: FloatParam::new(
                "Release",
                100.0,
                FloatRange::Skewed { min: 20.0, max: 500.0, factor: FloatRange::skew_factor(-1.0) },
            )
            .with_unit(" ms")
            .with_step_size(1.0),
            mix: FloatParam::new("Mix", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_step_size(1.0),
            extra_clean: BoolParam::new("Extra Clean", false),
            hum_cut: BoolParam::new("Hum Cut", true),
            hum_cut_hz: FloatParam::new(
                "Hum Cut Freq",
                75.0,
                FloatRange::Skewed { min: 40.0, max: 150.0, factor: FloatRange::skew_factor(-1.0) },
            )
            .with_unit(" Hz")
            .with_step_size(1.0),
            mono: BoolParam::new("Mono", true),
        }
    }
}

impl Default for CleanVocal {
    fn default() -> Self {
        Self {
            params: Arc::new(CleanVocalParams::default()),
            status: Arc::new(Status::default()),
            engine: None,
            engine_cfg: (0, 0, 0, false),
            reported_latency: 0,
        }
    }
}

impl CleanVocal {
    fn settings(&self) -> Settings {
        let p = &self.params;
        Settings {
            mode: p.mode.value().into(),
            cut_db: p.cut.value(),
            extra_clean: p.extra_clean.value(),
            release_ms: p.release.value(),
            mix: p.mix.value() / 100.0,
            hum_cut: p.hum_cut.value(),
            hum_cut_hz: p.hum_cut_hz.value(),
            mono: p.mono.value(),
        }
    }
}

impl Plugin for CleanVocal {
    const NAME: &'static str = "Clean Vocal";
    const VENDOR: &'static str = "Painters Near You Audio";
    const URL: &'static str = "https://github.com/1angelquepaso-ops/clean-vocal-vst";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = false;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self.params.clone(), self.status.clone())
    }

    fn initialize(
        &mut self,
        layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate.round() as usize;
        let ch = layout.main_output_channels.map(|c| c.get() as usize).unwrap_or(2);
        let max_block = buffer_config.max_buffer_size as usize;
        let offline = buffer_config.process_mode == ProcessMode::Offline;
        let cfg = (sr, ch, max_block, offline);
        if self.engine.is_none() || self.engine_cfg != cfg {
            // Drop the old engine (and its worker thread) before loading a new model.
            self.engine = None;
            self.engine = if Engine::supports_rate(sr) {
                match Engine::new(sr, ch, max_block, offline, self.settings(), self.status.clone()) {
                    Ok(e) => Some(e),
                    Err(err) => {
                        nih_log!("Clean Vocal: engine init failed: {err}");
                        None
                    }
                }
            } else {
                nih_log!("Clean Vocal: unsupported sample rate {sr}, passing audio through");
                None
            };
            self.engine_cfg = cfg;
        }
        let latency = self.engine.as_ref().map(|e| e.latency_samples()).unwrap_or(0) as u32;
        context.set_latency_samples(latency);
        self.reported_latency = latency;
        true
    }

    fn reset(&mut self) {
        if let Some(e) = self.engine.as_mut() {
            e.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let settings = self.settings();
        if let Some(engine) = self.engine.as_mut() {
            engine.set_settings(settings);
            engine.process(buffer.as_slice());
            // Switching between LIVE and the AI modes changes the delay; tell FL.
            let lat = engine.latency_samples() as u32;
            if lat != self.reported_latency {
                self.reported_latency = lat;
                context.set_latency_samples(lat);
            }
        }
        ProcessStatus::Normal
    }
}

impl ClapPlugin for CleanVocal {
    const CLAP_ID: &'static str = "com.paintersnearyou.clean-vocal";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("AI vocal cleaner: removes hiss, hum, room noise");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] =
        &[ClapFeature::AudioEffect, ClapFeature::Restoration, ClapFeature::Stereo, ClapFeature::Mono];
}

impl Vst3Plugin for CleanVocal {
    const VST3_CLASS_ID: [u8; 16] = *b"CleanVocalAngel1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Restoration];
}

nih_export_clap!(CleanVocal);
nih_export_vst3!(CleanVocal);
