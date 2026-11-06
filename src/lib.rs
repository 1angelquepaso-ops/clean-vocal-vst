//! Clean Vocal: AI vocal cleaner plugin (VST3 / CLAP), powered by DeepFilterNet3.

pub mod editor;
pub mod engine;

use engine::{Engine, Settings};
use nih_plug::prelude::*;
use std::num::NonZeroU32;
use std::sync::Arc;

pub struct CleanVocal {
    params: Arc<CleanVocalParams>,
    engine: Option<Engine>,
    engine_cfg: (usize, usize), // (sample rate, channels) the engine was built for
}

#[derive(Params)]
pub struct CleanVocalParams {
    /// Remembers the panel size/state with the project.
    #[persist = "editor-state"]
    pub editor_state: Arc<nih_plug_egui::EguiState>,

    /// How much noise to remove. 100% = everything that isn't voice.
    #[id = "strength"]
    pub strength: FloatParam,

    /// Extra cleanup pass on what's left between phrases.
    #[id = "extra"]
    pub extra_clean: BoolParam,

    /// Cut rumble and electrical hum below the frequency below.
    #[id = "humcut"]
    pub hum_cut: BoolParam,

    #[id = "humhz"]
    pub hum_cut_hz: FloatParam,
}

impl Default for CleanVocalParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),
            strength: FloatParam::new("Strength", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_step_size(1.0),
            extra_clean: BoolParam::new("Extra Clean", true),
            hum_cut: BoolParam::new("Hum Cut", true),
            hum_cut_hz: FloatParam::new(
                "Hum Cut Freq",
                75.0,
                FloatRange::Skewed { min: 40.0, max: 150.0, factor: FloatRange::skew_factor(-1.0) },
            )
            .with_unit(" Hz")
            .with_step_size(1.0),
        }
    }
}

impl Default for CleanVocal {
    fn default() -> Self {
        Self { params: Arc::new(CleanVocalParams::default()), engine: None, engine_cfg: (0, 0) }
    }
}

impl CleanVocal {
    fn settings(&self) -> Settings {
        Settings {
            strength: self.params.strength.value() / 100.0,
            extra_clean: self.params.extra_clean.value(),
            hum_cut: self.params.hum_cut.value(),
            hum_cut_hz: self.params.hum_cut_hz.value(),
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
        editor::create(self.params.clone())
    }

    fn initialize(
        &mut self,
        layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate.round() as usize;
        let ch = layout.main_output_channels.map(|c| c.get() as usize).unwrap_or(2);
        if self.engine.is_none() || self.engine_cfg != (sr, ch) {
            self.engine = if Engine::supports_rate(sr) {
                match Engine::new(sr, ch) {
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
            self.engine_cfg = (sr, ch);
        }
        let latency = self.engine.as_ref().map(|e| e.latency_samples()).unwrap_or(0);
        context.set_latency_samples(latency as u32);
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
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let settings = self.settings();
        if let Some(engine) = self.engine.as_mut() {
            engine.set_settings(settings);
            engine.process(buffer.as_slice());
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
