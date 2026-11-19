# Clean Vocal

AI vocal cleaner plugin (VST3 + CLAP), v3, for FL Studio and other DAWs. Removes hiss, hum, fan and
room noise from vocals while leaving the voice alone. Runs fully offline on your own computer:
no account, no internet, no credits.

Powered by the DeepFilterNet3 speech-enhancement model (and RNNoise for LIVE mode).

## Install (Windows)

1. Go to this repo's **Releases** page and download `Clean-Vocal-Windows.zip`.
2. Unzip it.
3. Copy the **`Clean Vocal.vst3`** folder into `C:\Program Files\Common Files\VST3\`.
4. In FL Studio: **Options → Manage plugins → Find more plugins** (or Start scan).
5. Load **Clean Vocal** on your vocal mixer track.

## Modes

| Mode | Engine | Delay | Use it for |
|---|---|---|---|
| **LIVE** | RNNoise (light) | ~30 ms | Hearing yourself through the plugin while recording |
| **STANDARD** | DeepFilterNet3 | ~50 ms render, ~65 ms live at a 128 buffer | Everyday mixing |
| **HQ** | DeepFilterNet3, strongest (cut at least 30 dB, Extra Clean on, faster release) | same as STANDARD | Bouncing a very noisy take |
| **GENTLE** | DeepFilterNet3, cut capped at 8 dB, slow release | same as STANDARD | Natural sound, keeps air and breaths |

The delay is reported to FL, which lines the vocal back up automatically. Switching between LIVE and
the other modes changes the delay, so switch while stopped if you can.

## Controls

The panel: click a mode at the top, drag a knob up/down to turn it (hold Shift for fine control),
double-click a knob to reset it, click a switch to flip it. The top-right corner shows the current
delay and how many times the AI had to catch up (see below).

| Control | What it does | Default |
|---|---|---|
| Max Cut | The most the plugin may turn noise down. Higher = cleaner but more "gated". MAX = no limit. | 18 dB |
| Release | How slowly it closes in after a phrase ends, so word endings and tails don't get chopped. | 100 ms |
| Mix | Dry/wet. | 100% |
| Hum Freq | Where the hum cut starts. 75 Hz suits most voices. | 75 Hz |
| Extra Clean | A stronger final pass between phrases (STANDARD; always on in HQ). | Off |
| Hum Cut | Cuts rumble and electrical hum below Hum Freq. | On |
| Mono | Cleans L+R together: half the CPU. Right for a mono mic. Turn off for real stereo sources. | On |

## Good to know

- Works at any normal project sample rate (44.1k, 48k, 88.2k, 96k...).
- The AI (STANDARD / HQ / GENTLE) runs on its own thread, so it never makes FL's audio crackle.
  If the computer is ever too busy, the plugin briefly lets the (hum-cut) dry vocal through with
  a smooth crossfade instead of popping, and counts it as a "catch-up" on the panel. Lots of
  catch-ups: raise FL's buffer, turn Mono on, or use LIVE.
- Renders are always fully processed: when FL bounces, the plugin waits for the AI.
- Playback stop/start no longer reloads the AI, so there's no hitch when you hit record.

## Changes in v3

- Removed the hidden hard gate (frames the AI thought were noise were fully muted, and frames
  quieter than about -70 dBFS were zeroed with the AI's memory frozen). Cut depth is now set only
  by Max Cut, with a Release so tails fade naturally.
- Every frame is processed (no more flipping between cleaned and raw on loud phrases).
- AI moved off the audio thread; dropouts crossfade to dry instead of clicking.
- New modes (LIVE, HQ, GENTLE), Mix, Release and Mono; all knobs are smoothed; Hum Cut on/off
  is crossfaded.

## Build from source

`cargo xtask bundle clean_vocal --release` (bundles land in `target/bundled/`). Every push to
`main` builds Windows bundles automatically with GitHub Actions.

## License

GPL-3.0-or-later (required by the VST3 bindings). Third-party parts: DeepFilterNet (MIT/Apache-2.0,
Hendrik Schröter), nnnoiseless / RNNoise (BSD-3-Clause), nih-plug (ISC, Robbert van der Helm), VST3 bindings (GPLv3).
