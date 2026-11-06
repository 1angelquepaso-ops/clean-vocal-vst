# Clean Vocal

AI vocal cleaner plugin (VST3 + CLAP) for FL Studio and other DAWs. Removes hiss, hum, fan and
room noise from vocals while leaving the voice alone. Runs fully offline on your own computer:
no account, no internet, no credits.

Powered by the DeepFilterNet3 speech-enhancement model.

## Install (Windows)

1. Go to this repo's **Releases** page and download `Clean-Vocal-Windows.zip`.
2. Unzip it.
3. Copy the **`Clean Vocal.vst3`** folder into `C:\Program Files\Common Files\VST3\`.
4. In FL Studio: **Options → Manage plugins → Find more plugins** (or Start scan).
5. Load **Clean Vocal** on your vocal mixer track.

## Controls

| Control | What it does | Default |
|---|---|---|
| Strength | How much noise to remove. 100% removes everything that isn't voice; lower keeps some room/air. | 100% |
| Extra Clean | A stronger final pass on whatever is left between phrases. | On |
| Hum Cut | Cuts rumble and electrical hum below the Hum Cut Freq. | On |
| Hum Cut Freq | Where the hum cut starts. 75 Hz suits most voices. | 75 Hz |

## Good to know

- Works at any normal project sample rate (44.1k, 48k, 88.2k, 96k...).
- Adds about 50 ms of delay, which it reports to FL, so FL keeps the vocal lined up with the beat
  automatically.
- On a low-power PC, render (bounce) the cleaned vocal to audio once and then turn the plugin off,
  instead of leaving it running live.
- Renders are always fully processed: the plugin never lets uncleaned audio through to keep up.

## Build from source

`cargo xtask bundle clean_vocal --release` (bundles land in `target/bundled/`). Every push to
`main` builds Windows bundles automatically with GitHub Actions.

## License

GPL-3.0-or-later (required by the VST3 bindings). Third-party parts: DeepFilterNet (MIT/Apache-2.0,
Hendrik Schröter), nih-plug (ISC, Robbert van der Helm), VST3 bindings (GPLv3).
