# Daniel

A real-time rice difficulty calculator for 4k osu!mania which runs on its own with osu!stable on Windows. osu!lazer and osu!stable on Linux need [tosu](https://tosu.app).

**[Download](https://github.com/TheBagelOfMan/Daniel/releases/latest)**

## Requirements
Windows 10/11. osu!stable works on its own. osu!lazer needs tosu running.
Steps: download Daniel-v2.1.exe, run it next to osu!, select a map.

## Usage
Tab cycles through 3 views (compact, statistics, graph) and L toggles LN on or off.
Dans run from 4th to Kappa with tiers of Low, Mid and High. Red lines on the graph are pauses.
DT, HT, and lazer custom rates are all supported.

## How it works
As of v2.0 the difficulty algorithm is [XDC1](https://github.com/TheBagelOfMan/xdc1) which is an algorithm built from the ground up for 4K osu!mania.
XDC1 computes an overall difficulty, a strain against time graph and 7 unique skillsets (of which it shows the most relevant ones). It works by accurately modelling the different tapping techniques a player may use per hand such as using fingers, wrists or jittering and then adds hold and stamina strain on top. 
The graph displayed on Daniel is the strain without the stamina included.
The calculation happens when the map is selected so there should be little impact on performance.

## Accuracy
Daniel is now tested on LeoBlackMT's [VSRG Dan Estimation Benchmark](https://benchmark.leoblack.top) which compares an algorithm's rating with the labelled dan of each map. MAE (mean absolute error) is the average distance from the labelled dan so 0.17 means that Daniel is off by around a sixth of a dan on average. Note that on the site Daniel (v1) was not fitted to the scale properly and its actual results (excluding marathon courses) are below. On the site v2 is XDC1.

Daniel is built primarily for 4K rice from Alpha to Theta so the set I've measured them on is: rice maps with a labelled dan of 10.5 to 18.5 (11 to 19 in Daniel's scale) excluding marathon courses.

| 4K rice, Alpha to Theta (10.5 to 18.5), excluding marathon courses | Daniel v1 | Daniel v2 (XDC1) |
|---|---|---|
| MAE | 0.230 | **0.174** |
| Within 0.2 of the labelled dan | 58% | **68%** |
| Within half a dan | 90% | **95%** |
| Within a full dan | 98% | **99.6%** |

Both versions were scored on the same 1,115 maps. XDC1's official result on this set is 0.173 MAE and 0.180 across all rated rice maps. As of November 2026 it is the lowest rice MAE on the benchmark by a decent margin.

### Please note
- Marathon dan courses are overrated due to the stamina model. Daniel is intended for individual maps and not courses.
- Long notes are deliberately not fitted to the benchmark's LN scale as I believe other algorithms tend to rate LNs too high.

## Building from source
### Windows
Needs Python 3 and the [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ workload).

```bat
build_windows.bat
```
This builds XDC1 then packages Daniel into `dist\Daniel.exe`.

### XDC1
The difficulty algorithm XDC1 is a C++ program in `xdc1/`. On Windows, `build_xdc1.bat` builds `src/xdc1.exe` with MSVC 2022. On Linux, `build_linux.sh` builds `src/xdc1` with g++. Set `XDC1_BIN_PATH` to use a binary from somewhere else if needed.

### Linux build
Use `build_linux.sh` to create a Linux binary:

```bash
./build_linux.sh
```

This installs dependencies in a Python venv and outputs `dist/Daniel-linux`. If g++ is missing, set `XDC1_BIN_PATH` at runtime to a Linux-compatible `xdc1` executable. On Linux/macOS it will use Wine if only `src/xdc1.exe` is available.

The Linux build of v2.0 hasn't been tested yet.

## Credits
- TheBagelOfMan
- [tosu](https://tosu.app) (KotRik, Cherry)
- LeoBlackMT, for [VSRG Dan Estimation Benchmark](https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark) and the osu! memory reading approach from [osumania map analyser](https://github.com/LeoBlackMT/osumania_map_analyser)

## License

[MIT](LICENSE)
