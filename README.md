# Daniel

A real-time difficulty calculator and pattern recognizer for 4K and 7K osu!mania, designed to run alongside [tosu](https://tosu.app).

**[Website](https://thebagelofman.github.io/Daniel/)** · **[Download](https://github.com/TheBagelOfMan/Daniel/releases/latest)**

## Features

* **Native 7K Engine & 4K Support:** 7K calculations run on a 100% native Python engine, completely eliminating external dependencies for 7-key maps. 4K calculations continue to use the highly reliable Etterna `msd` engine.
* **Advanced Pattern Recognition:** Analyzes finger movements and detects 7K-specific patterns in real-time, including: *Chordstream, Chordjack, LNstream, Inverse, Bracket,* and *Stairs*.
* **Dynamic Map Categorization:** Automatically categorizes 7K maps into three realms based on their Long Note (LN) ratio:
  * **Rice Maps (<25% LN):** Focuses on detailed pure tapping pattern analysis.
  * **Hybrid Maps (25%-55% LN):** Displays a fair clash between heavy Rice patterns and grueling LN mechanics.
  * **Full LN Maps (>55% LN):** Disables Rice parameters to focus purely on *Inverse* and *LNstream* difficulties.
* **Smart UI Filtering:** The UI intelligently filters out less relevant skillsets based on the map's dominance (e.g., displaying Top 3 skills for complex Rice maps, but streamlining to Top 2 for heavy LN maps).
* **Anti-Spike Algorithm:** Uses a "Top 15% Average" calculation to ensure Star Rating and Skillset values are stable and not biased by short, sudden difficulty spikes.

## Building from Source

### Windows Build
To create a Windows standalone executable, make sure you have `pyinstaller` installed. Ensure `msd.exe` is in the same directory (required for 4K mode), then run:

```bash
pyinstaller --onefile --noconsole --add-data "msd.exe;." daniel.py

#### Linux Build 
Use build_linux.sh

