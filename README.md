# Daniel

A real-time rice difficulty calculator for 4k osu!mania must which must be run alongside [tosu](https://tosu.app).

**[Website](https://thebagelofman.github.io/Daniel/)** · **[Download](https://github.com/TheBagelOfMan/Daniel/releases/latest)**

## Linux build

Use `build_linux.sh` to create a Linux binary:

```bash
./build_linux.sh
```

This script installs dependencies using python venv and outputs `dist/Daniel-linux`. If `src/msd` is missing, set `MSD_BIN_PATH` at runtime to a Linux-compatible `msd` executable.

On Linux/macOS, it will use Wine if only `src/msd.exe` is available.

## License

[MIT](LICENSE)
