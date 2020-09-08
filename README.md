

# WOW 1.12 external wireframe renderer reading game's memory

Caveats:
- World-to-screen formula involves a magic constant and thus works only for windowed-fullscreen 1920x1080 mode (see cam.py) (need to generalize the formula, https://wowdev.wiki/M2#Cameras maybe)
- Doesn't render animation (moving objects are rendered as static objects)
- Doesn't render equipement
- Doesn't render world
- Doesn't render NPCs
- Rendering is long, but it's fine if application is not real-time
- No way of knowing if something is occluded
- Made for windows (no macos)
- Game files (containing 3d models etc..) have to be manually extracted using a third party software
- Requires to be administrator (There might not be any other way)
- `Pymem` reads the process' memory using debugger (There might not be any other way)

With a complete rendering engine it would be possible record clips of the game accompanied
by the on-screen location of every objects for every frame, thus allowing to experiment with any
computer-vision task without worrying about the labelling of a dataset.

```bat
python truc.py
```
![fig.jpg](fig.jpg)

See `peep.py` for informations

# `bobbers/*`
A manually labeled screenshot dataset to train a CNN at localizing bobbers on screen. It contains only 135 images which is more than enough for training.

The images have been compressed and downsampled to be stored on github.

![fig.jpg]

# `fish.py`
A 100 line `WOW 1.12` fishing bot reading game's memory
