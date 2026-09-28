# Fun with Filters and Frequencies

Frequency-domain filtering, hybrid images, and Laplacian pyramid blending.

## Setup

Put input images in `data/`. Results go in `results/`. Create the output folder a script writes to before running it with `save` set to `True`.

```bash
pip install numpy pillow scipy opencv-python matplotlib scikit-image
```

Boolean arguments are the strings `True` and `False`.

## Convolution

`homemade_convolution.py` filters `data/self.jpg` with a hand-written 2D convolution and writes `results/selfs/{action}_self.jpg`.

```bash
python homemade_convolution.py [action] [mode] [implementation]
```

| Arg | Options | Default |
|---|---|---|
| `action` | `blur`, `boxblur`, `dx`, `dy`, `none` | `none` |
| `mode` | `valid`, `same`, `full` | `same` |
| `implementation` | `TL`, `FL` | `TL` |

`TL` is the patch (cookie-cutter) version. `FL` is the flip-and-drag version.

```bash
python homemade_convolution.py blur same TL
python homemade_convolution.py dx same FL
```

## Edge detection

`edge_detection.py` finds edges on `data/cameraman.jpg`.

```bash
python edge_detection.py [blur_level] [threshold] [dog] [save]
```

| Arg | Options | Default |
|---|---|---|
| `blur_level` | odd Gaussian kernel size | `7` |
| `threshold` | fraction of the peak response kept | `0.1` |
| `dog` | `True`, `False` | `False` |
| `save` | `True`, `False` | `False` |

`dog` `True` convolves with pre-blurred derivative and Laplacian kernels. `False` blurs the image first, then applies the unblurred kernels.

With `save` `False`, the six results open in a window. With `save` `True`, they go in `results/blurred_filters_{blur_level}/` or `results/blurred_imgs_{blur_level}/`. The Laplacian kernel is always written to `results/filters/laplacian.jpg`.

```bash
python edge_detection.py 7 0.1 False False
python edge_detection.py 21 0.1 True True
```

## Sharpening

`image_sharpening.py` unsharp-masks an image in `data/`.

```bash
python image_sharpening.py [image] [blur_level] [sharp_level] [save] [blur_then_sharp] [viz_high_pass]
```

| Arg | Options | Default |
|---|---|---|
| `image` | filename in `data/` | `cameraman.jpg` |
| `blur_level` | odd Gaussian kernel size | `7` |
| `sharp_level` | high-pass gain | `1.0` |
| `save` | `True`, `False` | `False` |
| `blur_then_sharp` | `True`, `False` | `False` |
| `viz_high_pass` | `True`, `False` | `False` |

With `save` `True`, files go in `results/sharp_mask/`:

- `{blur}_{sharp}_{image}` — sharpened image
- `blurredfirst_{blur}_{sharp}_{image}` — blur, then sharpen (`blur_then_sharp` `True`)
- `high_pass_{blur}_{image}` — high-pass image (`viz_high_pass` `True`)

```bash
python image_sharpening.py taj.jpg 7 1.5 True False True
```

## Hybrid images

`hybrid_image.py` high-passes one image, low-passes another, and adds them. A window asks you to click two corresponding points on each image so they can be aligned.

```bash
python hybrid_image.py [pair] [f_hp] [f_lp] [hp_gain] [save] [fft]
```

| Arg | Options | Default |
|---|---|---|
| `pair` | `derek_and_nutmeg`, `anakin_and_vader`, `whitehouse` | `derek_and_nutmeg` |
| `f_hp` | high-pass cutoff, cycles/pixel | `0.1` |
| `f_lp` | low-pass cutoff, cycles/pixel | `0.01` |
| `hp_gain` | scale on the high-pass image | `2` |
| `save` | `True`, `False` | `False` |
| `fft` | `True`, `False` | `False` |

| Pair | High-pass | Low-pass |
|---|---|---|
| `derek_and_nutmeg` | `data/nutmeg.jpg` | `data/derek.jpg` |
| `anakin_and_vader` | `data/anakin.jpg` | `data/vader.jpg` |
| `whitehouse` | `data/wh_before.jpg` | `data/wh_after.jpg` |

With `save` `True`, the hybrid is `results/hybrid/{pair}.jpg`. With `fft` `True`, log-magnitude FFTs of the inputs, filtered images, and hybrid are written next to it.

```bash
python hybrid_image.py derek_and_nutmeg 0.1 0.01 2 True True
```

## Gaussian and Laplacian stacks

`stacks.py` builds a Gaussian stack and a Laplacian stack for one image in `data/`.

```bash
python stacks.py [image] [levels] [save]
```

| Arg | Options | Default |
|---|---|---|
| `image` | filename in `data/` | `apple.jpeg` |
| `levels` | number of filtered levels | `5` |
| `save` | `True`, `False` | `False` |

With `save` `True`, levels go in `results/gaussian_stacks/` and `results/laplacian_stacks/` as `level{i}_{image}`.

```bash
python stacks.py orange.jpeg 5 True
```

## Oraple blend

`oraple.py` blends two images across a straight seam with a Laplacian stack.

```bash
python oraple.py [levels] [save] [pair]
```

| Arg | Options | Default |
|---|---|---|
| `levels` | number of filtered levels | `4` |
| `save` | `True`, `False` | `False` |
| `pair` | `oraple`, `nyc`, `whitehouse` | `oraple` |

| Pair | Left / top | Right / bottom | Seam |
|---|---|---|---|
| `oraple` | `data/apple.jpeg` | `data/orange.jpeg` | vertical |
| `nyc` | `data/nyc_sat.jpg` | `data/nyc_map.jpg` | vertical |
| `whitehouse` | `data/wh_before.jpg` | `data/wh_after.jpg` | horizontal |

With `save` `True`, each stack level (left, right, blended) and the final image go in `results/{pair}/`.

```bash
python oraple.py 4 True oraple
```

## Mask blend

`mask_and_blend.py` blends an object into a background using a black mask on a white background.

```bash
python mask_and_blend.py [save] [name] [levels] [scale]
```

| Arg | Options | Default |
|---|---|---|
| `save` | `True`, `False` | `False` |
| `name` | `swamp`, `jeffryan`, `yung_arvind` | `swamp` |
| `levels` | number of filtered levels | `4` |
| `scale` | how fast the Gaussian cutoff drops | `0.5` |

Each name uses `data/{name}_obj.jpg`, `data/{name}_bkg.jpg`, and `data/{name}_msk.jpg`. Black pixels in the mask are the object.

With `save` `True`, the script creates `results/{name}/` and writes each stack level plus `results/{name}/{name}.jpg`.

```bash
python mask_and_blend.py True swamp 4 0.5
```
