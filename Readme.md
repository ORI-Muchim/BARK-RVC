- # BARK-RVC

Multilingual-Speech-Synthesis-Voice-Conversion Using [Bark](https://github.com/suno-ai/bark) + [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

## Contents
- [Installation](#installation)
- [Text-Prompt & Support-Languages](#text-prompt--support-languages)
- [Usage](#usage)
- [To-do](#to-do)
- [Reference](#reference)

## Installation
- A Windows/Linux system with a minimum of `16GB` RAM.
- A GPU with at least `12GB` of VRAM.
- Python >= 3.8
- Anaconda installed.
- Pytorch installed.
- CUDA 11.7 installed.

Pytorch install command:
```sh
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```

CUDA 11.7 install:
`https://developer.nvidia.com/cuda-11-7-0-download-archive`

---

1. **Create an Anaconda environment:**

```sh
conda create -n barkrvc python=3.9
```

2. **Activate the environment:**

```sh
conda activate barkrvc
```

3. **Clone this repository to your local machine:**

```sh
git clone https://github.com/ORI-Muchim/BARK-RVC.git
```

4. **Navigate to the cloned directory:**

```sh
cd BARK-RVC
```

5. **Install the necessary dependencies:**

```sh
pip install -r requirements.txt
```

---

## Text-Prompt & Support-Languages

If you open `./main.py`, There is sample text. There are many text prompts in the bark.

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- CAPITALIZATION for emphasis of a word
- `[MAN]` and `[WOMAN]` to bias Bark toward male and female speakers, respectively

---

### Supported Languages

| Language | Status |
| --- | :---: |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |

Voice Presets can be found here:
```sh
https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
```

---

## Usage

Place the audio files `./datasets/{speaker_name}`.

.mp3 or .wav files are okay.


And, use the following command:

```sh
python main.py {speaker_name}
```

---

## To-do
- Audio-Upsampling Using [NU-Wave2](https://github.com/mindslab-ai/nuwave2)

---

## Reference

For more infomation, Please refer to the following repositories:
- [Bark](https://github.com/suno-ai/bark)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [NU-Wave2](https://github.com/mindslab-ai/nuwave2)
