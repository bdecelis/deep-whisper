# Legal

This document covers third-party software licenses, terms of use, and
important notices regarding the responsible use of deep-whisper.

---

## Third-Party Software

deep-whisper depends on several open-source projects. Each carries its own
license and usage terms, which may impose obligations or restrictions beyond
those of the Apache 2.0 license under which deep-whisper itself is released.

**You are responsible for ensuring your use of deep-whisper complies with
the terms of all third-party licenses listed below.**

The most important restrictions to be aware of are noted in the table.
Links to full license texts are provided for each project.

| Project | License | Key restrictions |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) | [MIT](https://github.com/openai/whisper/blob/main/LICENSE) | Attribution required |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | [MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) | Attribution required |
| [WhisperX](https://github.com/m-bain/whisperX) | [BSD-4-Clause](https://github.com/m-bain/whisperX/blob/main/LICENSE) | Attribution required; no endorsement of derived products |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | [MIT](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE) | Attribution required |
| [Silero VAD](https://github.com/snakers4/silero-vad) | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) | Attribution required |
| [wav2vec2 / MMS](https://huggingface.co/facebook/wav2vec2-base-960h) | [Apache 2.0](https://huggingface.co/facebook/wav2vec2-base-960h) | Attribution required |
| [librosa](https://github.com/librosa/librosa) | [ISC](https://github.com/librosa/librosa/blob/main/LICENSE.md) | Attribution required |
| [PyTorch](https://github.com/pytorch/pytorch) | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Attribution required; no endorsement of derived products |
| [torchaudio](https://github.com/pytorch/audio) | [BSD-2-Clause](https://github.com/pytorch/audio/blob/main/LICENSE) | Attribution required |
| [num2words](https://github.com/savoirfairelinux/num2words) | [LGPL-2.1](https://github.com/savoirfairelinux/num2words/blob/master/COPYING) | If you distribute a modified version of num2words itself, the modified source must be made available under LGPL |
| [diff-match-patch](https://github.com/google/diff-match-patch) | [Apache 2.0](https://github.com/google/diff-match-patch/blob/master/LICENSE) | Attribution required |
| [soundfile](https://github.com/bastibe/python-soundfile) | [BSD-3-Clause](https://github.com/bastibe/python-soundfile/blob/master/LICENSE) | Attribution required |

### Notes on specific licenses

**WhisperX (BSD-4-Clause):** The BSD 4-Clause license includes a
"advertising clause" — you may not use the names of WhisperX or its
contributors to advertise or promote products that incorporate it without
specific written permission.

**num2words (LGPL-2.1):** The LGPL applies to modifications of num2words
itself, not to applications that use it as a library. If you use
deep-whisper without modifying num2words, no LGPL obligations apply to
your own code.

**Model weights:** The Whisper model weights distributed by OpenAI are
licensed under MIT. The wav2vec2 model weights are licensed under Apache 2.0.
Any models you download and use are subject to their own license terms,
which may differ from the code license.

---

## Terms of Use

By using deep-whisper you agree to the following:

### 1. Lawful use

You will use deep-whisper only for purposes that are lawful in your
jurisdiction. This includes but is not limited to compliance with:

- Laws governing the recording, transcription, and storage of audio
  (which vary significantly by country and state — many jurisdictions
  require the consent of all parties being recorded)
- Copyright law — you are responsible for ensuring you have the right to
  transcribe any audio you process
- Privacy and data protection laws — transcripts may constitute personal
  data under applicable regulations (e.g. GDPR, CCPA)
- Regulations governing the use of AI-generated content

### 2. Consent

When processing audio that contains recordings of individuals, you are
responsible for obtaining all necessary consent from those individuals
prior to processing. deep-whisper provides no mechanism to verify consent
and places this obligation entirely on the user.

### 3. No unlawful surveillance

You will not use deep-whisper to transcribe or process audio recordings
obtained without the knowledge or consent of the individuals involved,
except where explicitly permitted by applicable law.

### 4. No harmful or deceptive use

You will not use deep-whisper to produce transcripts that are intentionally
falsified, misleading, or designed to misrepresent what was said by an
individual. You will not use it in connection with the creation of
non-consensual synthetic media (deepfakes) or to produce content that
harasses, defames, or harms individuals.

### 5. Attribution

If you distribute software or services built upon deep-whisper, you must
retain the copyright notice and license as required by the Apache 2.0
License, and you should clearly communicate to your users that deep-whisper
and its dependencies are used.

---

## Misuse and Abuse — Disclaimer

deep-whisper is designed for legitimate transcription use cases including
content creation, accessibility, research, and development tooling. It is
released with the expectation that it will be used responsibly and lawfully.

**The authors of deep-whisper:**

- Accept no responsibility for how this software is used by others
- Make no warranties, express or implied, as to the fitness of this software
  for any particular purpose
- Accept no liability for any direct, indirect, incidental, or consequential
  damages arising from the use or misuse of this software
- Do not endorse, support, or condone any use of this software that violates
  applicable laws, infringes on the rights of individuals, or causes harm to
  any person or organisation
- Reserve the right to take appropriate action — including revoking access
  to official distributions — in response to documented misuse

This disclaimer applies to the maximum extent permitted by applicable law.
It does not limit the terms of the Apache 2.0 License, which governs the
rights granted to use, copy, modify, and distribute this software.

---

## Contributing

By submitting a contribution to this project (pull request, patch, or
otherwise), you agree that:

- Your contribution is your original work, or you have the right to
  contribute it
- You grant the project maintainers a perpetual, royalty-free license to
  include your contribution under the Apache 2.0 License
- You accept that your contribution may be modified, redistributed, or
  removed at the maintainers' discretion

---

*This document is provided for informational purposes and does not
constitute legal advice. If you have questions about your legal obligations,
consult a qualified legal professional.*
