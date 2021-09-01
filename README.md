<img src="images/wake_word_detect.png">

# Table of Contents
- [Background](#background)
- [Introduction](#introduction)
- [Implementation](#implementation)
- [Conclusion](#conclusion)
- [Enhancements](#enhancements)

# Background
Personal Assistant devices like Google Home, Alexa and Apple Homepod, will be constantly listening for specific set of wake words like “Ok, Google” or “Alexa” or “Hey Siri”, and once these sequence of words are detected it would prompt to user for next commands and respond to them appropriately.

# Introduction
To create a open-source custom wake word detector, which will take audio as input and once the sequence of words are detected then prompt to the user. <br>

Goal is to provide configurable custom detector so that anyone can use it on their own application to perform operations, once configured wake words are detected.

# Implementation
- [Prepare labelled dataset from Mozilla Common Voice](#prepare-labelled-dataset-from-mozilla-common-voice)
- [Do word alignment](#do-word-alignment)
- [Fix any data imbalance](#fix-any-data-imbalance)
- [Extract audio features](#extract-audio-features)
- [Do transformations](#do-transformations)
- [Define model architecture](#define-model-architecture)
- [Train model using train and validation dataset](#train-model-using-train-and-validation-dataset)
- [Get results using test dataset](#get-results-using-test-dataset)
- [Deploy model and do Inference](#deploy-model-and-do-inference)

## Prepare labelled dataset from Mozilla Common Voice
Used Mozilla Common Voice dataset, 
- Go through each wake word and check transcripts for match
- If found then it will be in positive dataset
- If not found then it will be in negative dataset
- Load appropriate mp3 files and trim the silence parts
- save as .wav file and transcript as .lab file
- Code reference [fetch_dataset_mcv.py](train/fetch_dataset_mcv.py)
<img src="images/mcv-dataset.png">

## Do word alignment
- For positive dataset, used [Montreal Forced Alignment](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get timestamps of each word in audio.
- Download the [stable version](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases)
    ```
    wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
    tar -xf montreal-forced-aligner_linux.tar.gz
    rm montreal-forced-aligner_linux.tar.gz
    ```
- Download the [Librispeech Lexicon dictionary](https://www.openslr.org/resources/11/librispeech-lexicon.txt)
    ```
    wget https://www.openslr.org/resources/11/librispeech-lexicon.txt
    ```
- Known issues in MFA
    ```
    # known mfa issue https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/109
    cp montreal-forced-aligner/lib/libpython3.6m.so.1.0 montreal-forced-aligner/lib/libpython3.6m.so
    cd montreal-forced-aligner/lib/thirdparty/bin && rm libopenblas.so.0 && ln -s ../../libopenblasp-r0-8dca6697.3.0.dev.so libopenblas.so.0
    ```
- Creating aligned data
    ```
    montreal-forced-aligner\bin\mfa_align -q positive\audio librispeech-lexicon.txt montreal-forced-aligner\pretrained_models\english.zip aligned_data
    ```

<img src="images/montreal-forced-align.png">
Generated textgrid file 
<img src = "images/textgrid.png">

## Fix any data imbalance
Check for any data imbalance, if the dataset does not have enough samples containing wake words, consider using text to speech services to generate more samples. 
- Used Google Text To Speech Api, set environment variable `GOOGLE_APPLICATION_CREDENTIALS` with your key.
- Used various speed rates, pitches and voices to generate data for wake words. 
- Code [generate_dataset_google_tts.py](/train/generate_dataset_google_tts.py)

## Extract audio features
- Below is how sound looks like when plotted on time (x-axis) and amplitude (y-axis)
    ```python
    import librosa
    sounddata = librosa.core.load("hey.wav", sr=16000, mono=True)[0]

    # plotting the signal in time series
    plt.plot(sounddata)
    plt.title('Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    ```
    <img src='images/signal.png' width=400>
- When Short-time Fourier transform (STFT) computed, below is how spectrogram looks like
    ```python
    from torchaudio.transforms import Spectrogram
    spectrogram  = Spectrogram(n_fft=512,hop_length=200)
    spectrogram.to(device)

    inp = torch.from_numpy(sounddata).float().to(device)
    hey_spectrogram = spectrogram(inp.float())
    plot_spectrogram(hey_spectrogram.cpu(), title="Spectrogram")
    ```
    <img src='images/spectrogram.png' width=400>
- A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale.
    ```python
    from torchaudio.transforms import MelSpectrogram
    mel_spectrogram  = MelSpectrogram(n_mels=40,sample_rate=16000,
                                    n_fft=512,hop_length=200,
                                    norm="slaney")

    mel_spectrogram.to(device)
    inp = torch.from_numpy(sounddata).float().to(device)
    hey_mels_slaney = mel_spectrogram(inp.float())
    plot_spectrogram(hey_mels_slaney.cpu(), title="MelSpectrogram", ylabel='mel freq')
    ```
    <img src="images/melspectrogram.png" width=400>
- After adding offset and taking log on mels, below is how final mel spectrogram looks like
    ```python
    log_offset = 1e-7
    log_hey_mel_specgram = torch.log(hey_mels_slaney + log_offset)
    plot_spectrogram(log_hey_mel_specgram.cpu(), title="MelSpectrogram (Log)", ylabel='mel freq')
    ```
    <img src="images/logmelspectrogram.png" width=400>


## Do transformations
## Define model architecture
## Train model using train and validation dataset
## Get results using test dataset
## Deploy model and do Inference


# Conclusion


# Enhancements