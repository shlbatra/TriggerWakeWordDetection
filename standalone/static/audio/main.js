window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext();
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;

const classes = ["hey", "fourth", "brain", "oov"]
const wakeWords = ["hey", "fourth", "brain"]
const bufferSize = 1024
const channels = 1
const windowSize = 750
const zmuv_mean = 0.000016
const zmuv_std = 0.072771
const bias = 1e-7
const SPEC_HOP_LENGTH = 200;
const MEL_SPEC_BINS = 40;
const NUM_FFTS = 512;
const audioFloatSize = 32767
const sampleRate = 16000
const numOfBatches = 3

let predictWords = []
let arrayBuffer = []
let targetState = 0

const windowBufferSize = windowSize/1000 * sampleRate

let session;
async function loadModel() {
    session = new onnx.InferenceSession();
    await session.loadModel("static/audio/onnx_model.onnx");
}
loadModel()

const addprediction = function(word) {
    words = document.createElement('p');
    words.innerHTML = '<b>' + word + '</b>';
    document.getElementById('wavefiles').appendChild(words);
}

function toggleRecording( e ) {
    if (e.classList.contains('recording')) {
        // stop recording
        e.classList.remove('recording');
        recording = false;
    } else {
        // start recording
        document.getElementById('wavefiles').innerHTML = ""
        addprediction('Listening for wake words ...')
        e.classList.add('recording');
        recording = true;
    }
}

function convertToMono( input ) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect( splitter );
    splitter.connect( merger, 0, 0 );
    splitter.connect( merger, 0, 1 );
    return merger;
}

function cancelAnalyserUpdates() {
    window.cancelAnimationFrame( rafID );
    rafID = null;
}

function updateAnalysers(time) {
    if (!analyserContext) {
        var canvas = document.getElementById('analyser');
        canvasWidth = canvas.width;
        canvasHeight = canvas.height;
        analyserContext = canvas.getContext('2d');
    }

    // analyzer draw code here
    {
        var SPACING = 3;
        var BAR_WIDTH = 1;
        var numBars = Math.round(canvasWidth / SPACING);
        var freqByteData = new Uint8Array(analyserNode.frequencyBinCount);

        analyserNode.getByteFrequencyData(freqByteData); 

        analyserContext.clearRect(0, 0, canvasWidth, canvasHeight);
        analyserContext.fillStyle = '#F6D565';
        analyserContext.lineCap = 'round';
        var multiplier = analyserNode.frequencyBinCount / numBars;

        // Draw rectangle for each frequency bin.
        for (var i = 0; i < numBars; ++i) {
            var magnitude = 0;
            var offset = Math.floor( i * multiplier );
            // gotta sum/average the block, or we miss narrow-bandwidth spikes
            for (var j = 0; j< multiplier; j++)
                magnitude += freqByteData[offset + j];
            magnitude = magnitude / multiplier;
            var magnitude2 = freqByteData[i * multiplier];
            analyserContext.fillStyle = "hsl( " + Math.round((i*360)/numBars) + ", 100%, 50%)";
            analyserContext.fillRect(i * SPACING, canvasHeight, BAR_WIDTH, -magnitude);
        }
    }
    
    rafID = window.requestAnimationFrame( updateAnalysers );
}

function flatten(log_mels) {
    flatten_arry = []
    for(i = 0; i < MEL_SPEC_BINS; i++) {
        for(j = 0; j < log_mels.length; j++) {
            flatten_arry.push((Math.log(log_mels[j][i] + bias) - zmuv_mean) / zmuv_std)
        }
    }
    return flatten_arry
}

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}

const padArray = function(arr,len,fill) {
    return arr.concat(Array(len).fill(fill)).slice(0,len);
 }

function gotStream(stream) {
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;

    audioInput = convertToMono( audioInput );
    audioInput.connect(inputPoint);

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect( analyserNode );

    // bufferSize, in_channels, out_channels
    scriptNode = (audioContext.createScriptProcessor || audioContext.createJavaScriptNode).call(audioContext, bufferSize, channels, channels);
    scriptNode.onaudioprocess = async function (audioEvent) {
        if (recording) {
            let resampledMonoAudio = await resampleAndMakeMono(audioEvent.inputBuffer);
            arrayBuffer = [...arrayBuffer, ...resampledMonoAudio]
            batchSize = Math.floor(arrayBuffer.length/windowBufferSize)
            // if we got batches * 750 ms seconds of buffer 
            if (arrayBuffer.length >= numOfBatches * windowBufferSize) {
                let batch = 0
                let dataProcessed, log_mels;
                for (let i = 0; i < arrayBuffer.length; i = i + windowBufferSize) {
                    batchBuffer = arrayBuffer.slice(i, i+windowBufferSize)
                    //  if it is less than 750 ms then pad it with ones
                    if (batchBuffer.length < windowBufferSize) {
                        //batchBuffer = padArray(batchBuffer, windowBufferSize, 1)
                        // discard last slice
                        break
                    }
                    // arrayBuffer = arrayBuffer.filter(x => x/audioFloatSize)
                    // calculate log mels
                    log_mels = melSpectrogram(batchBuffer, {
                        sampleRate: sampleRate,
                        hopLength: SPEC_HOP_LENGTH,
                        nMels: MEL_SPEC_BINS,
                        nFft: NUM_FFTS
                    });
                    if (batch == 0) {
                        dataProcessed = []
                    }
                    dataProcessed = [...dataProcessed, ...flatten(log_mels)]
                    batch = batch + 1
                }
                // clear buffer
                arrayBuffer = []
                let inputTensor = new onnx.Tensor(dataProcessed, 'float32', [batch, 1, MEL_SPEC_BINS, dataProcessed.length/(batch * MEL_SPEC_BINS)]);
                // Run model with Tensor inputs and get the result.
                let outputMap = await session.run([inputTensor]);
                let outputData = outputMap.values().next().value.data;
                for (let i = 0; i<outputData.length; i = i+classes.length) {
                    let scores = Array.from(outputData.slice(i,i+classes.length))
                    console.log("scores", scores)
                    let probs = softmax(scores)
                    probs_sum = probs.reduce( (sum, x) => x+sum)
                    probs = probs.filter(x => x/probs_sum)
                    let class_idx = argMax(probs)
                    console.log("probabilities", probs)
                    console.log("predicted word", classes[class_idx])
                    if (classes[targetState] == classes[class_idx]) {
                        console.log(classes[class_idx])
                        addprediction(classes[class_idx])
                        predictWords.push(classes[class_idx]) 
                        targetState += 1
                        if (wakeWords.join(' ') == predictWords.join(' ')) {
                            addprediction(`Wake word detected - ${predictWords.join(' ')}`)
                            predictWords = []
                            targetState = 0
                        }
                    }
                }
            }
        }
    }
    inputPoint.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect( zeroGain );
    zeroGain.connect( audioContext.destination );
    updateAnalysers();
}

function initAudio() {
    if (!navigator.getUserMedia)
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
        navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
        navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    navigator.getUserMedia({audio: true}, gotStream, function(e) {
        alert('Error getting audio');
        console.log(e);
    });
}

window.addEventListener('load', initAudio );