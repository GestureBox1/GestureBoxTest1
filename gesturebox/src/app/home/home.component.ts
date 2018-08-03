
import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { WaterfallService } from './waterfall.service';
import { ISettings } from './utils/settings';
import { IData } from './utils/data';
declare var d3v;
declare var tf;

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {
  public title = 'gesturebox';
  data: IData;
  settings: ISettings;
  chart;
  enableDrag: boolean;
  disableCursor: boolean;
  hideDates: boolean;
  

  CONTROLS = ['up', 'down', 'left', 'right'];
  mouseDown;
  NUM_CLASSES = 4;
  mobilenet;
  model;
  webcamElement;
  controllerDataset = new ControllerDataset(this.NUM_CLASSES);
  totals = [0, 0, 0, 0];
  isPredicting = false;
  constructor(private waterfallService: WaterfallService) {

  }
  value: number;
  getData() {
    // this.waterfallService.getData()
    // .subscribe(data => {
    //   console.log(data);
    // })
    this.settings = this.waterfallService.getSetiings();
    this.data = this.waterfallService.getData();
    this.drawChart();
  }
  ngOnInit() {
    this.webcamElement = document.getElementById('webcam');
    this.getData();
    this.webCamInIt();
  }
  drawChart() {

    this.chart = new d3v.Waterfall(this.data, this.settings);
  }
  webCamInIt() {
    this.init();
    this.addEvents();
    document.getElementById('train').addEventListener('click', async () => {
      await tf.nextFrame();
      await tf.nextFrame();
      this.train();
    });
    document.getElementById('predict').addEventListener('click', () => {
      this.isPredicting = true;
      this.predict();
    });
  }
  async  predict() {
    // ui.isPredicting();
    while (this.isPredicting) {
      const predictedClass = tf.tidy(() => {
        // Capture the frame from the webcam.
        const img = this.capture();
  
        // Make a prediction through mobilenet, getting the internal activation of
        // the mobilenet model.
        const activation = this.mobilenet.predict(img);
  
        // Make a prediction through our newly-trained model using the activation
        // from mobilenet as input.
        const predictions = this.model.predict(activation);
  
        // Returns the index with the maximum probability. This number corresponds
        // to the class the model thinks is the most probable given the input.
        return predictions.as1D().argMax();
      });
  
      const classId = (await predictedClass.data())[0];
      predictedClass.dispose();
      this.addBinding(classId);
      //ui.predictClass(classId);
      await tf.nextFrame();
    }
    // ui.donePredicting();
  }
  addBinding(classId){
    switch(classId) {
      case 0:
      this.chart.rotateTo('up');
          break;
      case 1:
      this.chart.rotateTo('down');
          break;
      case 2:
      this.chart.rotateTo('left');
          break;
      case 3:
      this.chart.rotateTo('right');
          break;
      default:
          console.log("Do Nothing");
  }
  }
  async  train() {
    if (this.controllerDataset.xs == null) {
      throw new Error('Add some examples before training!');
    }
  
    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    this.model = tf.sequential({
      layers: [
        // Flattens the input to a vector so we can use it in a dense layer. While
        // technically a layer, this only performs a reshape (and has no training
        // parameters).
        tf.layers.flatten({inputShape: [7, 7, 256]}),
        // Layer 1
        tf.layers.dense({
          units: 100,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true
        }),
        // Layer 2. The number of units of the last layer should correspond
        // to the number of classes we want to predict.
        tf.layers.dense({
          units: this.NUM_CLASSES,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax'
        })
      ]
    });
  
    // Creates the optimizers which drives training of the model.
    const optimizer = tf.train.adam(0.00001);
    // We use categoricalCrossentropy which is the loss function we use for
    // categorical classification which measures the error between our predicted
    // probability distribution over classes (probability that an input is of each
    // class), versus the label (100% probability in the true class)>
    this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  
    // We parameterize batch size as a fraction of the entire dataset because the
    // number of examples that are collected depends on how many examples the user
    // collects. This allows us to have a flexible batch size.
    const batchSize =
        Math.floor(this.controllerDataset.xs.shape[0] * 0.05);
    if (!(batchSize > 0)) {
      throw new Error(
          `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
    }
  
    // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
    this.model.fit(this.controllerDataset.xs, this.controllerDataset.ys, {
      batchSize,
      epochs: 50,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
            console.log("Trained");
          //ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
          await tf.nextFrame();
         
        }
      }
    });
    
  }

  async saveModel () {
    const saveResult = await this.model.save('indexeddb://my-model-1');
  }
  async ReadLo(){
     this.model = await tf.loadModel('indexeddb://my-model-1');
  }

  addEvents() {
    this.mouseDown = false;

    const upButton = document.getElementById('up');
    const downButton = document.getElementById('down');
    const leftButton = document.getElementById('left');
    const rightButton = document.getElementById('right');
    upButton.addEventListener('mousedown', () => this.handler(0));
    upButton.addEventListener('mouseup', () => this.mouseDown = false);

    downButton.addEventListener('mousedown', () => this.handler(1));
    downButton.addEventListener('mouseup', () => this.mouseDown = false);

    leftButton.addEventListener('mousedown', () => this.handler(2));
    leftButton.addEventListener('mouseup', () => this.mouseDown = false);

    rightButton.addEventListener('mousedown', () => this.handler(3));
    rightButton.addEventListener('mouseup', () => this.mouseDown = false);

  }
  async  handler(label) {
    this.mouseDown = true;
    const className = this.CONTROLS[label];
    const button = document.getElementById(className);
    const total: any = document.getElementById(className + '-total');
    while (this.mouseDown) {
      this.addExampleHandler(label);
      document.body.setAttribute('data-active', this.CONTROLS[label]);
      total.innerText = this.totals[label]++;
      await tf.nextFrame();
    }
    document.body.removeAttribute('data-active');
  }
  addExampleHandler(label) {

    tf.tidy(() => {
      const img = this.capture();
      this.controllerDataset.addExample(this.mobilenet.predict(img), label);

      // Draw the preview thumbnail.
      // ui.drawThumb(img, label);
    });
  }
  capture() {
    return tf.tidy(() => {
      // Reads the image as a Tensor from the webcam <video> element.
      const webcamImage = tf.fromPixels(this.webcamElement);

      // Crop the image so we're using the center square of the rectangular
      // webcam.
      const croppedImage = this.cropImage(webcamImage);

      // Expand the outer most dimension so we have a batch size of 1.
      const batchedImage = croppedImage.expandDims(0);

      // Normalize the image between -1 and 1. The image comes in between 0-255,
      // so we divide by 127 and subtract 1.
      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  }
  cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }
  async  init() {
    try {
      await this.setup(document.getElementById('webcam'));
    } catch (e) {
      document.getElementById('no-webcam').style.display = 'block';
    }
    this.mobilenet = await this.loadMobilenet();

    // Warm up the model. This uploads weights to the GPU and compiles the WebGL
    // programs so the first time we collect data from the webcam it will be
    // quick.
    // tf.tidy(() => mobilenet.predict(webcam.capture()));

    // ui.init();
  }
  async  loadMobilenet() {
    const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  }

  async  setup(webcamElement) {
    return new Promise((resolve, reject) => {
      const navigatorAny: any = navigator;
      navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          { video: true },
          stream => {
            webcamElement.srcObject = stream;
            webcamElement.addEventListener('loadeddata', async () => {
              this.adjustVideoSize(
                webcamElement.videoWidth,
                webcamElement.videoHeight, webcamElement);
              resolve();
            }, false);
          },
          error => {
            reject();
          });
      } else {
        reject();
      }
    });
  }
   adjustVideoSize(width, height, webcamElement) {
    const aspectRatio = width / height;
    if (width >= height) {
        webcamElement.width = aspectRatio * webcamElement.height;
    } else if (width < height) {
        webcamElement.height = webcamElement.width / aspectRatio;
    }
}

}
class ControllerDataset {
  numClasses: number;
  xs: any;
  ys: any;
  constructor(numClasses) {
    this.numClasses = numClasses;
  }

  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} example A tensor representing the example. It can be an image,
   *     an activation, or any other type of Tensor.
   * @param {number} label The label of the example. Should be a number.
   */
  addExample(example, label) {
    // One-hot encode the label.
    const y = tf.tidy(
      () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    if (this.xs == null) {
      // For the first example that gets added, keep example and y so that the
      // ControllerDataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}
