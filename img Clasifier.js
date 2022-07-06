const webcamElement = document.getElementsByClassName("webcam")[0];
const buttons = document.getElementsByTagName("button");
const predictButton = document.getElementsByClassName("predict")[0];
const classes = ["This Is Pic 1", " This Is Pic 2", "This Is Pic 3", "This Is Pic 4"];
const predictionParagraph = document.getElementsByClassName("prediction")[0];
async function app() {
  const classifier = knnClassifier.create();
  const net = await mobilenet.load();
  const webcam = await tf.data.webcam(webcamElement);
  const addExample = async classId => {
    const img = await webcam.capture();
    const activation = net.infer(img, "conv_preds");
    classifier.addExample(activation, classId);
    img.dispose();
  };
  for (var i = 0; i < buttons.length; i++) {
    if (buttons[i] !== predictButton) {
      let index = i;
      buttons[i].onclick = () => addExample(index);
    }
  }
  predictButton.onclick = () => runPredictions();
  async function runPredictions() {
    while (true) {
      if (classifier.getNumClasses() > 0) {
        const img = await webcam.capture();
        const activation = net.infer(img, "conv_preds");
        const result = await classifier.predictClass(activation);
        predictionParagraph.innerText = `
            prediction: ${classes[result.label]},
            probability: ${result.confidences[result.label]}`;
        img.dispose();
      }
      await tf.nextFrame();
    }
    
  }
}
app();