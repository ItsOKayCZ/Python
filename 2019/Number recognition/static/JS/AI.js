var model;

async function AI(){
  console.log("Loading model");
  model = await tf.loadLayersModel(location.origin + "/model/model.json");
  console.log("Done loading model");
}

function predict(data){
  console.log("Predicting");
  var arr = [];
  for(var i = 0; i < data.length; i++){
    var val = data[i] / 255;
    arr.push(val);
  }
  arr = tf.tensor(arr);
  arr = tf.reshape(arr, [1, 28, 28, 1]);

  var prediction = Array.from(model.predict(arr).dataSync());
  prediction = argMax(prediction);

  document.getElementById("output").children[0].innerHTML = prediction;
}

// Source: https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}