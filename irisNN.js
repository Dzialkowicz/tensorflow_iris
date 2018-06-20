import * as tf from '@tensorflow/tfjs';
import iris from './iris.json';
import irisTest from './iris-test.json';
import { sequential } from '@tensorflow/tfjs';

//set up and convert data
const trainingData = tf.tensor2d(iris.map(i => [
    i.sepalLength, i.sepalWidth, i.petalLength, i.petalWidth,    
]))

const outputData = tf.tensor2d(iris.map(i => [
    i.species === "setosa" ? 1 : 0,
    i.species === "virginica" ? 1 : 0,
    i.species === "versicolor" ? 1 : 0,  
]))

const testData = tf.tensor2d(iris.map(i => [
    i.sepalLength, i.sepalWidth, i.petalLength, i.petalWidth,    
]))

//build NN
const model = tf.sequential()

model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5,
}))

model.add(tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3,
}))

model.add(tf.layers.dense({
    activation: "sigmoid",
    units: 3,
}))

model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(.06),

})

//train/fit network
const startTimer = Date.now()
model.fit(trainingData, outputData, {epochs: 100})
    .then((history) => {
        //console.log("NN trained in", (Date.now()- startTimer)/1000, " seconds!")
        //console.log(history)
        model.predict(testData).print()
    })
//test network