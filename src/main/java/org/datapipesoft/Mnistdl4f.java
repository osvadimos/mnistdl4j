package org.datapipesoft;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class Mnistdl4f {

    private static Logger log = LoggerFactory.getLogger(Mnistdl4f.class);

    public static void main(String[] args) throws Exception {

        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 7;// 15; // number of epochs to perform

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();
        log.debug(conf.toJson());
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(100));
        File f = new File("/home/vadim/tmp/networkfile");
        MultiLayerNetwork loadedNetwork = MultiLayerNetwork.load(f, true);
        log.debug("Network initialized");
        log.debug(network.toString());
        for (int i = 0; i < numEpochs; i++) {
            log.debug("Training epoch:" + i);
            network.fit(mnistTrain);
        }
        log.info("Evaluate model....");

        network.save(f);
        Evaluation evaluation = new Evaluation();
        Evaluation loadedEvaluation = new Evaluation();
        while (mnistTest.hasNext()){
            DataSet dataSet = mnistTest.next();
            INDArray output = network.output(dataSet.getFeatures());
            INDArray loadedOutput = loadedNetwork.output(dataSet.getFeatures());
            loadedEvaluation.eval(dataSet.getLabels(), loadedOutput);
            evaluation.eval(dataSet.getLabels(), output);
        }
        log.info("****************Trained stats********************");
        log.info(evaluation.stats());
        log.info("****************Loaded stats********************");
        log.info(loadedEvaluation.stats());
        log.info("****************Example finished********************");
    }
}
