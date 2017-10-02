package com.andy.feedforward.classfication;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * @Description:
 * @Author: Andy Hoo
 * @Date: 2017/10/2 16:22
 */
public class MLPClassifierLinear {
    private static final Logger log = LoggerFactory.getLogger("MLPClassifierLinear");

    public static void main(String[] args) throws IOException, InterruptedException {

        log.info("classification begins...");

        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        final String trainFileName = new ClassPathResource("/classification/linear_data_train.csv").getFile().getAbsolutePath();
        final String testFileName = new ClassPathResource("/classification/linear_data_eval.csv").getFile().getAbsolutePath();

        // load the train data
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(trainFileName)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

        // load the test data
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(testFileName)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        // conf settings
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        //training set
        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainIter);
        }

        // test set
        Evaluation evaluation = new Evaluation(numOutputs);
        while (testIter.hasNext()){
            DataSet next =testIter.next();
            INDArray features = next.getFeatureMatrix();
            INDArray labels = next.getLabels();

            INDArray predicted = model.output(features, false);
            evaluation.eval(labels,predicted);
        }

        System.out.println(evaluation.stats());

        log.info("classification end!!!");
    }
}
