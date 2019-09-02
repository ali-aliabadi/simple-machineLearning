package ir.jimbo;

import org.apache.spark.ml.classification.NaiveBayesModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class LoadLearned {

    public void start() throws IOException {
        NaiveBayesModel model = NaiveBayesModel.load("naiveBayesModel");

        String test = new String(Files.readAllBytes(Paths.get("test")));


    }

}
