package ir.jimbo;

import com.fasterxml.jackson.databind.ObjectMapper;
import ir.jimbo.model.Data;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Learn {

    public void start() throws IOException, InterruptedException {

        SparkSession spark = SparkSession.builder()
                .appName("learning")
                .master("local")
                .getOrCreate();

        JavaSparkContext javaSparkContext = new JavaSparkContext(spark.sparkContext());
        List<Data> data = readData();

        System.out.println("data readed");

        JavaRDD<Data> dataRDD = javaSparkContext.parallelize(data);

        data = null;
        System.out.println("creating java rdd");

        Dataset<Row> dataSet = spark.createDataFrame(dataRDD, Data.class);

        System.out.println("after creating dataset");

        Tokenizer tokenizer = new Tokenizer().setInputCol("content").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(dataSet);

        System.out.println("after tokenize and before hashing");

        HashingTF hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setNumFeatures(100);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("feature");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
        Dataset<Row> features = rescaledData.select("tag", "feature");

        NaiveBayes naiveBayes = new NaiveBayes()
                .setModelType("multinomial")
                .setLabelCol("tag")
                .setFeaturesCol("feature");

        Dataset<Row>[] tmp = features.randomSplit(new double[]{0.85, 0.15});
        Dataset<Row> training = tmp[0]; // training set
        Dataset<Row> test = tmp[1]; // test set

        NaiveBayesModel model = naiveBayes.train(training);
        model.set("modelType", "multinomial");

        JavaPairRDD<Double, Double> predictionAndLabel =
                test.toJavaRDD().mapToPair((Row p) ->
                        new Tuple2<>(model.predict(p.getAs(1)), p.getDouble(0)));
        test.show(false);
        System.out.println(predictionAndLabel.collect());

        double accuracy =
                predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();
        System.out.println("accuracy : " + accuracy + " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

        try {
            model.save("naiveBayesModel");
        } catch (IOException e) {
            System.out.println("be fana raftim ke ..." + e);
        }

        javaSparkContext.close();
        spark.stop();
    }

    private static List<Data> readData() throws IOException, InterruptedException {
        List<String> stopWords = new ArrayList<>(Arrays.asList("i", "me", "my", "myself", "we", "our", "ours"
                , "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she"
                , "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"
                , "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were"
                , "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the"
                , "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about"
                , "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from"
                , "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here"
                , "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other"
                , "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t"
                , "can", "will", "just", "don", "should", "now"));
        List<Data> list = new ArrayList<>();
        ObjectMapper objectMapper = new ObjectMapper();
        Data data;
        System.out.println("start reading urls2 ...");
        String datas = new String(Files.readAllBytes(Paths.get("urls2")));
        datas = datas.replace("}{", "}\n!@#\n{");
        String[] split = datas.split("!@#");
        for (String d : split) {
            try {
                data = objectMapper.readValue(d, Data.class);
                deleteStopWords(data, stopWords);
                list.add(data);
            } catch (Exception e) {
                System.out.println("parse error");
            }
        }
        System.out.println("number of urls to now " + list.size());
        System.out.println("reading urls2 done.");
        //////////////////////////////////////////////////////////////////
        System.out.println("start reading urls3 ...");
        datas = new String(Files.readAllBytes(Paths.get("urls3")));
        datas = datas.replace("}{", "}\n!@#\n{");
        split = datas.split("!@#");
        for (String d : split) {
            try {
                data = objectMapper.readValue(d, Data.class);
                deleteStopWords(data, stopWords);
                list.add(data);
            } catch (Exception e) {
                System.out.println("parse error");
            }
        }
        System.out.println("reading urls3 done.");
        System.out.println("number of urls to now " + list.size());
        //////////////////////////////////////////////////////////////
        System.out.println("start reading urls4 ...");
        datas = new String(Files.readAllBytes(Paths.get("urls4")));
        datas = datas.replace("}{", "}\n!@#\n{");
        split = datas.split("!@#");
        for (String d : split) {
            try {
                data = objectMapper.readValue(d, Data.class);
                deleteStopWords(data, stopWords);
                list.add(data);
            } catch (Exception e) {
                System.out.println("parse error");
            }
        }
        System.out.println("reading urls4 done.");
        System.out.println("number of urls to now " + list.size());
        ///////////////////////////////////////////////////////////////
        Thread.sleep(2000);
        System.out.println("start reading urls1 ...");
        datas = new String(Files.readAllBytes(Paths.get("url1")));
        datas = datas.replace("}{", "}\n!@#\n{");
        split = datas.split("!@#");
        for (String d : split) {
            try {
                data = objectMapper.readValue(d, Data.class);
                deleteStopWords(data, stopWords);
                list.add(data);
            } catch (Exception e) {
                System.out.println("parse error");
            }
        }
        System.out.println("reading urls1 done.");
        System.out.println("number of urls to now " + list.size());
        stopWords = null;
        return list;
    }

    private static void deleteStopWords(Data data, List<String> stopWords) {
        String content = data.getContent();
        for (String stopWord : stopWords) {
            content = content.replace(stopWord, "");
        }
        data.setContent(content);
    }
}