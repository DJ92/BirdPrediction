package MRProject

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils

/**
  * @author ${user.name}
  */
object App {

  def foo(x : Array[String]) = x.foldLeft("")((a,b) => a + b)

  def main(args : Array[String]) {
    //Handle invalid arguments
    if (args.length < 1) {
      System.err.println("Specify Input Directory")
      System.exit(1)
    }
    //Spark Configuration
    //This needs app name. The Master and other spark-defaults are set in the makefile
    val conf = new SparkConf().setAppName("PageRank").setMaster("local")

    //Spark Context
    val sc = new SparkContext(conf)
    val input = sc.textFile(args(0)).map(line => line.split(",")).map(x => (x(0),x(1),x(26),x(955)
      ,x(956) ,x(957) ,x(958) , x(959), x(960),x(962),x(963),x(966),x(967)))
    val filtered = input.filter(x => (!x._3.equals("0") || x._3.equalsIgnoreCase("X")))
    filtered.repartition(1)
      .saveAsTextFile(args(1));

    //val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    val Array(training, test) = filtered.randomSplit(Array(0.6, 0.4))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    // Save and load model
    model.save(sc, "target/tmp/myNaiveBayesModel")
    val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
  }
}