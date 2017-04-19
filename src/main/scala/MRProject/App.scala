package MRProject

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
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
    val input = sc.textFile(args(0)).map(line => line.split(",")).map(x => (if(x(26).toInt > 0 || x(26).equals("X")){1}else{0},List(
      x(955).toDouble,
      x(956).toDouble,
      x(957).toDouble,
      x(959).toDouble,
      x(962).toDouble,
      x(963).toDouble,
      x(966).toDouble,
      x(967).toDouble)))
    val parsed = input.map { case (k, vs) =>
      LabeledPoint(k.toDouble, Vectors.dense(vs.toArray))
    }

    /*filtered.repartition(1)
      .saveAsTextFile(args(1));*/

    //val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    //val Array(training, test) = filtered.randomSplit(Array(0.6, 0.4))


    // Split data into training (60%) and test (40%).
    val splits = parsed.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    // Save and load model
    model.save(sc, "target/tmp/scalaSVMWithSGDModel")
    val sameModel = SVMModel.load(sc, "target/tmp/scalaSVMWithSGDModel")
  }
}