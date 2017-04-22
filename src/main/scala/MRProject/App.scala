package MRProject

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, RandomForestModel}

/**
  * @author ${user.name}
  */

object App {
  val mappings = scala.collection.mutable.ListBuffer[Int]()
  val colList = List(
    "Agelaius_phoeniceus",
    "POP00_SQMI",
    "HOUSING_DENSITY",
    "HOUSING_PERCENT_VACANT",
    "ELEV_NED",
    "BCR",
    "OMERNIK_L3_ECOREGION",
    "CAUS_TEMP_AVG",
    "CAUS_PREC",
    "CAUS_SNOW",
    "NLCD2001_FS_C11_7500_PLAND",
    "NLCD2001_FS_C12_7500_PLAND",
    "NLCD2001_FS_C21_7500_PLAND",
    "NLCD2001_FS_C22_7500_PLAND",
    "NLCD2001_FS_C23_7500_PLAND",
    "NLCD2001_FS_C24_7500_PLAND",
    "NLCD2001_FS_C31_7500_PLAND",
    "NLCD2001_FS_C41_7500_PLAND",
    "NLCD2001_FS_C42_7500_PLAND",
    "NLCD2001_FS_C43_7500_PLAND",
    "NLCD2001_FS_C52_7500_PLAND",
    "NLCD2001_FS_C71_7500_PLAND",
    "NLCD2001_FS_C81_7500_PLAND",
    "NLCD2001_FS_C82_7500_PLAND",
    "NLCD2001_FS_C90_7500_PLAND",
    "NLCD2001_FS_C95_7500_PLAND",
    "NLCD2006_FS_C11_7500_PLAND",
    "NLCD2006_FS_C12_7500_PLAND",
    "NLCD2006_FS_C21_7500_PLAND",
    "NLCD2006_FS_C22_7500_PLAND",
    "NLCD2006_FS_C23_7500_PLAND",
    "NLCD2006_FS_C24_7500_PLAND",
    "NLCD2006_FS_C31_7500_PLAND",
    "NLCD2006_FS_C41_7500_PLAND",
    "NLCD2006_FS_C42_7500_PLAND",
    "NLCD2006_FS_C43_7500_PLAND",
    "NLCD2006_FS_C52_7500_PLAND",
    "NLCD2006_FS_C71_7500_PLAND",
    "NLCD2006_FS_C81_7500_PLAND",
    "NLCD2006_FS_C82_7500_PLAND",
    "NLCD2006_FS_C90_7500_PLAND",
    "NLCD2006_FS_C95_7500_PLAND",
    "NLCD2011_FS_C11_7500_PLAND",
    "NLCD2011_FS_C12_7500_PLAND",
    "NLCD2011_FS_C21_7500_PLAND",
    "NLCD2011_FS_C22_7500_PLAND",
    "NLCD2011_FS_C23_7500_PLAND",
    "NLCD2011_FS_C24_7500_PLAND",
    "NLCD2011_FS_C31_7500_PLAND",
    "NLCD2011_FS_C41_7500_PLAND",
    "NLCD2011_FS_C42_7500_PLAND",
    "NLCD2011_FS_C43_7500_PLAND",
    "NLCD2011_FS_C52_7500_PLAND",
    "NLCD2011_FS_C71_7500_PLAND",
    "NLCD2011_FS_C81_7500_PLAND",
    "NLCD2011_FS_C82_7500_PLAND",
    "NLCD2011_FS_C90_7500_PLAND",
    "NLCD2011_FS_C95_7500_PLAND",
    "CAUS_TEMP_MIN",
    "CAUS_TEMP_MAX",
    "DIST_FROM_FLOWING_FRESH",
    "DIST_IN_FLOWING_FRESH",
    "DIST_FROM_STANDING_FRESH",
    "DIST_IN_STANDING_FRESH",
    "DIST_FROM_WET_VEG_FRESH",
    "DIST_IN_WET_VEG_FRESH",
    "DIST_FROM_FLOWING_BRACKISH",
    "DIST_IN_FLOWING_BRACKISH",
    "DIST_FROM_STANDING_BRACKISH",
    "DIST_IN_STANDING_BRACKISH",
    "DIST_FROM_WET_VEG_BRACKISH",
    "DIST_IN_WET_VEG_BRACKISH"
  )

  def main(args : Array[String]) {
    //Handle invalid arguments
    if (args.length < 1) {
      System.err.println("Specify Input Directory")
      System.exit(1)
    }
    //Spark Configuration
    //This needs app name. The Master and other spark-defaults are set in the makefile
    println(System.currentTimeMillis());
    val conf = new SparkConf().setAppName("DJNS")

    //val conf = new SparkConf().setAppName("DJNS").setMaster("local")
    //Spark Context
    val sc = new SparkContext(conf)
    var header: String = ""
    val input = sc.textFile(args(0)).mapPartitionsWithIndex((idx, iter) => if (idx == 0) {
      iter.next().split(",").zipWithIndex.foreach(x => {
        if(colList.contains(x._1)){
          mappings += x._2
        }
      })
      iter.drop(1)
    }
    else iter)

    val inputRDD = input.map(line => line.split(",").zipWithIndex).map(x => rCheck(x))

    println("InputRDD"+inputRDD.count())


    val parsed = inputRDD.map { case (k, vs) =>
      LabeledPoint(k.toDouble, Vectors.dense(vs.toArray))
    }


    // Split data into training (60%) and test (40%).
    val splits = parsed.randomSplit(Array(0.8, 0.2))
    val (trainingData, testData) = (splits(0), splits(1))

    //val Array(training, test) = filtered.randomSplit(Array(0.6, 0.4))
    // Split data into training (60%) and test (40%).
    /*val splits = parsed.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    //L2 Regularization
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    //L1 Reg
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer
      .setNumIterations(100)
      .setRegParam(0.1)
      .setUpdater(new L1Updater)
    val model = svmAlg.run(training)

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
    println(System.currentTimeMillis());
    // Save and load model
    model.save(sc, "target/tmp/scalaSVMWithSGDModel")
    val sameModel = SVMModel.load(sc, "target/tmp/scalaSVMWithSGDModel")*/



    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    /*val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 8
    val maxBins =64

    val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression forest model:\n" + model.toDebugString)

    val accuracy = 1.0 * labelsAndPredictions.filter(x => (x._1 == 1 && x._2 > 0.5) || (x._1 == 0 && x._2 < 0.5)).count() / testData.count()
    println("Accuracy ="+ accuracy);
    // Save and load model
    model.save(sc, "target/tmp/myRandomForestRegressionModel")
    val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")*/

    //GBT Model

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 10 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 6
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = scala.Predef.Map[Int, Int]()
    //Validation Tolerance Factor
    boostingStrategy.validationTol = -0.0001
    val model = new GradientBoostedTrees(boostingStrategy).runWithValidation(trainingData, testData)
    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

   /* val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)*/
    val accuracy = 1.0 * labelAndPreds.filter(x => (x._1 == 1 && x._2 > 0.5) || (x._1 == 0 && x._2 < 0.5)).count() / testData.count()
    println("Accuracy : "+accuracy);
    println("Learned classification GBT model:\n" + model.toDebugString)

    // Save and load model
    model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
    val sameModel = GradientBoostedTreesModel.load(sc,
      "target/tmp/myGradientBoostingClassificationModel")
  }

  def rCheck(x: Array[(String,Int)]): (Double,List[Double]) = {
    val y = x.filter(x => mappings.contains(x._2)).map(_._1)
    var arr = scala.collection.mutable.ListBuffer[Double] ()
    var label:Double = 0.0;
    if(y(0).equalsIgnoreCase("X") || Integer.parseInt(y(0)) > 0){label = 1.0}
    for(elem <- y.drop(1)){
      if (elem.equals("?") || elem.toDouble < 0.0) {
        arr += 0.0
      } else {
        arr += elem.toDouble
      }
    }
    return (label,arr.toList)
  }
}