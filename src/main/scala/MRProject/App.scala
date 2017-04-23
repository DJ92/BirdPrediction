package MRProject

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{RandomForest}
import org.apache.spark.mllib.tree.model.{RandomForestModel}
import org.apache.spark.rdd.RDD

/**
  * @author DJ
  * @author NS
  * */

object App {

  def main(args : Array[String]) {

    //Handle invalid arguments
    if (args.length < 1) {
      System.err.println("Specify Input Directory")
      System.exit(1)
    }

    //Spark Configuration
    //This needs app name. The Master and other spark-defaults are set in the makefile
    val conf = new SparkConf().setAppName("DJNS")

    //Spark Context
    val sc = new SparkContext(conf)

    // InputRDD
    // This does the following Steps:
    // - Drops the header
    // - Parses and Casts selected column values
    // Returns tuple of the form (label,List(features))
    val input = sc.textFile(args(0)).mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .map(line => line.split(",")).map(x =>
      (if(x(26).equals("X") || Integer.parseInt(x(26)) > 0){1}else{0},List(       //  Agelaius_phoeniceus
        if(x(955).equals("?") || x(955).toDouble < 0.0) {0} else{x(955).toDouble},//  POP00_SQMI
        if(x(956).equals("?") || x(956).toDouble < 0.0) {0} else{x(956).toDouble},//  HOUSING_DENSITY
        if(x(957).equals("?") || x(957).toDouble < 0.0) {0} else{x(957).toDouble},//  HOUSING_PERCENT_VACANT
        if(x(959).equals("?") || x(959).toDouble < 0.0) {0} else{x(959).toDouble},//  ELEV_NED
        if(x(960).equals("?") || x(960).toDouble < 0.0) {0} else{x(960).toDouble},//  BCR
        if(x(962).equals("?") || x(962).toDouble < 0.0) {0} else{x(962).toDouble},//  OMERNIK_L3_ECOREGION
        if(x(963).equals("?") || x(963).toDouble < 0.0) {0} else{x(963).toDouble},//  CAUS_TEMP_AVG
        if(x(964).equals("?") || x(964).toDouble < 0.0) {0} else{x(964).toDouble},//  CAUS_TEMP_MIN
        if(x(965).equals("?") || x(965).toDouble < 0.0) {0} else{x(965).toDouble},//  CAUS_TEMP_MAX
        if(x(966).equals("?") || x(966).toDouble < 0.0) {0} else{x(966).toDouble},//  CAUS_PREC
        if(x(967).equals("?") || x(967).toDouble < 0.0) {0} else{x(967).toDouble},//  CAUS_SNOW
        if(x(968).equals("?") || x(968).toDouble < 0.0) {0} else{x(968).toDouble},//  NLCD2001_FS_C11_7500_PLAND
        if(x(969).equals("?") || x(969).toDouble < 0.0) {0} else{x(969).toDouble},//  NLCD2001_FS_C12_7500_PLAND
        if(x(970).equals("?") || x(970).toDouble < 0.0) {0} else{x(970).toDouble},//  NLCD2001_FS_C21_7500_PLAND
        if(x(971).equals("?") || x(971).toDouble < 0.0) {0} else{x(971).toDouble},//  NLCD2001_FS_C22_7500_PLAND
        if(x(972).equals("?") || x(972).toDouble < 0.0) {0} else{x(972).toDouble},//  NLCD2001_FS_C23_7500_PLAND
        if(x(973).equals("?") || x(973).toDouble < 0.0) {0} else{x(973).toDouble},//  NLCD2001_FS_C24_7500_PLAND
        if(x(974).equals("?") || x(974).toDouble < 0.0) {0} else{x(974).toDouble},//  NLCD2001_FS_C31_7500_PLAND
        if(x(975).equals("?") || x(975).toDouble < 0.0) {0} else{x(975).toDouble},//  NLCD2001_FS_C41_7500_PLAND
        if(x(976).equals("?") || x(976).toDouble < 0.0) {0} else{x(976).toDouble},//  NLCD2001_FS_C42_7500_PLAND
        if(x(977).equals("?") || x(977).toDouble < 0.0) {0} else{x(977).toDouble},//  NLCD2001_FS_C43_7500_PLAND
        if(x(978).equals("?") || x(978).toDouble < 0.0) {0} else{x(978).toDouble},//  NLCD2001_FS_C52_7500_PLAND
        if(x(979).equals("?") || x(979).toDouble < 0.0) {0} else{x(979).toDouble},//  NLCD2001_FS_C71_7500_PLAND
        if(x(980).equals("?") || x(980).toDouble < 0.0) {0} else{x(980).toDouble},//  NLCD2001_FS_C81_7500_PLAND
        if(x(981).equals("?") || x(981).toDouble < 0.0) {0} else{x(981).toDouble},//  NLCD2001_FS_C82_7500_PLAND
        if(x(982).equals("?") || x(982).toDouble < 0.0) {0} else{x(982).toDouble},//  NLCD2001_FS_C90_7500_PLAND
        if(x(983).equals("?") || x(983).toDouble < 0.0) {0} else{x(983).toDouble},//  NLCD2001_FS_C95_7500_PLAND
        if(x(984).equals("?") || x(984).toDouble < 0.0) {0} else{x(984).toDouble},//  NLCD2006_FS_C11_7500_PLAND
        if(x(985).equals("?") || x(985).toDouble < 0.0) {0} else{x(985).toDouble},//  NLCD2006_FS_C12_7500_PLAND
        if(x(986).equals("?") || x(986).toDouble < 0.0) {0} else{x(986).toDouble},//  NLCD2006_FS_C21_7500_PLAND
        if(x(987).equals("?") || x(987).toDouble < 0.0) {0} else{x(987).toDouble},//  NLCD2006_FS_C22_7500_PLAND
        if(x(988).equals("?") || x(988).toDouble < 0.0) {0} else{x(988).toDouble},//  NLCD2006_FS_C23_7500_PLAND
        if(x(989).equals("?") || x(989).toDouble < 0.0) {0} else{x(989).toDouble},//  NLCD2006_FS_C24_7500_PLAND
        if(x(990).equals("?") || x(990).toDouble < 0.0) {0} else{x(990).toDouble},//  NLCD2006_FS_C31_7500_PLAND
        if(x(991).equals("?") || x(991).toDouble < 0.0) {0} else{x(991).toDouble},//  NLCD2006_FS_C41_7500_PLAND
        if(x(992).equals("?") || x(992).toDouble < 0.0) {0} else{x(992).toDouble},//  NLCD2006_FS_C42_7500_PLAND
        if(x(993).equals("?") || x(993).toDouble < 0.0) {0} else{x(993).toDouble},//  NLCD2006_FS_C43_7500_PLAND
        if(x(994).equals("?") || x(994).toDouble < 0.0) {0} else{x(994).toDouble},//  NLCD2006_FS_C52_7500_PLAND
        if(x(995).equals("?") || x(995).toDouble < 0.0) {0} else{x(995).toDouble},//  NLCD2006_FS_C71_7500_PLAND
        if(x(996).equals("?") || x(996).toDouble < 0.0) {0} else{x(996).toDouble},//  NLCD2006_FS_C81_7500_PLAND
        if(x(997).equals("?") || x(997).toDouble < 0.0) {0} else{x(997).toDouble},//  NLCD2006_FS_C82_7500_PLAND
        if(x(998).equals("?") || x(998).toDouble < 0.0) {0} else{x(998).toDouble},//  NLCD2006_FS_C90_7500_PLAND
        if(x(999).equals("?") || x(999).toDouble < 0.0) {0} else{x(999).toDouble},//  NLCD2006_FS_C95_7500_PLAND
        if(x(1000).equals("?") || x(1000).toDouble < 0.0) {0} else{x(1000).toDouble},// NLCD2011_FS_C11_7500_PLAND
        if(x(1001).equals("?") || x(1001).toDouble < 0.0) {0} else{x(1001).toDouble},// NLCD2011_FS_C12_7500_PLAND
        if(x(1002).equals("?") || x(1002).toDouble < 0.0) {0} else{x(1002).toDouble},// NLCD2011_FS_C21_7500_PLAND
        if(x(1003).equals("?") || x(1003).toDouble < 0.0) {0} else{x(1003).toDouble},// NLCD2011_FS_C22_7500_PLAND
        if(x(1004).equals("?") || x(1004).toDouble < 0.0) {0} else{x(1004).toDouble},// NLCD2011_FS_C23_7500_PLAND
        if(x(1005).equals("?") || x(1005).toDouble < 0.0) {0} else{x(1005).toDouble},// NLCD2011_FS_C24_7500_PLAND
        if(x(1006).equals("?") || x(1006).toDouble < 0.0) {0} else{x(1006).toDouble},// NLCD2011_FS_C31_7500_PLAND
        if(x(1007).equals("?") || x(1007).toDouble < 0.0) {0} else{x(1007).toDouble},// NLCD2011_FS_C41_7500_PLAND
        if(x(1008).equals("?") || x(1008).toDouble < 0.0) {0} else{x(1008).toDouble},// NLCD2011_FS_C42_7500_PLAND
        if(x(1009).equals("?") || x(1009).toDouble < 0.0) {0} else{x(1009).toDouble},// NLCD2011_FS_C43_7500_PLAND
        if(x(1010).equals("?") || x(1010).toDouble < 0.0) {0} else{x(1010).toDouble},// NLCD2011_FS_C52_7500_PLAND
        if(x(1011).equals("?") || x(1011).toDouble < 0.0) {0} else{x(1011).toDouble},// NLCD2011_FS_C71_7500_PLAND
        if(x(1012).equals("?") || x(1012).toDouble < 0.0) {0} else{x(1012).toDouble},// NLCD2011_FS_C81_7500_PLAND
        if(x(1013).equals("?") || x(1013).toDouble < 0.0) {0} else{x(1013).toDouble},// NLCD2011_FS_C82_7500_PLAND
        if(x(1014).equals("?") || x(1014).toDouble < 0.0) {0} else{x(1014).toDouble},// NLCD2011_FS_C90_7500_PLAND
        if(x(1015).equals("?") || x(1015).toDouble < 0.0) {0} else{x(1015).toDouble},// NLCD2011_FS_C95_7500_PLAND
        if(x(1090).equals("?") || x(1090).toDouble < 0.0) {0} else{x(1090).toDouble},// DIST_FROM_FLOWING_FRESH
        if(x(1091).equals("?") || x(1091).toDouble < 0.0) {0} else{x(1091).toDouble},// DIST_IN_FLOWING_FRESH
        if(x(1092).equals("?") || x(1092).toDouble < 0.0) {0} else{x(1092).toDouble},// DIST_FROM_STANDING_FRESH
        if(x(1093).equals("?") || x(1093).toDouble < 0.0) {0} else{x(1093).toDouble},// DIST_IN_STANDING_FRESH
        if(x(1094).equals("?") || x(1094).toDouble < 0.0) {0} else{x(1094).toDouble},// DIST_FROM_WET_VEG_FRESH
        if(x(1095).equals("?") || x(1095).toDouble < 0.0) {0} else{x(1095).toDouble},// DIST_IN_WET_VEG_FRESH
        if(x(1096).equals("?") || x(1096).toDouble < 0.0) {0} else{x(1096).toDouble},// DIST_FROM_FLOWING_BRACKISH
        if(x(1097).equals("?") || x(1097).toDouble < 0.0) {0} else{x(1097).toDouble},// DIST_IN_FLOWING_BRACKISH
        if(x(1098).equals("?") || x(1098).toDouble < 0.0) {0} else{x(1098).toDouble},// DIST_FROM_STANDING_BRACKISH
        if(x(1099).equals("?") || x(1099).toDouble < 0.0) {0} else{x(1099).toDouble},// DIST_IN_STANDING_BRACKISH
        if(x(1100).equals("?") || x(1100).toDouble < 0.0) {0} else{x(1100).toDouble},// DIST_FROM_WET_VEG_BRACKISH
        if(x(1101).equals("?") || x(1101).toDouble < 0.0) {0} else{x(1101).toDouble}//  DIST_IN_WET_VEG_BRACKISH
      ))).persist

    println("Input "+ input.count())

    val parsed = input.map { case (k, vs) =>
      LabeledPoint(k.toDouble, Vectors.dense(vs.toArray))
    }
    // Split data into training (60%) and test (40%).
    val splits = parsed.randomSplit(Array(0.8, 0.2),seed= 5043l)
    val (trainingData, testData) = (splits(0), splits(1))

    //RANDOM FOREST
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int](6 -> 9,
      7 -> 9,
      8 -> 9,
      60 -> 9,
      61 -> 9,
      62 -> 9,
      63 -> 9,
      64 -> 9,
      65 -> 9,
      66 -> 9,
      67 -> 9,
      68 -> 9,
      69 -> 9,
      70 -> 9,
      71 -> 9)
    val numTrees = 21 // Use more in practice.
    val featureSubsetStrategy = "sqrt" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 15
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val accuracy = 1.0 * labelAndPreds.filter(x => x._1 == x._2).count() / testData.count()

    println("Accuracy: " + accuracy.toString)

    // Save and load model
    model.save(sc, args(1)+"_model/RandomForest")
    val sameModel = RandomForestModel.load(sc,args(1)+"_model/RandomForest")

    val unlabeled = sc.textFile(args(2)).mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .map(line => line.split(",")).map(x =>
      (x(0),List(
        if(x(955).equals("?") || x(955).toDouble < 0.0) {0} else{x(955).toDouble},
        if(x(956).equals("?") || x(956).toDouble < 0.0) {0} else{x(956).toDouble},
        if(x(957).equals("?") || x(957).toDouble < 0.0) {0} else{x(957).toDouble},
        if(x(959).equals("?") || x(959).toDouble < 0.0) {0} else{x(959).toDouble},
        if(x(960).equals("?") || x(960).toDouble < 0.0) {0} else{x(960).toDouble},
        if(x(962).equals("?") || x(962).toDouble < 0.0) {0} else{x(962).toDouble},
        if(x(963).equals("?") || x(963).toDouble < 0.0) {0} else{x(963).toDouble},
        if(x(964).equals("?") || x(964).toDouble < 0.0) {0} else{x(964).toDouble},
        if(x(965).equals("?") || x(965).toDouble < 0.0) {0} else{x(965).toDouble},
        if(x(966).equals("?") || x(966).toDouble < 0.0) {0} else{x(966).toDouble},
        if(x(967).equals("?") || x(967).toDouble < 0.0) {0} else{x(967).toDouble},
        if(x(968).equals("?") || x(968).toDouble < 0.0) {0} else{x(968).toDouble},
        if(x(969).equals("?") || x(969).toDouble < 0.0) {0} else{x(969).toDouble},
        if(x(970).equals("?") || x(970).toDouble < 0.0) {0} else{x(970).toDouble},
        if(x(971).equals("?") || x(971).toDouble < 0.0) {0} else{x(971).toDouble},
        if(x(972).equals("?") || x(972).toDouble < 0.0) {0} else{x(972).toDouble},
        if(x(973).equals("?") || x(973).toDouble < 0.0) {0} else{x(973).toDouble},
        if(x(974).equals("?") || x(974).toDouble < 0.0) {0} else{x(974).toDouble},
        if(x(975).equals("?") || x(975).toDouble < 0.0) {0} else{x(975).toDouble},
        if(x(976).equals("?") || x(976).toDouble < 0.0) {0} else{x(976).toDouble},
        if(x(977).equals("?") || x(977).toDouble < 0.0) {0} else{x(977).toDouble},
        if(x(978).equals("?") || x(978).toDouble < 0.0) {0} else{x(978).toDouble},
        if(x(979).equals("?") || x(979).toDouble < 0.0) {0} else{x(979).toDouble},
        if(x(980).equals("?") || x(980).toDouble < 0.0) {0} else{x(980).toDouble},
        if(x(981).equals("?") || x(981).toDouble < 0.0) {0} else{x(981).toDouble},
        if(x(982).equals("?") || x(982).toDouble < 0.0) {0} else{x(982).toDouble},
        if(x(983).equals("?") || x(983).toDouble < 0.0) {0} else{x(983).toDouble},
        if(x(984).equals("?") || x(984).toDouble < 0.0) {0} else{x(984).toDouble},
        if(x(985).equals("?") || x(985).toDouble < 0.0) {0} else{x(985).toDouble},
        if(x(986).equals("?") || x(986).toDouble < 0.0) {0} else{x(986).toDouble},
        if(x(987).equals("?") || x(987).toDouble < 0.0) {0} else{x(987).toDouble},
        if(x(988).equals("?") || x(988).toDouble < 0.0) {0} else{x(988).toDouble},
        if(x(989).equals("?") || x(989).toDouble < 0.0) {0} else{x(989).toDouble},
        if(x(990).equals("?") || x(990).toDouble < 0.0) {0} else{x(990).toDouble},
        if(x(991).equals("?") || x(991).toDouble < 0.0) {0} else{x(991).toDouble},
        if(x(992).equals("?") || x(992).toDouble < 0.0) {0} else{x(992).toDouble},
        if(x(993).equals("?") || x(993).toDouble < 0.0) {0} else{x(993).toDouble},
        if(x(994).equals("?") || x(994).toDouble < 0.0) {0} else{x(994).toDouble},
        if(x(995).equals("?") || x(995).toDouble < 0.0) {0} else{x(995).toDouble},
        if(x(996).equals("?") || x(996).toDouble < 0.0) {0} else{x(996).toDouble},
        if(x(997).equals("?") || x(997).toDouble < 0.0) {0} else{x(997).toDouble},
        if(x(998).equals("?") || x(998).toDouble < 0.0) {0} else{x(998).toDouble},
        if(x(999).equals("?") || x(999).toDouble < 0.0) {0} else{x(999).toDouble},
        if(x(1000).equals("?") || x(1000).toDouble < 0.0) {0} else{x(1000).toDouble},
        if(x(1001).equals("?") || x(1001).toDouble < 0.0) {0} else{x(1001).toDouble},
        if(x(1002).equals("?") || x(1002).toDouble < 0.0) {0} else{x(1002).toDouble},
        if(x(1003).equals("?") || x(1003).toDouble < 0.0) {0} else{x(1003).toDouble},
        if(x(1004).equals("?") || x(1004).toDouble < 0.0) {0} else{x(1004).toDouble},
        if(x(1005).equals("?") || x(1005).toDouble < 0.0) {0} else{x(1005).toDouble},
        if(x(1006).equals("?") || x(1006).toDouble < 0.0) {0} else{x(1006).toDouble},
        if(x(1007).equals("?") || x(1007).toDouble < 0.0) {0} else{x(1007).toDouble},
        if(x(1008).equals("?") || x(1008).toDouble < 0.0) {0} else{x(1008).toDouble},
        if(x(1009).equals("?") || x(1009).toDouble < 0.0) {0} else{x(1009).toDouble},
        if(x(1010).equals("?") || x(1010).toDouble < 0.0) {0} else{x(1010).toDouble},
        if(x(1011).equals("?") || x(1011).toDouble < 0.0) {0} else{x(1011).toDouble},
        if(x(1012).equals("?") || x(1012).toDouble < 0.0) {0} else{x(1012).toDouble},
        if(x(1013).equals("?") || x(1013).toDouble < 0.0) {0} else{x(1013).toDouble},
        if(x(1014).equals("?") || x(1014).toDouble < 0.0) {0} else{x(1014).toDouble},
        if(x(1015).equals("?") || x(1015).toDouble < 0.0) {0} else{x(1015).toDouble},
        if(x(1090).equals("?") || x(1090).toDouble < 0.0) {0} else{x(1090).toDouble},
        if(x(1091).equals("?") || x(1091).toDouble < 0.0) {0} else{x(1091).toDouble},
        if(x(1092).equals("?") || x(1092).toDouble < 0.0) {0} else{x(1092).toDouble},
        if(x(1093).equals("?") || x(1093).toDouble < 0.0) {0} else{x(1093).toDouble},
        if(x(1094).equals("?") || x(1094).toDouble < 0.0) {0} else{x(1094).toDouble},
        if(x(1095).equals("?") || x(1095).toDouble < 0.0) {0} else{x(1095).toDouble},
        if(x(1096).equals("?") || x(1096).toDouble < 0.0) {0} else{x(1096).toDouble},
        if(x(1097).equals("?") || x(1097).toDouble < 0.0) {0} else{x(1097).toDouble},
        if(x(1098).equals("?") || x(1098).toDouble < 0.0) {0} else{x(1098).toDouble},
        if(x(1099).equals("?") || x(1099).toDouble < 0.0) {0} else{x(1099).toDouble},
        if(x(1100).equals("?") || x(1100).toDouble < 0.0) {0} else{x(1100).toDouble},
        if(x(1101).equals("?") || x(1101).toDouble < 0.0) {0} else{x(1101).toDouble}
      ))).persist
    println("UnlabeledRDD "+unlabeled.count())

    val headerRDD:RDD[String] = sc.parallelize(Array("SAMPLING_EVENT_ID,SAW_AGELAIUS_PHOENICEUS"))

    val unlabeledParsed = unlabeled.map { case (s, vs) =>
      (s,Vectors.dense(vs.toArray))
    }

    val predictions = unlabeledParsed.map(point => point._1+","+sameModel.predict(point._2).toInt)

    sc.parallelize(headerRDD.union(predictions).collect(),1).saveAsTextFile(args(1))

    //GBT Model
    /*val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 20 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 6
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = scala.Predef.Map[Int, Int]()
    //Validation Tolerance Factor
    boostingStrategy.validationTol = -0.2
    val model = new GradientBoostedTrees(boostingStrategy).runWithValidation(trainingData, testData)
    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

   /* val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)*/
    val accuracy = 1.0 * labelAndPreds.filter(x => (x._1 == 1 && x._2 > 0.5) || (x._1 == 0 && x._2 < 0.5)).count() / testData.count()
    println("GBT Accuracy : "+accuracy);
    //println("Learned classification GBT model:\n" + model.toDebugString)

    // Save and load model
    model.save(sc, args(1)+"_model/GBT")
    val sameModel = GradientBoostedTreesModel.load(sc,args(1)+"_model/GBT")

    val unlabeled = sc.textFile(args(2)).mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .map(line => line.split(",")).map(x =>
      (x(0),List(
        if(x(955).equals("?") || x(955).toDouble < 0.0) {0} else{x(955).toDouble},
        if(x(956).equals("?") || x(956).toDouble < 0.0) {0} else{x(956).toDouble},
        if(x(957).equals("?") || x(957).toDouble < 0.0) {0} else{x(957).toDouble},
        if(x(959).equals("?") || x(959).toDouble < 0.0) {0} else{x(959).toDouble},
        if(x(960).equals("?") || x(960).toDouble < 0.0) {0} else{x(960).toDouble},
        if(x(962).equals("?") || x(962).toDouble < 0.0) {0} else{x(962).toDouble},
        if(x(963).equals("?") || x(963).toDouble < 0.0) {0} else{x(963).toDouble},
        if(x(964).equals("?") || x(964).toDouble < 0.0) {0} else{x(964).toDouble},
        if(x(965).equals("?") || x(965).toDouble < 0.0) {0} else{x(965).toDouble},
        if(x(966).equals("?") || x(966).toDouble < 0.0) {0} else{x(966).toDouble},
        if(x(967).equals("?") || x(967).toDouble < 0.0) {0} else{x(967).toDouble},
        if(x(968).equals("?") || x(968).toDouble < 0.0) {0} else{x(968).toDouble},
        if(x(969).equals("?") || x(969).toDouble < 0.0) {0} else{x(969).toDouble},
        if(x(970).equals("?") || x(970).toDouble < 0.0) {0} else{x(970).toDouble},
        if(x(971).equals("?") || x(971).toDouble < 0.0) {0} else{x(971).toDouble},
        if(x(972).equals("?") || x(972).toDouble < 0.0) {0} else{x(972).toDouble},
        if(x(973).equals("?") || x(973).toDouble < 0.0) {0} else{x(973).toDouble},
        if(x(974).equals("?") || x(974).toDouble < 0.0) {0} else{x(974).toDouble},
        if(x(975).equals("?") || x(975).toDouble < 0.0) {0} else{x(975).toDouble},
        if(x(976).equals("?") || x(976).toDouble < 0.0) {0} else{x(976).toDouble},
        if(x(977).equals("?") || x(977).toDouble < 0.0) {0} else{x(977).toDouble},
        if(x(978).equals("?") || x(978).toDouble < 0.0) {0} else{x(978).toDouble},
        if(x(979).equals("?") || x(979).toDouble < 0.0) {0} else{x(979).toDouble},
        if(x(980).equals("?") || x(980).toDouble < 0.0) {0} else{x(980).toDouble},
        if(x(981).equals("?") || x(981).toDouble < 0.0) {0} else{x(981).toDouble},
        if(x(982).equals("?") || x(982).toDouble < 0.0) {0} else{x(982).toDouble},
        if(x(983).equals("?") || x(983).toDouble < 0.0) {0} else{x(983).toDouble},
        if(x(984).equals("?") || x(984).toDouble < 0.0) {0} else{x(984).toDouble},
        if(x(985).equals("?") || x(985).toDouble < 0.0) {0} else{x(985).toDouble},
        if(x(986).equals("?") || x(986).toDouble < 0.0) {0} else{x(986).toDouble},
        if(x(987).equals("?") || x(987).toDouble < 0.0) {0} else{x(987).toDouble},
        if(x(988).equals("?") || x(988).toDouble < 0.0) {0} else{x(988).toDouble},
        if(x(989).equals("?") || x(989).toDouble < 0.0) {0} else{x(989).toDouble},
        if(x(990).equals("?") || x(990).toDouble < 0.0) {0} else{x(990).toDouble},
        if(x(991).equals("?") || x(991).toDouble < 0.0) {0} else{x(991).toDouble},
        if(x(992).equals("?") || x(992).toDouble < 0.0) {0} else{x(992).toDouble},
        if(x(993).equals("?") || x(993).toDouble < 0.0) {0} else{x(993).toDouble},
        if(x(994).equals("?") || x(994).toDouble < 0.0) {0} else{x(994).toDouble},
        if(x(995).equals("?") || x(995).toDouble < 0.0) {0} else{x(995).toDouble},
        if(x(996).equals("?") || x(996).toDouble < 0.0) {0} else{x(996).toDouble},
        if(x(997).equals("?") || x(997).toDouble < 0.0) {0} else{x(997).toDouble},
        if(x(998).equals("?") || x(998).toDouble < 0.0) {0} else{x(998).toDouble},
        if(x(999).equals("?") || x(999).toDouble < 0.0) {0} else{x(999).toDouble},
        if(x(1000).equals("?") || x(1000).toDouble < 0.0) {0} else{x(1000).toDouble},
        if(x(1001).equals("?") || x(1001).toDouble < 0.0) {0} else{x(1001).toDouble},
        if(x(1002).equals("?") || x(1002).toDouble < 0.0) {0} else{x(1002).toDouble},
        if(x(1003).equals("?") || x(1003).toDouble < 0.0) {0} else{x(1003).toDouble},
        if(x(1004).equals("?") || x(1004).toDouble < 0.0) {0} else{x(1004).toDouble},
        if(x(1005).equals("?") || x(1005).toDouble < 0.0) {0} else{x(1005).toDouble},
        if(x(1006).equals("?") || x(1006).toDouble < 0.0) {0} else{x(1006).toDouble},
        if(x(1007).equals("?") || x(1007).toDouble < 0.0) {0} else{x(1007).toDouble},
        if(x(1008).equals("?") || x(1008).toDouble < 0.0) {0} else{x(1008).toDouble},
        if(x(1009).equals("?") || x(1009).toDouble < 0.0) {0} else{x(1009).toDouble},
        if(x(1010).equals("?") || x(1010).toDouble < 0.0) {0} else{x(1010).toDouble},
        if(x(1011).equals("?") || x(1011).toDouble < 0.0) {0} else{x(1011).toDouble},
        if(x(1012).equals("?") || x(1012).toDouble < 0.0) {0} else{x(1012).toDouble},
        if(x(1013).equals("?") || x(1013).toDouble < 0.0) {0} else{x(1013).toDouble},
        if(x(1014).equals("?") || x(1014).toDouble < 0.0) {0} else{x(1014).toDouble},
        if(x(1015).equals("?") || x(1015).toDouble < 0.0) {0} else{x(1015).toDouble},
        if(x(1091).equals("?") || x(1091).toDouble < 0.0) {0} else{x(1091).toDouble},
        if(x(1092).equals("?") || x(1092).toDouble < 0.0) {0} else{x(1092).toDouble},
        if(x(1093).equals("?") || x(1093).toDouble < 0.0) {0} else{x(1093).toDouble},
        if(x(1094).equals("?") || x(1094).toDouble < 0.0) {0} else{x(1094).toDouble},
        if(x(1095).equals("?") || x(1095).toDouble < 0.0) {0} else{x(1095).toDouble},
        if(x(1096).equals("?") || x(1096).toDouble < 0.0) {0} else{x(1096).toDouble},
        if(x(1097).equals("?") || x(1097).toDouble < 0.0) {0} else{x(1097).toDouble},
        if(x(1098).equals("?") || x(1098).toDouble < 0.0) {0} else{x(1098).toDouble},
        if(x(1099).equals("?") || x(1099).toDouble < 0.0) {0} else{x(1099).toDouble},
        if(x(1100).equals("?") || x(1100).toDouble < 0.0) {0} else{x(1100).toDouble},
        if(x(1101).equals("?") || x(1101).toDouble < 0.0) {0} else{x(1101).toDouble},
        if(x(1102).equals("?") || x(1102).toDouble < 0.0) {0} else{x(1102).toDouble}
      ))).persist
    println("UnlabeledRDD "+unlabeled.count())

    val headerRDD:RDD[String] = sc.parallelize(Array("SAMPLING_EVENT_ID,SAW_AGELAIUS_PHOENICEUS"))

    val unlabeledParsed = unlabeled.map { case (s, vs) =>
      (s,Vectors.dense(vs.toArray))
    }

    val predictions = unlabeledParsed.map(point => point._1+","+sameModel.predict(point._2).toInt)
    sc.parallelize(headerRDD.union(predictions).collect(),1).saveAsTextFile(args(1))

    //Decision Tree
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 15
    val maxBins = 64

    val decModel = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val predictionAndLabel = testData.map(p => (decModel.predict(p.features), p.label))
    val accuracyDec = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testData.count()
    System.out.println("DEC Accuracy: "  + accuracyDec.toString());

    // Save and load model
    decModel.save(sc, args(1)+"_model/DecisionTree")
    val sameDecModel = DecisionTreeModel.load(sc, args(1)+"_model/DecisionTree")


    val predictionsDed = unlabeledParsed.map(point => point._1+","+sameDecModel.predict(point._2).toInt)
    sc.parallelize(headerRDD.union(predictions).collect(),1).saveAsTextFile(args(1)+"DEC")*/
  }
}