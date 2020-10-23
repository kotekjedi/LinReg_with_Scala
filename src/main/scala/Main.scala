import java.io.File

import breeze.linalg.csvwrite

object Main {
  def main(args: Array[String]): Unit = {
//    loading fish splited fish data source: https://www.kaggle.com/aungpyaeap/fish-market
    val trainDataSet = DataLoader.loadData("src/main/scala/Fish_train.csv", train = true)
    val testDataSet = DataLoader.loadData("src/main/scala/Fish_test.csv")

    val linReg = new LinearRegression

    linReg.fit(trainDataSet.getFeatures, trainDataSet.getTarget)

    print("Train error:")
    println(linReg.mse(trainDataSet.getTarget, linReg.predict(trainDataSet.getFeatures)))

    print("Test error:")
    println(linReg.mse(testDataSet.getTarget, linReg.predict(testDataSet.getFeatures)))

    csvwrite(new File("src/main/scala/out.csv"), linReg.predict(testDataSet.getFeatures).toDenseMatrix)
  }
}
