import breeze.linalg.{DenseMatrix, DenseVector}

class DataSet(features: DenseMatrix[Double], target: DenseVector[Double]) {
  def getFeatures: DenseMatrix[Double] = {
    features
  }

  def getTarget: DenseVector[Double] = {
    target
  }
}
