import UIKit

extension Double {
    public static var random: Double {
        get {
            return Double(arc4random()) / 0xFFFFFFFF
        }
    }
}

class NeuralNetwork {
    
    let inputNodes = 2
    let hiddenNodes = 2
    let outputNodes = 1
    
    var totalNodes: Int {
        return inputNodes + hiddenNodes + outputNodes
    }
    
    var values = [Double]()
    var weights = [[Double]]()
    
    var trainSet = (0, 0) {
        didSet {
            values[0] = Double(trainSet.0)
            values[1] = Double(trainSet.1)
        }
    }
    
    var expectedResult = 0
    
    init() {
        values = Array(repeating: 0, count: totalNodes)
        weights = Array(repeating: Array(repeating: 0, count: totalNodes), count: totalNodes)
        for i in 0..<totalNodes {
            for j in 0..<totalNodes {
                weights[i][j] = Double.random * 2
            }
        }
    }
    
    private func sigmoid(x: Double) -> Double {
        return 1/(1 + pow(M_E, -x))
    }
    
    func process() {
        for i in inputNodes..<inputNodes + hiddenNodes {
            var sum: Double = 0
            for j in 0..<inputNodes {
                sum += weights[j][i] * values[j]
            }
            values[i] = sigmoid(x: sum)
        }
        
        for i in inputNodes + hiddenNodes..<totalNodes {
            var sum: Double = 0
            for j in inputNodes..<inputNodes + hiddenNodes {
                sum += weights[j][i] * values[j]
            }
            values[i] = sigmoid(x: sum)
        }
    }
    
    func processError() {
        for i in inputNodes + hiddenNodes ..< totalNodes {
            let error = Double(expectedResult) - values[i]
            let outputErrorGradient = values[i] * (1 - values[i]) * error
            
            for j in inputNodes ..< inputNodes + hiddenNodes {
                let delta = values[j] * outputErrorGradient
                weights[j][i] += delta
                let hiddenErrorGradient = values[j] * (1 - values[j]) * outputErrorGradient * weights[j][i]
                
                for k in 0 ..< inputNodes {
                    let delta = values[k] * hiddenErrorGradient
                    weights[k][j] += delta
                }
            }
        }
    }
    
}

let trainSets = [(0, 0), (0, 1), (1, 0), (1, 1)]
let expectedResults = [0, 1, 1, 0]

let n = NeuralNetwork()
let iterations = 5000

for i in 0...iterations {
    for (pos, set) in trainSets.enumerated() {
        n.trainSet = set
        n.expectedResult = expectedResults[pos]
        n.process()
        n.processError()
        if i > iterations - 5 {
            print("out: \(n.values.last!), expected: \(expectedResults[pos])")
            print("\n")
        }
    }
}


