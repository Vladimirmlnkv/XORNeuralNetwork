//: Playground - noun: a place where people can play

import UIKit

extension Double {
    public static var random:Double {
        get {
            return Double(arc4random()) / 0xFFFFFFFF
        }
    }

    public static func random(min: Double, max: Double) -> Double {
        return Double.random * (max - min) + min
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
    var savedResults = [Double]()
    
    var trainSet = (0, 0) {
        didSet {
            values[0] = Double(trainSet.0)
            values[1] = Double(trainSet.1)
        }
    }
    var expectedResults = [Int]()
    
    init() {
        values = Array(repeating: 0, count: totalNodes)
        weights = Array(repeating: Array(repeating: 0, count: totalNodes), count: totalNodes)
        for i in 0..<totalNodes {
            for j in 0..<totalNodes {
                weights[i][j] = Double.random(min: -10, max: 10)
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
        savedResults.append(values.last!)
    }
    
    func calculateError() {
        var sum: Double = 0
        for (pos, res) in savedResults.enumerated() {
            let expected = Double(expectedResults[pos])
            sum += pow(Double(expected) - res, 2)
        }
        let error = sum / Double(savedResults.count)
        print(error)
    }
    
}

let trainSets = [(0, 0), (0, 1), (1, 0), (1, 1)]
let expectedResults = [0, 1, 1, 0]

let n = NeuralNetwork()
n.expectedResults = expectedResults

for (pos, set) in trainSets.enumerated() {
    n.trainSet = set
    n.process()
    n.calculateError()
}
