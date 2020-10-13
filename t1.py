import numpy as np
import NeuralNetworkEdited as nn
import matplotlib.pyplot as plt

#load the data from txt file
name='c:/Users/Freire/Downloads/tareaRedesNeuronales/t1Data.txt'
data=np.loadtxt(name,delimiter=',',skiprows=0)

#randomize the data set
np.random.shuffle(data)

#split the randomized data set in 80% and 20%
learningData=data[0:96]
testingData=data[96:]

#function for initialize the confusion matrix
def initializeConfusionMatrix(numberOfClasses):
    confusionMatrix=np.zeros((numberOfClasses,numberOfClasses))
    return confusionMatrix

#function for the normalization the data sets
def normalization(x,nh,nl):
    for j in range(0,x.shape[1]):
        maxValue=max(x[:,j])
        minValue=min(x[:,j])
        for i in range(0,x.shape[0]):
            x[i][j]=(((x[i][j]-minValue) * (nh-nl))/(maxValue-minValue)) + nl

#function for calculate the precition of the label x in the confusion matrix
def precision(matrix,labelX):
    #TP_X (labelX element of the diagonal of the matrix)
    truePositivesX=matrix[labelX][labelX]

    #total predicted as a particular label x (sum of the row labelX)
    totalPredictedX=matrix[labelX, :].sum()

    precisionX=truePositivesX/totalPredictedX
    return precisionX

#function for calculate the recall of the label x in the confusion matrix
def recall(matrix,labelX):
    #TP_X (labelX element of the diagonal of the matrix)
    truePositivesX=matrix[labelX][labelX]

    #total number of instances that should have label x (sum of the column labelX)
    totalLabel=matrix[:, labelX].sum()
    
    recallX=truePositivesX/totalLabel
    return recallX

#function for apply the one hot encoding to the outputs of the original data set
#apply for 2 classes (the represent of 4 clases in terms of combination (0,0) (0,1) (1,0) (1,1))
def oneHotEncoding(matrix,numberInputs,numberOfClasses):
    #matrix for put the one hot encoding new outputs
    oneHotMatrix=np.zeros((matrix.shape[0],numberInputs+numberOfClasses))
    for i in range(0,matrix.shape[0]):
        #output1 is 0
        if matrix[i][numberInputs]==0:
            oneHotMatrix[i][numberInputs+1]=1 #one hot output is (0,1)

        #output1 is 1
        else:
            oneHotMatrix[i][numberInputs]=1 #one hot output is (0,1)

    #slice of the original matrix (contains only the inputs)
    slicedMatrix=matrix[:,0:numberInputs]
    #slice of the oneHotMatrix (contains only the outputs after being "one hot encoded")
    slicedOneHotMatrix=oneHotMatrix[:,numberInputs:]
    #final oneHotMatrix with the concatenation of the two matrix
    finalOneHotMatrix=np.append(slicedMatrix,slicedOneHotMatrix,1)

    return finalOneHotMatrix
    
#function for make predictions and save it in array
def makePredictions(testingArray,model,numberOfInputs):
    predictionArray=np.zeros((testingArray.shape[0],2))
    #make the predictions 
    for i in range(0,testingArray.shape[0]):
        #make the prediction for that row
        predictionI=nn.predict(testingArray[i,0:numberOfInputs].reshape(numberOfInputs,1),model)
        if(predictionI==1):
            oneHotPrediction=np.array([1,0])
        else:
            oneHotPrediction=np.array([0,1])
        predictionArray[i]=oneHotPrediction

    return predictionArray

#function for complete the confusion matrix with the values of true positives,
#true negatives, false positives and false negatives
def completeConfusionMatrix(predictionArray,realOutputsArray,labelA,labelB):
    #create the rows that contain the predictions
    predictedA=np.zeros(2)
    predictedB=np.zeros(2)

    #create the confusion matrix
    confusionMatrix=initializeConfusionMatrix(2)
    
    for i in range(0,predictionArray.shape[0]):
        #get the expected value
        realOutput=realOutputsArray[i][6]

        ##complete the predictedA and predictedB arrays
        if(predictionArray[i][0]==labelA):
            #verify if it is true negative
            if(realOutput==labelA):
                predictedA[0]+=1
            #it is a false negative
            else:
                predictedA[1]+=1
        if(predictionArray[i][0]==labelB):
            #verify if it is true positive
            if(realOutput==labelB):
                predictedB[1]+=1
            #it is a false positive
            else:
                predictedB[0]+=1

    #complete the confusion matrix
    confusionMatrix[0]=predictedA
    confusionMatrix[1]=predictedB

    return confusionMatrix

#function for calculate the recall above al the labels of the confusion matrix
def calculateRecall(confusion,numberOfClasses):
    confusionMatrix=confusion
    recallValues=np.zeros(2)
    for i in range(0,numberOfClasses):
        recallValues[i]=recall(confusionMatrix,i)
    return recallValues

#function for calculate the precision above al the labels of the confusion matrix
def calculatePrecision(confusion,numberOfClasses):
    confusionMatrix=confusion
    precisionValues=np.zeros(2)
    for i in range(0,numberOfClasses):
        precisionValues[i]=precision(confusionMatrix,i)
    return precisionValues


#normalizate the data sets (learning and testing)
normalization(testingData,1,0)
normalization(learningData,1,0)

#get the learning data inputs and the learning data outputs 
learningDataInputs=learningData[:, 0:6].T
learningDataOutputs=learningData[:, 6:7].T

#set the number of iterations and the learning rate
number_of_iterations = 10000
learning_rate = 0.01

#set the number of inputs,outputs and neurons in hidden layer
hiddenLayer=20
numberInputs=6
numberOfOutputs=1

#create the cost and iteration array
costArray=np.empty(number_of_iterations+1)
iterArray=np.empty(number_of_iterations+1)

#get the piece of testing data without the output value of each row
testingDataNew1=testingData[:, 0:6]

#get the model for the predictions
model=nn.model(learningDataInputs,learningDataOutputs,numberInputs,hiddenLayer,numberOfOutputs,number_of_iterations,learning_rate,costArray,iterArray)

#get the predictions with 6 inputs
predictions=makePredictions(testingDataNew1,model,6)

#get the confusion matrix with the labels 0 and 1
confusionMatrix=completeConfusionMatrix(predictions,testingData,0.0,1.0)

#calculate the recall and precision values for the confusion matrix
recallValues=calculateRecall(confusionMatrix,2)
precisionValues=calculatePrecision(confusionMatrix,2)

#Print the results
for i in range(0,testingData.shape[0]):
    print("Neural Network prediction for example:","["+str(testingData[i][0]), testingData[i][1], testingData[i][2],
    testingData[i][3], testingData[i][4], str(testingData[i][5])+"]","is: ["+str(int(predictions[i][0]))+str(int(predictions[i][1]))+"]")

#print the confusion matrix
print("Confusion matrix:","\n",confusionMatrix)

#print the recall and precision values of each label
print("Recall value for label 0:",recallValues[0])
print("Recall value for label 1:",recallValues[1])
print("Precision value for label 0:",precisionValues[0])
print("Precision value for label 1:",precisionValues[1])

#plot the results
labels1 = 'True Negative', 'False Negative', 'False Positive','True Positive'
sizes = [confusionMatrix[0][0],confusionMatrix[0][1],confusionMatrix[1][0],confusionMatrix[1][1]]
plt.pie(sizes,labels=labels1)
plt.show()

#chart the cost vs iterations
plt.plot(iterArray,costArray)
plt.title('Cost vs iterations')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()