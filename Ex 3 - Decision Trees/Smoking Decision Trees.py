
# ---------------------------- Import Libraries ----------------------------

import copy
import math
import pandas as pd
from pptree import *
import numpy as np
from scipy.stats import chi2

# ---------------------------- Data Fix ----------------------------

def getData():
    global column_titles
    df = pd.read_csv(f'Smoking.csv', encoding='unicode_escape')
    print(df)

    # ----- Fix Names -----
    df = df.drop('ID', axis=1)
    df = df.rename(columns={'gender': 'Gender'})
    df = df.rename(columns={'age': 'Age'})
    df = df.rename(columns={'height(cm)': 'Height(cm)'})
    df = df.rename(columns={'weight(kg)': 'Weight(kg)'})
    df = df.rename(columns={'waist(cm)': 'Waist(cm)'})
    df = df.rename(columns={'eyesight(right)': 'Eyesight_R'})
    df = df.rename(columns={'eyesight(left)': 'Eyesight_L'})
    df = df.rename(columns={'hearing(right)': 'Hearing_R'})
    df = df.rename(columns={'hearing(left)': 'Hearing_L'})
    df = df.rename(columns={'systolic': 'Systolic'})
    df = df.rename(columns={'relaxation': 'Relaxation'})
    df = df.rename(columns={'fasting blood sugar': 'FBS'})
    # Cholesterol
    df = df.rename(columns={'triglyceride': 'Triglyceride'})
    # HDL
    # LDL
    df = df.rename(columns={'hemoglobin': 'Hemoglobin'})
    df = df.rename(columns={'Urine protein': 'UrineProtein'})
    df = df.rename(columns={'serum creatinine': 'SerumCreatinine'})
    # AST
    # ALT
    df = df.rename(columns={'Gtp': 'GTP'})
    df = df.rename(columns={'dental caries': 'DentalCaries'})
    df = df.rename(columns={'tartar': 'Tartar'})
    df = df.rename(columns={'smoking': 'Smoking'})

    # ----- Change to Bins -----
    # Gender
    df['Age'] = df.apply(bin_Age, axis=1)
    df['Height(cm)'] = df.apply(bin_Height, axis=1)
    df['Weight(kg)'] = df.apply(bin_Weight, axis=1)
    df['Waist(cm)'] = df.apply(bin_Waist, axis=1)
    df['Eyesight_L'] = df.apply(bin_Eyesight_L, axis=1)
    df['Eyesight_R'] = df.apply(bin_Eyesight_R, axis=1)
    df['Hearing_L'] = df.apply(bin_Hearing_L, axis=1)
    df['Hearing_R'] = df.apply(bin_Hearing_R, axis=1)
    df['Systolic'] = df.apply(bin_Systolic, axis=1)
    df['Relaxation'] = df.apply(bin_Relaxation, axis=1)
    df['FBS'] = df.apply(bin_FBS, axis=1)
    df['Cholesterol'] = df.apply(bin_Cholesterol, axis=1)
    df['Triglyceride'] = df.apply(bin_Triglyceride, axis=1)
    df['HDL'] = df.apply(bin_HDL, axis=1)
    df['LDL'] = df.apply(bin_LDL, axis=1)
    df['Hemoglobin'] = df.apply(bin_Hemoglobin, axis=1)
    df['UrineProtein'] = df.apply(bin_UrineProtein, axis=1)
    df['SerumCreatinine'] = df.apply(bin_SerumCreatinine, axis=1)
    df['AST'] = df.apply(bin_AST, axis=1)
    df['ALT'] = df.apply(bin_ALT, axis=1)
    df['GTP'] = df.apply(bin_GTP, axis=1)
    df['DentalCaries'] = df.apply(bin_DentalCaries, axis=1)
    # Tartar
    df['Smoking'] = df.apply(bin_Smoking, axis=1)

    column_titles = df.columns
    return df

def getAttributes():
    attributes = {}
    attributes.update({'Gender': ['F', 'M']})
    attributes.update({'Age': ['20-35', '35-50', '50-65', '70+']})
    attributes.update({'Height(cm)': ['165-', '170-175', '180+']})
    attributes.update({'Weight(kg)': ['55-', '60-85', '90+']})
    attributes.update({'Waist(cm)': ['75-', '80-100', '100+', 'Q4']})
    attributes.update({'Eyesight_L': ['Normal', 'Under Normal', 'Above Normal']})
    attributes.update({'Eyesight_R': ['Normal', 'Under Normal', 'Above Normal']})
    attributes.update({'Hearing_L': ['Normal', 'Not Normal']})
    attributes.update({'Hearing_R': ['Normal', 'Not Normal']})
    attributes.update({'Systolic': ['Optimal', 'Normal', 'Normal High', 'High']})
    attributes.update({'Relaxation': ['Optimal', 'Normal', 'Normal High', 'High']})
    attributes.update({'FBS': ['Normal', 'Diabetes', 'PreDiabetes']})
    attributes.update({'Cholesterol': ['Desirable', 'Borderline High', 'High']})
    attributes.update({'Triglyceride': ['Normal', 'Borderline High', 'High']})
    attributes.update({'HDL': ['Optimal', 'Above Optimal', 'High']})
    attributes.update({'LDL': ['Optimal', 'Above Optimal', 'High']})
    attributes.update({'Hemoglobin': ['Normal', 'Slightly Low', 'Significantly Low']})
    attributes.update({'UrineProtein': ['Normal', 'Slightly Elevated', 'Significantly Elevated']})
    attributes.update({'SerumCreatinine': ['Normal', 'Slightly Elevated', 'Significantly Elevated']})
    attributes.update({'AST': ['Normal', 'Slightly Elevated', 'Significantly Elevated']})
    attributes.update({'ALT': ['Normal', 'Slightly Elevated', 'Significantly Elevated']})
    attributes.update({'GTP': ['Normal', 'Slightly Elevated', 'Significantly Elevated']})
    attributes.update({'DentalCaries': ['Y', 'N']})
    attributes.update({'Tartar': ['Y', 'N']})
    attributes.update({'Smoking': ['Smoking', 'Not Smoking']})
    return attributes

def bin_Triglyceride(example):
    if int(example['Triglyceride']) <= 150:
        return 'Normal'
    elif 150 < int(example['Triglyceride']) <= 199:
        return 'Borderline High'
    else:
        return 'High'

def bin_Cholesterol(example):
    if int(example['Cholesterol']) <= 200:
        return 'Desirable'
    elif 200 < int(example['Cholesterol']) <= 239:
        return 'Borderline High'
    else:
        return 'High'

def bin_Smoking(example):
    if int(example['Smoking']) == 1:
        return 'Smoking'
    else:
        return 'Not Smoking'

def bin_DentalCaries(example):
    if int(example['DentalCaries']) == 1:
        return 'Y'
    else:
        return 'N'

def bin_GTP(example):
    if int(example['GTP']) <= 70:
        return 'Normal'
    elif 70 < int(example['GTP']) <= 190:
        return 'Slightly Elevated'
    else:
        return 'Significantly Elevated'

def bin_ALT(example):
    if int(example['ALT']) <= 56:
        return 'Normal'
    elif 56 < int(example['ALT']) <= 125:
        return 'Slightly Elevated'
    else:
        return 'Significantly Elevated'

def bin_AST(example):
    if int(example['AST']) <= 40:
        return 'Normal'
    elif 40 < int(example['AST']) <= 80:
        return 'Slightly Elevated'
    else:
        return 'Significantly Elevated'

def bin_SerumCreatinine(example):
    if int(example['SerumCreatinine']) <= 1.2:
        return 'Normal'
    elif 1.2 < int(example['SerumCreatinine']) <= 1.9:
        return 'Slightly Elevated'
    else:
        return 'Significantly Elevated'

def bin_UrineProtein(example):
    if int(example['UrineProtein']) < 3:
        return 'Normal'
    elif int(example['UrineProtein']) == 3:
        return 'Slightly Elevated'
    else:
        return 'Significantly Elevated'

def bin_Hemoglobin(example):
    if int(example['Hemoglobin']) <= 11:
        return 'Significantly Low'
    elif 11 < int(example['Hemoglobin']) <= 13.5:
        return 'Slightly Low'
    else:
        return 'Normal'

def bin_LDL(example):
    if int(example['LDL']) <= 40:
        return 'High'
    elif 40 < int(example['LDL']) <= 60:
        return 'Above Optimal'
    else:
        return 'Optimal'

def bin_HDL(example):
    if int(example['HDL']) <= 100:
        return 'Optimal'
    elif 100 < int(example['HDL']) <= 139:
        return 'Above Optimal'
    else:
        return 'High'

def bin_FBS(example):
    if int(example['FBS']) <= 100:
        return 'Normal'
    elif 100 < int(example['FBS']) <= 125:
        return 'PreDiabetes'
    else:
        return 'Diabetes'

def bin_Relaxation(example):
    if int(example['Relaxation']) <= 80:
        return 'Optimal'
    elif 80 < int(example['Relaxation']) <= 84:
        return 'Normal'
    elif 84 < int(example['Relaxation']) <= 89:
        return 'Normal High'
    else:
        return 'High'

def bin_Systolic(example):
    if int(example['Systolic']) <= 120:
        return 'Optimal'
    elif 120 < int(example['Systolic']) <= 129:
        return 'Normal'
    elif 129 < int(example['Systolic']) <= 139:
        return 'Normal High'
    else:
        return 'High'

def bin_Hearing_R(example):
    # Right
    if int(example['Hearing_R']) == 1:
        return 'Normal'
    elif int(example['Hearing_R']) > 1:
        return 'Not Normal'

def bin_Hearing_L(example):
    # Left
    if int(example['Hearing_L']) == 1:
        return 'Normal'
    elif int(example['Hearing_L']) > 1:
        return 'Not Normal'

def bin_Eyesight_R(example):
    # Right
    if int(example['Eyesight_R']) == 1:
        return 'Normal'
    elif int(example['Eyesight_R']) < 1:
        return 'Under Normal'
    elif int(example['Eyesight_R']) > 1:
        return 'Above Normal'

def bin_Eyesight_L(example):
    # Left
    if int(example['Eyesight_L']) == 1:
        return 'Normal'
    elif int(example['Eyesight_L']) < 1:
        return 'Under Normal'
    elif int(example['Eyesight_L']) > 1:
        return 'Above Normal'

def bin_Waist(example):
    if int(example['Waist(cm)']) <= 75:
        return '75-'
    elif 75 < int(example['Waist(cm)']) <= 100:
        return '80-100'
    else:
        return '100+'

def bin_Height(example):
    if int(example['Height(cm)']) <= 165:
        return '165-'
    elif 165 < int(example['Height(cm)']) <= 175:
        return '170-175'
    else:
        return '180+'

def bin_Weight(example):
    if int(example['Weight(kg)']) <= 55:
        return '55-'
    elif 55 < int(example['Weight(kg)']) <= 85:
        return '60-85'
    else:
        return '90+'

def bin_Age(example):
    if int(example['Age']) <= 35:
        return '20-35'
    elif 35 < int(example['Age']) <= 50:
        return '35-50'
    elif 50 < int(example['Age']) <= 65:
        return '50-65'
    else:
        return '70+'

# ---------------------------- DT Object ----------------------------

class DecisionTree:
    def __init__(self, attribute, optionStr, father, ans, myExample, ifLeave):
        if ans == '':
            self.name = attribute
        else:
            self.name = str(ans) + ', ' + attribute
        self.children = []  # List of thr children trees.
        self.optionStr = optionStr
        self.parent = father  # Fatherhood - The tree that its came from.
        self.attribute = attribute
        self.ans = ans
        self.myExample = myExample
        self.ifLeave = ifLeave

    def __str__(self):
        t = self.name
        return t

    def proning(self):
        for child in self.children:     # For every child that is not a leaf.
            if not child.ifLeave:
                child.proning()
        ifCut = chiSquaredTest(self.myExample, self.parent.myExample, self)
        if not ifCut: #we need to cut
            if self.attribute == 'Smoking' or self.attribute == 'Not Smoking':
                self.parent.attribute = self.attribute
                print(self.parent.attribute)
                self.parent.children.remove(self)

# ---------------------------- Functions ----------------------------

def build_tree(ratio):
    print('build_tree')
    df = getData()    # Function that return normal dataframe.
    print("Done Read Data.")
    attributes = getAttributes()    # Function that return a dictionary of the attributes.
    msk = np.random.rand(len(df)) < ratio
    # Train Data and Test Data split by ratio:
    train = df[msk]
    test = df[~msk]
    fictive_Tree = DecisionTree('fictive', '', '', '', train, False)
    print("Calculating Decision Tree.")
    myTree = decisionTreeLearning(train, attributes, fictive_Tree, '', '')   # Call the recursive function.
    print("Proning Decision Tree.")
    myTree.proning()
    print("Printing Decision Tree.")
    print_tree(myTree, "children", "name", horizontal=True)     # Printing tree.
    print("Checking Error.")
    treeError = calculateError(myTree, test)
    print('The Decision Tree Error Rate Is: ' + str(round((treeError)*100, 2)) + "%")

def chiSquaredTest(ex, exFather, subTree):
    smokingFather = len(exFather[exFather['Smoking'] == 'Smoking'])
    notSmokingFather = len(exFather[exFather['Smoking'] == 'Not Smoking'])
    freedomDegrees = smokingFather + notSmokingFather - 1
    crit = chi2.ppf(q=0.05, df=freedomDegrees)       # Use in package that calculate the value from the Chi-Test table.
    delta = 0
    for ans in subTree.children:
        newData = ex[ex[subTree.attribute] == ans.ans]
        if not newData.empty:   # Calculate all the parameters for the delta value.
            newSmoking = len(newData[newData['Smoking'] == 'Smoking'])
            newNotSmoking = len(newData[newData['Smoking'] == 'Not Smoking'])
            Pk = smokingFather * (newSmoking+newNotSmoking) / (smokingFather + notSmokingFather)
            Nk = notSmokingFather * (newSmoking+newNotSmoking) / (smokingFather + notSmokingFather)
            delta = delta + ((newSmoking-Pk)**2/Pk) + ((newNotSmoking-Nk)**2/Nk)
    if delta < crit:    # When the critical value is less than the calculated value, we want to cut the increase.
        return False
    else:
        return True

def calculateError(tree, test):
    totalErrors = 0
    list2D = []
    for row in test.iterrows():     # For every row in the dataframe.
        if type(row) == tuple:
            list = row[1].values.tolist()
            list2D = rowToList(list)  # Change to 2D array according to the data.
        totalErrors = totalErrors + getSpecificError(tree, list2D)
    return totalErrors/len(test)    # Calculate the average.

def rowToList(list):#build the array
    temp = []
    temp.append(['Gender', list[0]])
    temp.append(['Age', list[1]])
    temp.append(['Height(cm)', list[2]])
    temp.append(['Weight(kg)', list[3]])
    temp.append(['Waist(cm)', list[4]])
    temp.append(['Eyesight_L', list[5]])
    temp.append(['Eyesight_R', list[6]])
    temp.append(['Hearing_L', list[7]])
    temp.append(['Hearing_R', list[8]])
    temp.append(['Systolic', list[9]])
    temp.append(['Relaxation', list[10]])
    temp.append(['FBS', list[11]])
    temp.append(['Cholesterol', list[12]])
    temp.append(['Triglyceride', list[13]])
    temp.append(['HDL', list[14]])
    temp.append(['LDL', list[15]])
    temp.append(['Hemoglobin', list[16]])
    temp.append(['UrineProtein', list[17]])
    temp.append(['SerumCreatinine', list[18]])
    temp.append(['AST', list[19]])
    temp.append(['ALT', list[20]])
    temp.append(['GTP', list[21]])
    temp.append(['DentalCaries', list[22]])
    temp.append(['Tartar', list[23]])
    temp.append(['Smoking', list[24]])
    return temp


def getSpecificError(tree, list):  # Calculate the error for every row, if wrong return 1 else return 0.
    if tree.ans == 'Smoking' or tree.ans == 'Not Smoking':
        if list[24][1] == tree.ans:
            return 0
        else:
            return 1
    elif tree.attribute == 'Smoking' or tree.attribute == 'Not Smoking':
        if list[24][1] == tree.attribute:
            return 0
        else:
            return 1
    else:
        for x in range(len(list)):
            if list[x][0] == tree.attribute:     # Same Group.
                for child in tree.children:
                    if child.ans == '':
                        if list[24][1] == tree.attribute:
                            return 0
                        else:
                            return 1
                    elif list[x][1] == child.ans:   # Same Sub Group.
                        return getSpecificError(child, list)

def decisionTreeLearning(dataTrain, attributes, cameFrom, parentex, ans):
    #print('decisionTreeLearning, len(dataTrain): ', len(dataTrain))

    if len(dataTrain) == 0:     # Stop Condition : No more data.
        tree = getMajority(parentex, cameFrom, ans, dataTrain)
        return tree
    elif len(dataTrain) == len(dataTrain.loc[dataTrain['Smoking'] == 'Not Smoking']):   # Stop Condition : All the data with the same answer of 'Not Smoking'.
        tree = DecisionTree('Not Smoking', '', cameFrom, ans, dataTrain, True)
        return tree
    elif len(dataTrain) == len(dataTrain.loc[dataTrain['Smoking'] == 'Smoking']):   # Stop Condition : All the data with the same answer of 'Smoking'.
        tree = DecisionTree('Smoking', '', cameFrom, ans, dataTrain, True)
        return tree
    elif len(attributes) == 0:  # Stop Condition - No more attributes.
        tree = getMajority(dataTrain, cameFrom, ans, dataTrain)
        return tree
    else:
        bestAttribute = getBestImportance(attributes, dataTrain)   # Return the attribute with the Minimum value.
        if bestAttribute != '':     # If not empty.
            tree = DecisionTree(bestAttribute, attributes[bestAttribute], cameFrom, ans, dataTrain, False)  # Creating new branch.
            for value in attributes[bestAttribute]:
                atr = copy.deepcopy(attributes)
                atr.pop(bestAttribute)      # Take off the relevant attribute.
                exs = dataTrain.loc[dataTrain[bestAttribute] == value]  # set the new example.
                subtree = decisionTreeLearning(exs, atr, tree, dataTrain, value)
                tree.children.append(subtree)   # Add that subtree as a children to his father.
        else:
            if len(dataTrain) == 0:
                tree = getMajority(parentex, cameFrom, ans, dataTrain)
                return tree
            elif len(dataTrain) == len(dataTrain.loc[dataTrain['Smoking'] == 'Not Smoking']):
                tree = DecisionTree('Not Smoking', '', cameFrom, ans, dataTrain, True)
                return tree
            elif len(dataTrain) == len(dataTrain.loc[dataTrain['Smoking'] == 'Smoking']):
                tree = DecisionTree('Smoking', '', cameFrom, ans, dataTrain, True)
                return tree
            else:
                tree = getMajority(dataTrain, cameFrom, ans, dataTrain)
                return tree
    return tree

def getMajority(exData, parent, ans, data):   # Find the majority of the data in the examples.
    smoking_count = exData['Smoking'].value_counts()['Smoking']
    not_smoking_count = exData['Smoking'].value_counts()['Not Smoking']
    if smoking_count >= not_smoking_count:
        return DecisionTree('Smoking', [], parent, ans, data, True)
    else:
        return DecisionTree('Not Smoking', [], parent, ans, data, True)

def getBestImportance(attributes, data):
    maxVal = 0
    entropyAttributesName = ''
    for key in attributes:      # Move on all the relevant attributes that left to check.
        if key != 'Smoking':
            Entropy = getEntropy(key, data)    # For every attribute we want the max value for minimize the total value.
            if Entropy > maxVal:
                maxVal = Entropy
                entropyAttributesName = key
    return entropyAttributesName

def getEntropy(attribute, data):   # Calculate the entropy per option for every attribute.
    values = data[attribute].unique()   # Collect al the option in the field of the specific attribute.
    totalEntropy = 0
    for option in values:
        tempData = data.loc[data[attribute] == option]
        countsFalse = len(tempData.loc[tempData['Smoking'] == 'Smoking'])
        countsTrue = len(tempData.loc[tempData['Smoking'] == 'Not Smoking'])
        totalOption = countsFalse+countsTrue
        pFalse = countsFalse/totalOption
        pTrue = countsTrue/totalOption
        entropy = calcTotalEntropy(pFalse, pTrue)
        totalEntropy = totalEntropy + (totalOption/len(data))*entropy
    return totalEntropy

def calcTotalEntropy(pFalse, pTrue):
    if pFalse == 0 and pTrue != 0:  # To ignore not valid step in math.
        return -pTrue*math.log(pTrue, 2)
    elif pTrue == 0 and pFalse != 0:
        return -pFalse*math.log(pFalse, 2)
    elif pTrue == 0 and pFalse == 0:
        return 0
    else:
        return -pFalse*math.log(pFalse, 2)-pTrue*math.log(pTrue, 2)

def is_busy(row_input):
    df = getData()
    attributes = getAttributes()
    fictive_Tree = DecisionTree('fictive', '', '', '', df, False)
    myTree = decisionTreeLearning(df, attributes, fictive_Tree, '', '')  # Call the recursive function.
    row_List = transformRowToList(row_input)
    ans = checkAns(myTree, row_List)
    if ans == 'Smoking':
        return 1
    else:
        return 0

def checkAns(tree, opt_List):
    if tree.ans == 'Smoking' or tree.ans == 'Not Smoking':
        return tree.ans
    elif tree.attribute == 'Smoking' or tree.attribute == 'Not Smoking':
        return tree.attribute
    else:
        for child in tree.children:
            for option in opt_List:
                if option[1] == child.ans and option[0] == child.attribute:
                    if child.ans == 'Smoking' or child.ans == 'Not Smoking':
                        return child.ans
                    elif child.attribute == 'Smoking' or child.attribute == 'Not Smoking':
                        return child.attribute
                    else:
                        return checkAns(child, opt_List)

def transformRowToList(row_input):    # Transform the "row_input" to a list of Features & Sub Features.
    TransformedList = []
    row_input.remove(row_input[0])
    # Gender - 0
    TransformedList.append(['Gender', row_input[0]])
    # Age - 1
    if row_input[1] <= 35:
        TransformedList.append(['Age', '20-35'])
    elif 35 < row_input[1] <= 50:
        TransformedList.append(['Age', '35-50'])
    elif 50 < row_input[1] <= 65:
        TransformedList.append(['Age', '50-65'])
    else:
        TransformedList.append(['Age', '70+'])
    # Height - 2
    if row_input[2] <= 165:
        TransformedList.append(['Height(cm)', '165-'])
    elif 165 < row_input[2] <= 175:
        TransformedList.append(['Height(cm)', '170-175'])
    else:
        TransformedList.append(['Height(cm)', '180+'])
    # Weight - 3
    if row_input[3] <= 55:
        TransformedList.append(['Weight(kg)', '55-'])
    elif 55 < row_input[3] <= 85:
        TransformedList.append(['Weight(kg)', '160-85'])
    else:
        TransformedList.append(['Weight(kg)', '90+'])
    # Waist - 4
    if row_input[4] <= 75:
        TransformedList.append(['Waist(cm)', '75-'])
    elif 75 < row_input[4] <= 100:
        TransformedList.append(['Waist(cm)', '80-100'])
    else:
        TransformedList.append(['Waist(cm)', '100+'])
    # Eyesight_L - 5
    if row_input[5] == 1:
        TransformedList.append(['Eyesight_L', 'Normal'])
    elif row_input[5] < 1:
        TransformedList.append(['Eyesight_L', 'Under Normal'])
    else:
        TransformedList.append(['Eyesight_L', 'Above Normal'])
    # Eyesight_R - 6
    if row_input[6] == 1:
        TransformedList.append(['Eyesight_R', 'Normal'])
    elif row_input[6] < 1:
        TransformedList.append(['Eyesight_R', 'Under Normal'])
    else:
        TransformedList.append(['Eyesight_R', 'Above Normal'])
    # Hearing_L - 7
    if row_input[7] == 1:
        TransformedList.append(['Hearing_L', 'Normal'])
    elif row_input[7] > 1:
        TransformedList.append(['Hearing_L', 'Not Normal'])
    # Hearing_R - 8
    if row_input[8] == 1:
        TransformedList.append(['Hearing_R', 'Normal'])
    elif row_input[8] > 1:
        TransformedList.append(['Hearing_R', 'Not Normal'])
    # Systolic - 9
    if row_input[9] <= 120:
        TransformedList.append(['Systolic', 'Optimal'])
    elif 120 < row_input[9] <= 129:
        TransformedList.append(['Systolic', 'Normal'])
    elif 129 < row_input[9] <= 139:
        TransformedList.append(['Systolic', 'Normal High'])
    else:
        TransformedList.append(['Systolic', 'High'])
    # Relaxation - 10
    if row_input[10] <= 80:
        TransformedList.append(['Relaxation', 'Optimal'])
    elif 80 < row_input[10] <= 84:
        TransformedList.append(['Relaxation', 'Normal'])
    elif 84 < row_input[10] <= 89:
        TransformedList.append(['Relaxation', 'Normal High'])
    else:
        TransformedList.append(['Relaxation', 'High'])
    # FBS - 11
    if row_input[11] <= 100:
        TransformedList.append(['FBS', 'Normal'])
    elif 100 < row_input[11] <= 125:
        TransformedList.append(['FBS', 'PreDiabetes'])
    else:
        TransformedList.append(['FBS', 'Diabetes'])
    # Cholesterol - 12
    if row_input[12] <= 200:
        TransformedList.append(['Cholesterol', 'Desirable'])
    elif 200 < row_input[12] <= 239:
        TransformedList.append(['Cholesterol', 'Borderline High'])
    else:
        TransformedList.append(['Cholesterol', 'High'])
    # Triglyceride - 13
    if row_input[13] <= 150:
        TransformedList.append(['Triglyceride', 'Desirable'])
    elif 150 < row_input[13] <= 199:
        TransformedList.append(['Triglyceride', 'Borderline High'])
    else:
        TransformedList.append(['Triglyceride', 'High'])
    # HDL - 14
    if row_input[14] <= 100:
        TransformedList.append(['HDL', 'Optimal'])
    elif 100 < row_input[14] <= 139:
        TransformedList.append(['HDL', 'Above Optimal'])
    else:
        TransformedList.append(['HDL', 'High'])
    # LDL - 15
    if row_input[15] <= 40:
        TransformedList.append(['LDL', 'High'])
    elif 40 < row_input[15] <= 60:
        TransformedList.append(['LDL', 'Above Optimal'])
    else:
        TransformedList.append(['LDL', 'Optimal'])
    # Hemoglobin - 16
    if row_input[16] <= 11:
        TransformedList.append(['Hemoglobin', 'Significantly Low'])
    elif 11 < row_input[16] <= 13.5:
        TransformedList.append(['Hemoglobin', 'Slightly Low'])
    else:
        TransformedList.append(['Hemoglobin', 'Normal'])
    # UrineProtein - 17
    if row_input[17] <= 11:
        TransformedList.append(['UrineProtein', 'Normal'])
    elif 11 < row_input[17] <= 13.5:
        TransformedList.append(['UrineProtein', 'Slightly Elevated'])
    else:
        TransformedList.append(['UrineProtein', 'Significantly Elevated'])
    # SerumCreatinine - 18
    if row_input[18] <= 1.2:
        TransformedList.append(['SerumCreatinine', 'Normal'])
    elif 1.2 < row_input[18] <= 1.9:
        TransformedList.append(['SerumCreatinine', 'Slightly Elevated'])
    else:
        TransformedList.append(['SerumCreatinine', 'Significantly Elevated'])
    # AST - 19
    if row_input[19] <= 40:
        TransformedList.append(['AST', 'Normal'])
    elif 40 < row_input[19] <= 80:
        TransformedList.append(['AST', 'Slightly Elevated'])
    else:
        TransformedList.append(['AST', 'Significantly Elevated'])
    # ALT - 20
    if row_input[20] <= 56:
        TransformedList.append(['ALT', 'Normal'])
    elif 56 < row_input[20] <= 125:
        TransformedList.append(['ALT', 'Slightly Elevated'])
    else:
        TransformedList.append(['ALT', 'Significantly Elevated'])
    # GTP - 21
    if row_input[21] <= 70:
        TransformedList.append(['GTP', 'Normal'])
    elif 70 < row_input[21] <= 190:
        TransformedList.append(['GTP', 'Slightly Elevated'])
    else:
        TransformedList.append(['GTP', 'Significantly Elevated'])
    # DentalCaries - 22
    if row_input[22] == 1:
        TransformedList.append(['DentalCaries', 'Y'])
    else:
        TransformedList.append(['DentalCaries', 'N'])
    # Tartar - 23
    TransformedList.append(['Tartar', row_input[23]])

    return TransformedList

def tree_error(k):
    # k = Number of Cross Validations.
    df = getData()    # Function that return normal dataframe.
    attributes = getAttributes()    # Function that return a dictionary of the attributes.
    total_Errors = 0    # Error Counter.
    fictive_Tree = DecisionTree('fictive', '', '', '', df, False)
    temp = np.array_split(df, k)
    test = pd.DataFrame([])
    train = None
    counter = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                test = temp[j]
            elif counter == 0:
                train = temp[j]
                counter = counter + 1
            else:
                train = pd.concat([train, temp[j]], axis=0)
        myTree = decisionTreeLearning(train, attributes, fictive_Tree, '', '')
        total_Errors = total_Errors + calculateError(myTree, test)
        print("Calculating K = ", i+1, " is Done.")
    print('The Average Error Rate Is: ' + str(round((total_Errors/k)*100, 2)) + "% .")


# --------------------- Main ---------------------

