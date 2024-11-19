import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as data:
        content = data.read().splitlines()
        content.pop(0)
        evidence = list()
        label = list()
        for row in content:
            evidence_item = list()
            columns = row.split(',') # verificar se o metodo split retorna uma lista ou um array. É preciso que seja um array
            evidence_item.append(int(columns[0])) # Administrative
            evidence_item.append(float(columns[1])) # Administrative_Duration
            evidence_item.append(int(columns[2])) # Informational
            evidence_item.append(float(columns[3])) # Informational_Duration
            evidence_item.append(int(columns[4])) # ProductRelated 
            evidence_item.append(float(columns[5])) # ProductRelated_Duration
            evidence_item.append(float(columns[6])) # BounceRates
            evidence_item.append(float(columns[7])) # ExitRates
            evidence_item.append(float(columns[8])) # PageValues
            evidence_item.append(float(columns[9])) # SpecialDay
            evidence_item.append(convertMonth(columns[10])) # Month
            evidence_item.append(int(columns[11])) # OperatingSystems
            evidence_item.append(int(columns[12])) # Browser
            evidence_item.append(int(columns[13])) # Region
            evidence_item.append(int(columns[14])) # TrafficType
            evidence_item.append(convertVisitorType(columns[15])) # VisitorType
            evidence_item.append(convertWeekend(columns[16])) # Weekend 
            evidence.append(evidence_item)
            label.append(convertRevenue(columns[17])) # Revenue
    return (evidence, label)
            

def convertMonth(month_str):
    import calendar

    month_str = month_str.strip().capitalize()  # Normalize input
    month_mapping = {month: idx for idx, month in enumerate(calendar.month_name) if month}
    abbrev_mapping = {month[:3]: idx for month, idx in month_mapping.items()}
    
    # Combine full and abbreviated mappings
    month_mapping.update(abbrev_mapping)

    return month_mapping.get(month_str) - 1


def convertVisitorType(visitorType):
    if visitorType == 'Returning_Visitor':
        return 1
    return 0

def convertWeekend(weekend):
    if weekend == 'FALSE':
        return 0
    return 1

def convertRevenue(revenue):
    if revenue == 'FALSE':
        return 0
    return 1

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    totalPurchaseLabel = 0
    totalNotPurchaseLabel = 0
    specificity = 0

    for label in labels:
        if label == 1:
            totalPurchaseLabel += 1
        else:
            totalNotPurchaseLabel += 1 

    for index in range(len(labels)):
        if labels[index] == 1 == predictions[index]:
            sensitivity += 1
        elif labels[index] == 0 == predictions[index]:
            specificity += 1
    
    return (float(sensitivity / totalPurchaseLabel), float(specificity / totalNotPurchaseLabel))

if __name__ == "__main__":
    main()
