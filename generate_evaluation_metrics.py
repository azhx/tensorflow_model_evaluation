import numpy as np
import pandas as pd
import pickle
from evaluate_test import evaluate_test_set
from matplotlib import pyplot as plt


def flatten_ypreds(unified_data):
    ypreds = {'classes':[], 'boxes':[]}
    for i, each in enumerate(unified_data['filename']):
        if unified_data.at[i, 'truenpreds'] > unified_data.at[i, 'npreds']:
            # Accounting for the case when a greater number of classes is predicted compared to the ground truth
            if unified_data.at[i, 'npreds'] != 0:
                for j in range(unified_data.at[i, 'npreds']):
                    ypreds['classes'].append(unified_data.at[i, 'class'][j])
                    ypreds['boxes'].append([unified_data.at[i, 'xmin'][j],
                                            unified_data.at[i, 'ymin'][j],
                                            unified_data.at[i, 'xmax'][j],
                                            unified_data.at[i, 'ymax'][j]])

            for j in range(unified_data.at[i, 'truenpreds']-unified_data.at[i, 'npreds']):
                ypreds['classes'].append('_none')
                ypreds['boxes'].append([])
        else:
            for j in range(unified_data.at[i, 'truenpreds']):
                ypreds['classes'].append(unified_data.at[i, 'class'][j])
                ypreds['boxes'].append([unified_data.at[i, 'xmin'][j],
                                        unified_data.at[i, 'ymin'][j],
                                        unified_data.at[i, 'xmax'][j],
                                        unified_data.at[i, 'ymax'][j]])
    # This method of flattening preds is defensible because within each [i, 'class'], predicted classes are ordered by score.
    return ypreds


def flatten_ytrue(unified_data):
    ytrue = {'classes':[], 'boxes':[]}
    for i, each in enumerate(unified_data['trueclass']):
        if len(each) > 1:
            for j in range(len(each)):
                ytrue['classes'].append(each[j])
                ytrue['boxes'].append([unified_data.at[i, 'truexmin'][j],
                                        unified_data.at[i, 'trueymin'][j],
                                        unified_data.at[i, 'truexmax'][j],
                                        unified_data.at[i, 'trueymax'][j]])
        else:
            ytrue['classes'].append(each[0])
            ytrue['boxes'].append([unified_data.at[i, 'truexmin'][0],
                                    unified_data.at[i, 'trueymin'][0],
                                    unified_data.at[i, 'truexmax'][0],
                                    unified_data.at[i, 'trueymax'][0]])
    return ytrue


def generate_df_confusion(ytrue, ypreds, category_index):
    #add one of every label allow a column for each label to be created
    for k, v in category_index.items():
        ytrue['classes'].append(v['name'])
        ypreds['classes'].append(v['name'])
    ytrueclasses = pd.Series(ytrue['classes'], name = 'Actual')
    ypredsclasses = pd.Series(ypreds['classes'], name = 'Predicted')
    df_confusion = pd.crosstab(ytrueclasses, ypredsclasses, rownames=['Actual'], colnames=['Predicted'])
    #remove the artificially added data
    for index, row in df_confusion.iterrows():
        if row.name == index:
            df_confusion.at[row.name, index] -= 1
    return df_confusion


def calculate_accuracy(df_confusion):

    classaccuracies = {}
    for i, v in df_confusion.iterrows():
        classaccuracies[i] = df_confusion.at[i, i]/ np.sum(v)

    #number of correctly predicted over total number of labels (_none is not on the diagonal)
    accuracy = np.sum(np.diag(df_confusion))/ np.sum(np.sum(df_confusion))
    accuracy = f"{accuracy*100:.3f}%"
    return accuracy, classaccuracies


def calculate_f1scores(df_confusion):
    '''
    https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
    https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc
    '''
    f1scores = {}
    for index, row in df_confusion.iterrows():
        if row.name == index:
            tp = df_confusion.at[row.name, index]
            fp = np.sum(df_confusion[index])-tp
            fn = np.sum(row)-tp
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            fscore = 2*(precision*recall)/(precision+recall)
            f1scores[row.name] = fscore #there may nans caused by divisions by 0. This is because tp/tp/fp can be 0
    return f1scores


def calculate_iou(ytrue, ypreds, df_confusion):
    '''
    ypreds['boxes'] lists are in the order xmin, ymin, xmax, ymax

    For each ground truth and predict box:
    Takes max of xmin, min of xmax, max of ymin, min of ymax to compute area
    of intersection

    compute union of truth and predict box by summing individual areas and
    subtracting their intersection

    iou = intersection / union

    Note: the actual contents of the boxes are not accounted for in these
    calculations. Ious are simply grouped by the species' that appear in the
    images in classious.

    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    '''
    classious = {}
    for i, v in enumerate(df_confusion.columns.to_list()[:-1]):
        classious[v] = {'n': 0, 'iousums': 0}

    for i, v in enumerate(ytrue['boxes']):

        # determine the (x, y)-coordinates of the intersection rectangle
        boxA = ytrue['boxes'][i]
        boxB = ypreds['boxes'][i]

        if boxB == []:
            classious[ytrue['classes'][i]]['n'] += 1
            continue

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area

        classious[ytrue['classes'][i]]['n'] += 1
        classious[ytrue['classes'][i]]['iousums'] += interArea / float(boxAArea + boxBArea - interArea)

    for k, v in classious.items():
        classious[k] = classious[k]['iousums'] / classious[k]['n']

    return classious


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=90, fontsize = 5)
    plt.yticks(tick_marks, df_confusion.index, fontsize = 5)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name, labelpad = -300)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 200
    fig_size[1] = 300
    plt.rcParams["figure.figsize"] = fig_size
    plt.savefig('confusion_matrix.png', dpi = 600, bbox_inches= 'tight')


def output_everything(accuracy, classaccuracies, f1scores, classious, df_confusion):
    plot_confusion_matrix(df_confusion) #saves 'confusion_matrix.png'

    out = {'class': [],
           'class accuracy': [],
           'class f1score': [],
           'class iou': [],
           'overall accuracy': [],
           'average of valid f1scores': [],
           'average iou': []}
    validfscores = []
    for i, v in enumerate(df_confusion.columns.to_list()[:-1]):
        out['class'].append(v)
        out['class accuracy'].append(classaccuracies[v])
        out['class f1score'].append(f1scores[v])
        if (np.isnan(f1scores[v]) == False):
            validfscores.append(f1scores[v])
        out['class iou'].append(classious[v])
        out['overall accuracy'].append('')
        out['average of valid f1scores'].append('')
        out['average iou'].append('')
    out['overall accuracy'][0] = accuracy
    out['average of valid f1scores'][0] = np.mean(validfscores)
    out['average iou'][0] = np.mean(out['class iou'])
    outcsv = pd.DataFrame(data = out)
    outcsv.to_csv('evaluation_metrics.csv', index = False) #saves to csv

    return



if __name__ == '__main__':

    #MODE = 1  # Evaluate full test set and calculate metrics
    MODE = 2 # Calculate metrics only from unifieddata.p and category_index.p files

    if (MODE == 1):
        #from evaluate_test.py, predict on all images of test set
        evaluate_test_set()

    #load in necessary data
    unified_data = pickle.load(open('unifieddata.p', 'rb'))
    category_index = pickle.load(open('category_index.p', 'rb'))
    print ('successfully loaded unified_data and category_index')
    #flatten predictions (some dimensions may be mismatched)
    ypreds, ytrue = flatten_ypreds(unified_data), flatten_ytrue(unified_data)
    print ('flattened predictions')
    #calculate metrics
    df_confusion = generate_df_confusion(ytrue, ypreds, category_index)
    print ('generated confusion matrix')
    accuracy, classaccuracies = calculate_accuracy(df_confusion)
    print ('calculated accuracy')
    f1scores = calculate_f1scores(df_confusion)
    print ('calculated f1scores')
    classious = calculate_iou(ytrue, ypreds, df_confusion)
    print ('calculated ious')

    #output and save everything
    output_everything(accuracy, classaccuracies, f1scores, classious, df_confusion)
    print ('success')
