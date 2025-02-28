import numpy as np
import pandas as pd



def calculate_entropy(feature):
    entropy = 0
    element, count = np.unique(feature, return_counts=True)
    for i in range(len(element)):
        prob = count[i]/np.sum(count)
        entropy -= prob*np.log2(prob)
    return entropy
def information_gain(data, split_feature, root_feature):
    E_S = calculate_entropy(data[root_feature])
    average_information = 0
    attributes, count = np.unique(data[split_feature], return_counts=True)
    for i in range(len(attributes)):
        split_data = data.where(data[split_feature] == attributes[i]).dropna()[root_feature]
        average_information += (count[i]/np.sum(count))*calculate_entropy(split_data)
    information_gain = E_S - average_information
    return information_gain
def split_dataset(data, feature, param):
    holder = data.where(data[feature] == param).dropna()

def ID3(data, features, target_name):
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]
    else:
        #Identify which feature to use for splitting
        feature_info_gain = [information_gain(data,feature,target_name) for feature in features]
        best_feature = features[np.argmax(feature_info_gain)]
        tree = {best_feature:{}}
        
        #create a new feature list
        feats = [i for i in features if i != best_feature]
        for param in np.unique(data[best_feature]):
            # Build a subdata splitting them based on the categories of the best feature
            subdata = split_dataset(data, best_feature, param)
            # Create sub-trees for that feature
            branch = ID3(subdata, feats, target_name)
            # Add the branch to the tree
            tree[best_feature][param] = branch
        return tree
    
def traverse(value,spec_list):
    global columns
    global combination_list
    for key,value in value.items():
        if isinstance(value,dict):
            if key in columns:
                spec_list.append(key)
                traverse(value,spec_list)
                spec_list.pop()
            else:
                spec_key=spec_list[-1]
                spec_list[-1]={spec_key:key}
                traverse(value,spec_list)
                spec_list[-1]=spec_key
        else:
            spec_key=spec_list[-1]
            spec_list[-1]={spec_key:key}
            combination_list.append((spec_list.copy(),value))
            spec_list[-1]=spec_key
    return combination_list
    

if __name__=='__main__':
    dataset=pd.read("/path/to/dataset")
    cat_cols=[]#put the features here
    determiner=None#put the target varible:mapped to 0 and 1 
    features=dataset[cat_cols]
    dtree = ID3(dataset, features)
    columns=cat_cols
    combinations_list=[]
    combi_list=traverse(dtree,[])