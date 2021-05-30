import pandas as pd

from pandas import DataFrame

def disagreement(data: DataFrame) -> DataFrame:
    '''Takes in a DataFrame containing forecasts of different predictors and
       calculates the disagreement score of the overall system.
    
        Parameters:
            data (DataFrame): individual predictors forecast output
        
        Returns:
            (DataFrame): containing overall system disagrement scores
    '''
    system_disagreement = []
    for k in range(data.shape[0]):
        individual_scores = []
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                individual_scores.append(abs(data.iloc[k,i] - data.iloc[k,j]))
            
        system_disagreement.append(sum(individual_scores) / len(individual_scores))
        individual_scores.clear()
    
    output = pd.DataFrame()
    output['System Disagreement'] = system_disagreement
    return output



def predictor_score(data: DataFrame) -> DataFrame:
    '''Takes in a DataFrame and calculates each individual predictors disagreement
       scores.
    
        Parameters:
            data (DataFrame): individual predictors forecast output
        
        Returns:
            (DataFrame): containing all predictors individual 
    '''
    individual_score_collection = []
    for k in range(data.shape[0]):
        average_values = []
        for j in range(data.shape[1]):
            individual_scores = []
            for i in range(data.shape[1]):
                individual_scores.append(abs(data.iloc[k, j] - data.iloc[k, i]))
        
            average_values.append(sum(individual_scores) / len(individual_scores))
            individual_scores.clear()
            
        individual_score_collection.append(average_values)

    result = pd.DataFrame(individual_score_collection) 
    result.columns = data.columns
    result = result.add_suffix(' disagreement score')


    return result


def formatting(target: list) -> list:
    '''Helper function to transform a list containing additional, unnecessary dataframe details into a pure list
       containing only target values.
        
        Parameters:
            target (list): list containing unnecessary additional information
         
        Returns:
            (list): list containing target values
    '''
    for i in range(len(target)):
        try:
            target[i] = target[i][0]
        except:
            target[i] = target[i]
    
    return target


def new_weights(preds: list, real_value: float) -> list:
    '''Helper function to calculated new weights, depending on t-1 forecast errors of predictors.
    
        Parameters:
            preds (list): t-1 predictions of each predictor
            real_value (float): real value at t
          
        Returns:
            (list): list containing the new weight values for each predictor
    '''
    if type(preds) != type(list):
        preds = list(preds)
        
    individual_error = []
    new_weights = []
    final_weights = []
    
    for i in range(len(preds)):
        individual_error.append(abs(preds[i] - real_value))
    
    total_error = sum(individual_error)
    for j in range(len(individual_error)):
        try:
            if sum(total_error) == 0:
                new_weights.append(1)
            else:
                new_weights.append(1-(individual_error[j]/total_error))
        except:
            if total_error == 0:
                new_weights.append(1)
            else:
                new_weights.append(1-(individual_error[j]/total_error))
        
    for k in range(len(new_weights)):
        final_weights.append((new_weights[k]/sum(new_weights)) * len(preds))
    
    return formatting(final_weights)


def new_weights_focused(preds: list, real_value: float) -> list:
    '''Helper function to calculated new weights, depending on t-1 forecast errors of predictors. Weights can only be 1 or 0.
    
        Parameters:
            preds (list): t-1 predictions of each predictor
            real_value (float): real value at t
         
        Returns:
            (list): list containing the new weight values for each predictor
    '''
    if type(preds) != type(list):
        preds = list(preds)
        
    individual_error = []
    new_weights = []
    final_weights = []
    
    for i in range(len(preds)):
        individual_error.append(abs(preds[i] - real_value))
    
    total_error = sum(individual_error)
    for j in range(len(individual_error)):
        if sum([total_error]) == 0: # new approach, substitutes try except clauses. Needs testing
            new_weights.append(1)
        else:
            new_weights.append(1-(individual_error[j]/total_error))

    for k in range(len(new_weights)):
        if new_weights[k] == max(new_weights):
            final_weights.append(1)
        else:
            final_weights.append(0)
    
    return formatting(final_weights)


def consolidated_predictions(data: DataFrame, real: DataFrame) -> list:
    '''Function to calculate the consolidated prediction value of all individual predictors.
    
        Parameters:
            data (DataFrame): predictions values from each individual predictor
            real (DataFrame): actual values
         
        Returns:
            (list): list containing consolidated prediction value considering new weight assignments for each predictor
    '''
    final_predictions = []
    weight_history = []
    weights = [1] * data.shape[1]

    for j in range(data.shape[0]):
        temp = []
        for i in range(data.shape[1]):
            temp.append(data.iloc[j, i]*weights[i])
            
        final_predictions.append(sum(temp)/data.shape[1])
        weight_history.append(weights)
        weights = new_weights(data.iloc[j], real.iloc[j])

    
    return final_predictions


def consolidated_predictions_memory(data: DataFrame, real: DataFrame) -> list:
    '''Function to calculate the consolidated prediction value of all individual predictors. This function furthermore
       extends consolidated_predictions by keeping a memory of prior assigned weights. An average of all prior assigned
       weights is calculated and applied to calculate the final consolidation value.
    
        Parameters:
            data (DataFrame): predictions values from each individual predictor
            real (DataFrame): actual value
         
        Returns:
            (list): list containing consolidated prediction value considering new weight assignments for each predictor
    '''
    final_predictions = []
    
    initialize = [1] * data.shape[1]
    weight_history = [initialize]
    weights = []

    for j in range(data.shape[0]):
        temp = []
        for i in range(data.shape[1]):
            temp.append(data.iloc[j, i]*([sum(z) for z in zip(*weight_history)][i]/(j+1))) # j number of rows, total value to take average
        
        final_predictions.append(sum(temp)/data.shape[1])
        weights = new_weights(data.iloc[j], real.iloc[j])
        weight_history.append(weights)
        

    
    return final_predictions

def consolidated_predictions_focused(data: DataFrame, real: DataFrame) -> list:
    '''Function to calculate the consolidated prediction value of all individual predictors.
       Takes the sole estimate of the individual predictor that best predicted in the past.
    
        Parameters:
            data (DataFrame): predictions values from each individual predictor
            real (DataFrame): actual value
         
        Returns:
            (list): list containing consolidated prediction value considering new weight assignments for each predictor
    '''
    final_predictions = []
    weight_history = []
    weights = [1] * data.shape[1]

    for j in range(data.shape[0]):
        temp = []
        for i in range(data.shape[1]):
            temp.append(data.iloc[j, i]*weights[i])
        final_predictions.append(sum(temp)/sum(weights))
        weight_history.append(weights)
        weights = new_weights_focused(data.iloc[j], real.iloc[j][0])

    
    return final_predictions




def consolidated_predictions_anchor(data: DataFrame, real: DataFrame, anchor: int) -> list:
    '''Function to calculate the consolidated prediction value of all individual predictors. To prevent the
       algorithm from being limited to produce consolidation values within the min and max value predicted by
       the individual predictors, min and max anchors are launched that extend above the biggest and smallest value
       estimated.
    
        Parameters:
            data (DataFrame): predictions values from each individual predictor
            real (DataFrame): actual value
            anchor (int): how far should max, min prediction be extended
         
        Returns:
            (list): list containing consolidated prediction value considering new weight assignments for each predictor
    '''
    if anchor <= 1:
        raise ValueError('Anchors need to be set at least > 1')
        
    final_predictions = []
    weight_history = []
    
    weights = [1] * data.shape[1]
    weights.append(1)
    weights.append(1)

    for j in range(data.shape[0]):
        data['Max Anchor'] = anchor * max(data.iloc[j])
        data['Min Anchor'] = (1- (anchor - 1)) * min(data.iloc[j])
        temp = []
        for i in range(data.shape[1]):
            temp.append(data.iloc[j, i]*weights[i])
            
        final_predictions.append(sum(temp)/data.shape[1])
        weight_history.append(weights)
        weights = new_weights(data.iloc[j], real.iloc[j])
        del data['Max Anchor']
        del data['Min Anchor']

    
    return final_predictions

def average_consolidation(data: DataFrame) -> list:
    '''Function to calculate simple average of all predictor forecasts.
    
        Parameters:
            data (DataFrame): prediction values from each individual predictor
         
        Returns:
            (list): list containing average values of predictor forecasts
    '''
    result = []
    for i in range(data.shape[0]):
        result.append(sum(data.iloc[i])/data.shape[1])
    
    return result
