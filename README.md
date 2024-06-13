
#######################################################################################################################
# this version of the gman will be trained on the masked data
# the train dataset is made with two types of data, one with missing values and the other with imputed values
# the historical data contains missing values filled with 0 or -1 hence masking the data
# the prediction data contains the imputed values - giving us as close as possible to the real data
# the model therefore will hopefully learn to predict the imputed values from the masked data


