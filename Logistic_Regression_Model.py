# Script Name: Logistic_Regression
# Author: Joshua (Jay) Wimhurst
# Date created: 8/23/2022
# Date Last Edited: 2/24/2023

############################### DESCRIPTION ###################################
# This script applies a logistic regression model to the predictors and wind
# farm locations across the CONUS (and individual states) in order to
# construct an equation-based relationship between them. The outputs of this
# model are various statistics (e.g., log-likelihood ratio) and charts
# (e.g., Receiver Operation Characteristic curve) and a map showing likelihood
# of wind farm existence and predicted versus actual wind farm occurrence.
# Four different versions of the model can be executed: all predictors,
# all predictors but wind speed, wind speed only, and a refined predictor set.
###############################################################################

def LogisticRegressionModel():
        
    import sys
    import os
    import arcpy
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from arcpy.da import TableToNumPyArray, UpdateCursor, SearchCursor
    from fpdf import FPDF
    from pandas import DataFrame, concat
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib import transforms
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, accuracy_score
    from statsmodels.api import OLS
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from math import e
    from scipy.stats import chi2, rankdata, mannwhitneyu
    from statistics import median
    from warnings import filterwarnings
    filterwarnings("ignore")
    
    ######################### DIRECTORY CONSTRUCTION ##########################
    
    # IMPORTANT - set up a file directory to place the Console Output, 
    # generated figures, and constructed WiFSS surfaces BEFORE attempting to
    # run this model
    directory = # Insert directory here
    
    # IMPORTANT - make sure to also create a folder for the figures 
    # created from the model's calibration and validation results
    directoryPlusFigures = directory + "\Figures"
    
    # IMPORTANT - make sure to also create a folder for the WiFSS surfaces
    # created from applying the trained and tested model to all grid cells
    directoryPlusSurfaces = directory + "\WiFSS_Surfaces"
    
    # IMPORTANT - make sure to also create a folder for the coefficients
    # obtained from fitting the model, which will be needed for users running
    # the cellular automaton portion of the model
    directoryPlusCoefficients = directory + "\Coefficients"
    
    # The script prints the console output to a text file, with a previous
    # version deleted prior to the model run
    if os.path.exists(directory + "\Console_Output.txt"):
        os.remove(directory + "\Console_Output.txt")
    
    ###################### DATASET SELECTION AND SETUP ############################
        
    # PDF file containing the console output is initiated and caveat is added
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(4, 4)
    pdf.set_font(family = 'arial', size = 13.0)
    pdf.multi_cell(w=0, h=5.0, align='R', txt="Console output", border = 0)
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="------------------ DATASET SELECTION AND SETUP ------------------"
                      +'\n\n'+ "NOTE: The desired study region must be specified as 'CONUS' if one "
                      +'\n'+ "wishes to execute the logistic regression model over states that "
                      +'\n'+ "contain zero commercial wind farms (Louisiana, Mississippi "
                      +'\n'+ "Alabama, Georgia, South Carolina, Kentucky), states that possess "
                      +'\n'+ "wind farms in only one grid cell at all but the highest spatial "
                      +'\n'+ "resolutions (Arkansas, Florida, Virginia, Delaware, Connecticut,"
                      +'\n'+ "New Jersey, Tennessee), or states at low spatial resolutions at which "
                      +'\n'+ "too many predictors were removed due to collinearity (Rhode Island "
                      +'\n'+ "at the 100th or 80th percentile).\n", border=0)
    
    # Select the study region for which the model will run:
    # The CONUS or a single state
    def studyRegion(values, message):
        while True:
            x = input(message) 
            if x in values:
                studyRegion.region = x
                break
            else:
                print("Invalid value; options are " + str(values))
    studyRegion(["CONUS","Arizona","California","Colorado","Idaho","Illinois",
                 "Indiana","Iowa","Kansas","Maine","Maryland","Massachusetts",
                 "Michigan","Minnesota","Missouri","Montana","Nebraska",
                 "Nevada","New_Hampshire","New_Mexico","New_Jersey","New_York",
                 "North_Carolina","North_Dakota","Ohio","Oklahoma","Oregon",
                 "Pennsylvania","Rhode_Island","South_Dakota","Texas","Utah",
                 "Vermont","Washington","West_Virginia","Wisconsin","Wyoming"], 
                 '''Enter desired study region \n(CONUS, Arizona, California, Colorado, '''
                 '''Idaho, Illinois, Indiana, Iowa, Kansas, Maine, Maryland, '''
                 '''Massachusetts, Michigan, Minnesota, Missouri, Montana, '''
                 '''Nebraska, Nevada, New_Hampshire, New_Mexico, New_York, '''
                 '''North_Carolina, North_Dakota, Ohio, Oklahoma, Oregon, '''
                 '''Pennsylvania, Rhode_Island, South_Dakota, Texas, '''
                 '''Utah, Vermont, Washington, West_Virginia, Wisconsin, '''
                 '''Wyoming):\n''')
    
    # User input for wind farm density in acres per Megawatt: 25, 45, 65, or 85
    def farmDensity(values,message):
        while True:
            x = input(message) 
            if x in values:
                farmDensity.density = x
                break
            else:
                print("Invalid value; options are " + str(values))
    farmDensity(["25","45","65","85"], "\nEnter desired wind farm density (25, 45, 65, or 85 acres/MW):\n")

    # User input for wind power capacity as a percentile: 20, 40, 60, 80, or 100
    def farmCapacity(values,message):
        while True:
            x = input(message) 
            if x in values:
                farmCapacity.capacity = x
                break
            else:
                print("Invalid value; options are " + str(values))
    farmCapacity(["20","40","60","80","100"], "\nEnter desired wind power capacity (20, 40, 60, 80, or 100 percentile):\n")
    # NOTE: 20th percentile = 30MW, 40th percentile = 90MW, 60th percentile = 150 MW
    # 80th percentile = 201.5 MW, 20th percentile = 525 MW
    
    # Based on selected percentile, the print-out to the text file changes
    if farmCapacity.capacity == "100":
        power = "525 MW"
    elif farmCapacity.capacity == "80":
        power = "202 MW"
    elif farmCapacity.capacity == "60":
        power = "150 MW"
    elif farmCapacity.capacity == "40":
        power = "90 MW"
    elif farmCapacity.capacity == "20":
        power = "30 MW"

    # The user is asked for the predictor configurations for which they wish
    # to run the model, first the Full configuration
    def fullConfiguration(values,message):
        while True:
            x = input(message)
            if x in values:
                fullConfiguration.YesNo = x
                break
            else:
                print("Invalid value; options are " + str(values))
    fullConfiguration(["Y","N"], "".join(["\nDo you wish to use the Full predictor configuration? Y or N.\n"]))
    
    # Next the No_Wind configuration
    def noWindConfiguration(values,message):
        while True:
            x = input(message)
            if x in values:
                noWindConfiguration.YesNo = x
                break
            else:
                print("Invalid value; options are " + str(values))
    noWindConfiguration(["Y","N"], "".join(["\nDo you wish to use the No_Wind predictor configuration? Y or N.\n"]))
    
    # Next the Wind_Only configuration
    def windOnlyConfiguration(values,message):
        while True:
            x = input(message)
            if x in values:
                windOnlyConfiguration.YesNo = x
                break
            else:
                print("Invalid value; options are " + str(values))
    windOnlyConfiguration(["Y","N"], "".join(["\nDo you wish to use the Wind_Only predictor configuration? Y or N.\n"]))
    
    # Finally the Reduced configuration
    def reducedConfiguration(values,message):
        while True:
            x = input(message)
            if x in values:
                reducedConfiguration.YesNo = x
                break
            else:
                print("Invalid value; options are " + str(values))
    reducedConfiguration(["Y","N"], "".join(["\nDo you wish to use the Reduced predictor configuration? Y or N.\n"]))
    
    configList = []
    
    if fullConfiguration.YesNo == "Y":
        configList.append("Full")
    if noWindConfiguration.YesNo == "Y":
        configList.append("No_Wind")
    if windOnlyConfiguration.YesNo == "Y":
        configList.append("Wind_Only")
    if reducedConfiguration.YesNo == "Y":
        configList.append("Reduced")
    
    # User inputs are added to the console output
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\nSpecified study region: " + str(studyRegion.region)
                      +'\n'+"Specified wind farm density: " + str(farmDensity.density) + " acres/MW"
                      +'\n'+"Specified wind power capacity: " + str(farmCapacity.capacity)  + "th percentile (" + power + ")"
                      +'\n'+"Predictor configurations specified by the user: " + str(configList), border=0)

    # The filepaths to the desired gridded datasets depend on whether the
    # user selected the CONUS or an individual state 
    if studyRegion.region != "CONUS":        
        # File path to the dataset specified by user input
        table = "".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_Merged.gdb\Attribute_Table"])
    else:        
        table = "".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_Merged.gdb\Attribute_Table"])

    # The aggregated dataset are redefined to exclude the attribute table
    # rows with missing data
    array = TableToNumPyArray(table, "*", skip_nulls = True)
    df = DataFrame(array)

    # The model's ROC is more likely to fail if there are too few grid cells,
    # and too many predictors may also be removed due to collinearity. A user
    # input is created to allow the user the chance to abort the code if they
    # choose an inappropriate resolution.    
    if len(df) <= 300:
        print('''\nThis resolution comprises''', str(len(df)), '''grid cells, '''
              '''a resolution at which the model may be reduced to a small number '''
              '''of predictors due to multicollinearity, and at which the model's '''
              '''Receiver Operating Characteristic may be affected. Do you wish '''
              '''to proceed?''')
        
        # Specify using Y or N whether the user would like to choose a different
        # resolution
        def proceed(values, message):
            while True:
                x = input(message) 
                if x in values:
                    proceed.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        proceed(["Y", "N"], "Y or N:\n")
        
        # The code stops if the user specifies N
        if proceed.YesOrNo != "Y":
            sys.exit()
        # The warning about compromised model performance is written to the
        # console output if the end-user decides to still go ahead.
        else:
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nAt the user-specified resolution, less than 300 (" + str(len(df)) + ") grid cells exist over"
                              +'\n'+ str(studyRegion.region) + ", which increases the risk of fewer predictors being "
                              +'\n'+"retained due to multicollinearity, and of a compromised ROC.", border=0)
                            
    # The predictors that the model uses depend on the study region selected
    # by the user
    if studyRegion.region != "CONUS":
        # The predictors from the dataframe are isolated as a separate dataframe.
        # NOTE: The model's ability to predict wind farm locations cannot be 
        # assessed over the following states: Alabama, Arkansas, Delaware, DC,
        # Florida, Georgia, Kentucky, Louisiana, Mississippi, New Jersey,
        # South Carolina, or Virginia (all are states with 0 or 1 commercial 
        # wind farm).
        df = df[["Wind_Turb","Critical","Historical","Military","Mining",
                 "Nat_Parks","Dens_15_19","Trib_Land","Wild_Refug",
                 "Avg_Elevat","Avg_Temp","Avg_Wind","Bat_Count","Bird_Count",
                 "Prop_Rugg","Undev_Land","Near_Air","Near_Hosp",
                 "Near_Roads","Near_Trans","Near_Sch","Plant_Year",
                 "Farm_Year","ISO_YN","Near_Plant","Dem_Wins","Type_15_19",
                 "Unem_15_19","Fem_15_19","Hisp_15_19","Avg_25",
                 "Whit_15_19","supp_2018"]] 
        
    # Predictors if the model user specifies the CONUS as the study region
    else:
        df = df[["Wind_Turb","Critical","Historical","Military","Mining",
                 "Nat_Parks","Dens_15_19","Trib_Land","Wild_Refug",
                 "Avg_Elevat","Avg_Temp","Avg_Wind","Bat_Count","Bird_Count",
                 "Prop_Rugg","Undev_Land","Near_Air","Near_Hosp",
                 "Near_Roads","Near_Trans","Near_Sch","Plant_Year",
                 "Farm_Year","Cost_15_19","ISO_YN","Near_Plant",
                 "Farm_15_19","Prop_15_19","In_Tax_Cre","Tax_Prop",
                 "Tax_Sale","Numb_Incen","Rep_Wins","Dem_Wins","Interconn",
                 "Net_Meter","Renew_Port","Renew_Targ","Numb_Pols",
                 "Foss_Lobbs","Gree_Lobbs","Type_15_19","Unem_15_19",
                 "Fem_15_19","Hisp_15_19","Avg_25","Whit_15_19","supp_2018"]]
            
    # Rows containing null values are removed, otherwise the model will not run.
    # Binary categorical variables are also converted into numbers, where N (no) is
    # 0 and Y (yes) is 1.
    df = df.dropna().replace(to_replace = ["Other","N","Y"], value = [0,0,1])
    
    # Predictors that take the same value in every single grid cell should be 
    # dropped, since they have no predictive power. This applies to the
    # categorical predictors
    constantDropped = []
    nunique = df.nunique()
    constantDropped = nunique[nunique == 1].index.tolist()
    # The names of the dropped predictors are written to the console output
    if len(constantDropped) == 0:        
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nPredictors removed from the model based on having a constant"
                          +'\n'+ "value in all grid cells: None", border=0)
    else:
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nPredictors removed from the model based on having a constant"
                          +'\n'+ "value in all grid cells: " + str(constantDropped), border=0)
            
    # The respective columns are dropped from the dataset
    df = df.drop(columns = constantDropped)
     
    ######################### TESTING ASSUMPTIOMS ##############################
    
    # First is a test to ensure that the relationship between the predictors
    # and the logit of the occurrence of wind farms is indeed linear
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\n------------------ TESTING ASSUMPTIONS ------------------"
                      +'\n\n'+ "Assumption #1: All continuous predictors have a linear relationship "
                      +'\n'+ "with the logit of the dependent variable, based on a Box-Tidwell test.", border=0)
    
    # The Wind_Turb column in the attribute table defines the dependent variable,
    # i.e., whether or not a grid cell contains a wind turbine.
    dfy = DataFrame(df["Wind_Turb"])
    dfy = dfy["Wind_Turb"].tolist()
    # The Wind_Turb column must also be removed since it is not itself a predictor.
    dfx = df.drop(columns = ["Wind_Turb"])
    
    # Logistic regression assumes that each predictor has a linear
    # relationship with the logit of the dependent variable. Continuous 
    # predictors that do not pass a Box-Tidwell test are therefore dropped 
    columnNames = dfx.columns.tolist()

    # A dataframe is created for testing the first assumption
    dfAssumptionOne = pd.DataFrame()
    
    # Log-transformed versions of the continuous predictors are calculated e.g.. Wind_Speed * Log(Wind_Speed)
    for predictor in columnNames:
        
        # Categorical and discrete predictors are skipped
        if (predictor != "Critical" and predictor != "Historical"
        and predictor != "Military" and predictor != "Mining"
        and predictor != "Nat_Parks" and predictor != "Trib_Land"
        and predictor != "Wild_Refug" and predictor != "Bat_Count"
        and predictor != "Bird_Count" and predictor != "Plant_Year"
        and predictor != "Farm_Year" and predictor != "ISO_YN"
        and predictor != "In_Tax_Cre" and predictor != "Tax_Prop"
        and predictor != "Tax_Sale" and predictor != "Numb_Incen"
        and predictor != "Rep_Wins" and predictor != "Dem_Wins"
        and predictor != "Interconn" and predictor != "Net_Meter" 
        and predictor != "Renew_Port" and predictor != "Renew_Targ"
        and predictor != "Numb_Pols" and predictor != "Foss_Lobbs" 
        and predictor != "Gree_Lobbs" and predictor != "supp_2018"):
            dfAssumptionOne[predictor] = dfx[predictor].astype(float)
            dfAssumptionOne[f'{predictor}:Log_{predictor}'] = dfx[predictor].apply(lambda x: float(x) * np.log(float(x)))

    # Some of the continuous predictors (e.g., Land_Slope, Undevelopable_Land)
    # sometimes do take a value of 0, and thus become NaN when log-transformed.
    # Their log transformations are thus replaced with 0.
    dfAssumptionOne = dfAssumptionOne.fillna(0)
        
    # Column names of the non-transformed predictors only
    dfNonTransformed = [col for col in dfAssumptionOne.columns if ":" not in col]
    
    # Add a constant term
    dfAssumptionOne = sm.add_constant(dfAssumptionOne, prepend=False)   
        
    # A generalized linear model is constructed, first by adding a constant
    # and then fitting to the continuous predictors and their log transformations
    try:
        logit_results = sm.GLM(dfy, dfAssumptionOne, family=sm.families.Binomial()).fit()
        
        # If (quasi-)complete separation does not occur, then the linearity test
        # can be completed
        if logit_results:
            # The p-values for each predictor
            logit_pvalues = logit_results.pvalues
            
            # The p-values of the constant and the non-transformed predictors are
            # removed from the list
            logit_pvalues = logit_pvalues[:-1]
                    
            # A Bonferroni correction is used to account for false positive
            # statistical significance
            bonferroni = 0.05/len(logit_pvalues)
            
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nBonferroni-corrected p-value: " + str(bonferroni), border=0)

            # Interest is in the p-values of the log-transformed predictors only, which
            # are every other p-value in the list
            logit_pvalues = logit_pvalues[1::2]
            
            # Log-transformed predictors with a p-value less than the bonferroni
            # correction are statistically significantly non-linear in their
            # relationship with the logit of wind farm occurrence. These predictors
            # must be dropped from the model.
            assumptionOneDropped = []
            for i in range(len(logit_pvalues)):
                if logit_pvalues[i] < bonferroni:
                    assumptionOneDropped.append(dfNonTransformed[i])
            
            # Results from the Box-Tidwell test are presented as a dataframe
            pvalueList = logit_pvalues.tolist()
            dfBoxTidwell = pd.DataFrame()
            dfBoxTidwell["Predictor"] = dfNonTransformed
            dfBoxTidwell["pval"] = pvalueList
            dfBoxTidwell = dfBoxTidwell.sort_values("pval", ignore_index = True)
            dfBoxTidwell = dfBoxTidwell.to_string(justify = 'center', col_space = 30, index = False)
                        
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nResults of the Box-Tidwell test: \n", border=0)
            pdf.multi_cell(w=0, h=5.0, align='R', txt=dfBoxTidwell, border = 0)
            
            # When no predictors are dropped following the Box-Tidwell test,
            # the following is written to the console output
            if len(assumptionOneDropped) == 0:
                pdf.multi_cell(w=0, h=5.0, align='L', 
                              txt="\nPredictors to be removed based on a non-linear relationship "
                                  +'\n'+ "with the logit of likelihood of wind farm occurrence: None", border=0)
                
            # If predictors are dropped, the user is given the opportunity to
            # ask whether they should be dropped
            else: 
                print('''\nPredictors to be removed based on a non-linear relationship with '''
                      '''the logit of likelihood of wind farm occurrence: \n'''
                      + str(assumptionOneDropped) +
                      '''\nDo you wish to remove these predictors from the model?''')
                
                # Specify using Y or N whether the user would like to choose a different
                # resolution
                def remove(values, message):
                    while True:
                        x = input(message) 
                        if x in values:
                            remove.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                remove(["Y", "N"], "Y or N:\n")
                
                # If the user says yes, the predictors are dropped
                if remove.YesOrNo == "Y":
                    # The respective columns are dropped from the dataframe
                    dfx = dfx.drop(columns = assumptionOneDropped)
                    pdf.multi_cell(w=0, h=5.0, align='L', 
                                  txt="\nPredictors removed from the model based on the results "
                                      +'\n'+ "of the Box-Tidwell test: " + str(assumptionOneDropped), border=0)
                else:
                    pdf.multi_cell(w=0, h=5.0, align='L', 
                                  txt="\nThe Box-Tidwell test would have dropped " + str(assumptionOneDropped)
                                      +'\n'+ "from the model, though the user chose to retain them.", border=0)
                    
                    # The variable holding the dropped predictors is emptied
                    assumptionOneDropped = []
                                        
    # The linearity test can fail due to complete separation of the binary
    # dependent variable resulting from the value of one or more predictors
    except:
        print('''\nComplete or quasi-complete separation has occurred due to the coarse '''
              '''resolution and/or a lack of grid cells containing wind farms. The test for '''
              '''linearity cannot be completed; do you still wish to continue the model run?''')
        
        # Specify using Y or N whether the user would like to remove them
        def proceed(values, message):
            while True:
                x = input(message) 
                if x in values:
                    proceed.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        proceed(["Y", "N"], "Y or N:\n")
        
        # The code stops if the user specifies N
        if proceed.YesOrNo != "Y":
            sys.exit()
        else:            
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nThe Box-Tidwell test cannot complete due to (quasi-)complete separation "
                              +'\n'+ "of the logit of the depenent variable. Although the linearity of the model's "
                              +'\n'+ "predictors cannot be ascertained, the user has chosen to continue the model "
                              +'\n'+ "run. No predictors have been removed based on Assumption #1."
                              +'\n\n'+"The user has specified the following predictor configurations: "
                              +'\n'+ str(configList), border=0)
            
            # No predictors were dropped
            assumptionOneDropped = []
                        
    # The second assumption is the multicollinearity test, to ensure that 
    # all predictors have independent effects on WiFSS    
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\nAssumption #2: There is no multicollinearity, or pairwise collinearity, "
                      +'\n'+ "between the model's predictors, based on Variance Inflaction Factors (VIF).", border=0)
    
    # First step is to normalize the predictors and grid cells that have been
    # retained from applying the previous assumption
    columnNames = dfx.columns.tolist()
    dfNormalized = pd.DataFrame()
    dfCategorical = pd.DataFrame()
    
    # Categorical predictors must be separated from those that are to be
    # normalized
    for predictor in columnNames:
        if (predictor == "Critical" or predictor == "Historical"
        or predictor == "Military" or predictor == "Mining"
        or predictor == "Nat_Parks" or predictor == "Trib_Land"
        or predictor == "Wild_Refug" or predictor == "ISO_YN"
        or predictor == "In_Tax_Cre" or predictor == "Tax_Prop"
        or predictor == "Tax_Sale" or predictor == "Interconn"
        or predictor == "Net_Meter" or predictor == "Renew_Port"):
            dfCategorical[predictor] = dfx[predictor]
        else:
            dfNormalized[predictor] = dfx[predictor].astype(float)           
    
    # Quantitative data that take the same value at all grid cells should not
    # be normalized, and are thus separated before normalization
    nunique = dfNormalized.nunique()
    doNotNormalize = nunique[nunique == 1].index.tolist()
    dfNotNormalized = dfx[doNotNormalize]
    
    # The normalization is executed using standard scores for each grid cell
    dfNormalized = dfNormalized.drop(doNotNormalize, axis = 1).astype(float)
    dfNormalized = (dfNormalized - dfNormalized.mean())/dfNormalized.std()
        
    # The normalized and categorical columns can now be recombined
    dfx = concat([dfNormalized,dfNotNormalized,dfCategorical], axis = 1)    
    
    # Multicollinearity is assessed by first creating a dataframe to hold
    # VIF values, which are calculated for each predictor in the above
    # multiple linear regression model.
    vif = DataFrame()
    vif['Predictor'] = dfx.columns
    vif['VIF'] = [variance_inflation_factor(dfx.values, i) for i in range(dfx.shape[1])]
    # The VIF data are sorted, with each row showing the collinearity that each
    # variable has with the others. The bigger the value, the bigger the
    # multicollinearity and thus the less unique the effect the variable can have
    # in the logistic regression model.
    vif = vif.sort_values(["VIF"]).reset_index(drop = True)
    vifGrouped = vif.to_string(justify = 'center', col_space = 25, index = False)
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\nGrouped Multicollinearity Test Results:\n", border=0)
    pdf.multi_cell(w=0, h=5.0, align='R', txt=vifGrouped, border = 0)

    # A predictor with a VIF above 10 is considered to be too strongly correlated
    # with the other predictors. These predictors are isolated.
    vifCorrelate = vif[vif["VIF"] >= 10]
    vifCorrelateList = vifCorrelate["Predictor"].tolist()
    # Predictors that possess a value of NaN based on the multicollinearity test
    # have the same value across all grid cells, and thus have no unique spatial
    # influence on WiFSS. These predictors must also be isolated.
    vifNaN = vif[vif["VIF"].isna()]
    vifNaNList = vifNaN["Predictor"].tolist()
    vifRemoveList = vifCorrelateList + vifNaNList
    
    # Predictors are now tested for collinearity in pairwise combinations, rather
    # than altogether as done above. VIF is calculated for each pair.
    vif = 1/(1-dfx.corr()**2)
    # The upper triangle of the pairwise matrix is deleted, and VIF values
    # are sorted by magnitude.
    vif = vif.where(np.triu(np.ones(vif.shape),k=1).astype(bool))
    vif = vif.stack().sort_values(axis = 0).reset_index()
    vif.columns = ["Predictor1","Predictor2","VIF"]
    vifPairs = vif.to_string(justify = 'center', col_space = 25, index = False, max_rows = 10)
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\nPairwise Multicollinearity Test Results:\n", border=0)
    pdf.multi_cell(w=0, h=5.0, align='R', txt=vifPairs, border = 0)

    # Pairs of predictors with a VIF above 10 are isolated and added to a list.
    vifPairwiseRemove = vif[vif["VIF"] >= 10]
    var1List = vifPairwiseRemove["Predictor1"].tolist()
    var2List = vifPairwiseRemove["Predictor2"].tolist()
    vifPairwiseRemoveList = var1List + var2List
    
    # The predictors from the multicollinearity and pairwise collinearity tests
    # are combined and duplicates are removed
    assumptionTwoDropped = [*set(vifRemoveList + vifPairwiseRemoveList)]
    if len(assumptionTwoDropped) == 0:
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nPredictors to be removed based on multicollinearity: None", border=0)
        
    # The user is given the opportunity to remove the multicollinear predictors
    else: 
        print('''\nPredictors to be removed based on multicollinearity: \n'''
              + str(assumptionTwoDropped) +
              '''\nDo you wish to remove these predictors from the model?''')
        
        # Specify using Y or N whether the user would like to remove them
        def remove(values, message):
            while True:
                x = input(message) 
                if x in values:
                    remove.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        remove(["Y", "N"], "Y or N:\n")
        
        # If the user says yes, the predictors are dropped
        if remove.YesOrNo == "Y":
            # The respective columns are dropped from the dataframe
            dfx = dfx.drop(columns = assumptionTwoDropped)            
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nPredictors to be removed from the model based on multicollinearity: \n" + str(assumptionTwoDropped), border=0)
                         
        else:
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nThe multicollinearity test would have dropped \n" + str(assumptionTwoDropped) 
                          +'\n'+ "from the model, though the user chose to retain them.", border=0)
            
            # The variable holding the dropped predictors is emptied
            assumptionTwoDropped = []
    
    # The final assumption is the test for outliers, to ensure that none of the 
    # aggregated to the grid cells will bias the training and testing stages  
    # of this model
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\nAssumption #3: None of the grid cells contain data that represent "
                  +'\n'+ "extreme outliers, based on a Cook's distance test.", border=0)

    # An ordinary least squares regression fit between the wind turbine 
    # locations (dfy) and the grid cell data (dfRenamed) is conducted,
    # and identified outlying grid cells are dropped.
    outlierTest = OLS(dfy,dfx).fit()
    influence = outlierTest.get_influence()
    cooks = influence.cooks_distance
    count = 0
    for i in range(len(cooks[1])):
        if cooks[1][i] < 0.05:
            np.delete(dfy, i)
            np.delete(dfx, i)
            count = count + 1
    
    # Final results of the three assumptions are written to the console output
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\nNumber of grid cells removed due to outlying observations according to a "
                  +'\n'+ "Cook's distance test: " + str(count)
                  +'\n\n'+ "Final list of predictors that did not pass the model's three assumptions: "
                  +'\n'+ str(constantDropped + assumptionOneDropped + assumptionTwoDropped), border=0)
    
    ############### MAKING THE TRAINING AND TESTING DATASETS ##################

    # The predictors used in the training and testing datasets depend on 
    # the configuration(s) selected by the end-user
    for g in range(len(configList)):
        
        # The No_Wind configuration requires that wind speed be dropped as a 
        # predictor before the model run starts
        if configList[g] == "No_Wind":
            dfxConfig = dfx.loc[:, dfx.columns != "Avg_Wind"]
            
        # The Wind_Only configuration requires wind speed to be the only
        # predictor that is retained
        elif configList[g] == "Wind_Only":
            try:
                dfxConfig = dfx[["Avg_Wind"]]
            except:
                print('''\nAvg_Wind was removed as a predictor by the user when assumptions were tested, '''
                      '''\nmeaning the Wind_Only predictor configuration will not be used.''')
                pdf.multi_cell(w=0, h=5.0, align='L', 
                              txt="\nAvg_Wind was removed as a predictor by the user when assumptions were tested, "
                              +"\n"+"meaning the Wind_Only predictor configuration will not be used.", border=0)
                configList.remove(configList[g])
                break
            
        # For the Full and Reduced configurations, no predictors need to be
        # pre-emptively removed
        elif configList[g] == "Full" or configList[g] == "Reduced":
            dfxConfig = dfx
                
        # Numpy arrays are constructed for each of the predictors that will inform
        # the logistic regression model.    
        predictorArrays = [DataFrame(dfxConfig.iloc[:,i]).to_numpy().reshape(-1,1) for i in range(len(dfxConfig.columns))]
        
        # The predictors are concatenated into the same array
        dfxArray = np.concatenate((predictorArrays[0:len(predictorArrays)]), axis = 1)

        # Console output to signify the beginning of outputs from a predictor configuration
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n ################# " + configList[g] + " Configuration Output Begins ################# \n", border=0)
        
        # The names of the predictors retained by the model are added to a 
        # separate list, which will be used for holding the coefficients should
        # the user wish to employ a cellular automaton in the model's second
        # script
        predictorCodeNames = dfxConfig.columns.tolist()
        
        ######################## MODEL CALIBRATION #########################
        
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n--------------- MODEL CALIBRATION (Training Data): " + configList[g] + " Configuration ---------------", border=0)
        
        # These two lists will hold the log-likelihood scores obtained from each 
        # combination of grid cells (30) used to separately train the model
        trainedModelScores = []
        trainedModelAppend = trainedModelScores.append
        nullModelScores = []
        nullModelAppend = nullModelScores.append
    
        # This list holds the intercept of the logistic regression model, which
        # will be called upon when computing each grid cell's probability
        interceptList = []
        interceptAppend = interceptList.append
        
        # This list holds the McFadden's Adjusted Pseudo R-squared scores for each
        # of the 30 calibration runs of the model
        rSquaredList = []
        rSquaredAppend = rSquaredList.append
    
        # An empty list to hold the coefficient for each predictor produced 
        # over the 30 calibration model runs
        coefList = []
        coefAppend = coefList.append
        
        # This list holds the Area Under Curve statistic computed from plotting
        # a Receiver Operating Characteristic curve for each sample of testing
        # (validation) grid cells
        aucList = []
        aucAppend = aucList.append
    
        # This list holds the classification threshold from the ROC curve at which
        # true positive classification of grid cells as containing wind farms
        # (in the testing dataset) maximizes, and false positive classification
        # minimizes
        thresholdList = []
        thresholdAppend = thresholdList.append
    
        # These lists hold the false positive rate and true positive rates
        # obtained from classifying the training data at the various thresholds
        # in the list above
        fprList = []
        fprAppend = fprList.append
        tprList = []
        tprAppend = tprList.append    
    
        # This list holds a confusion matrix of the true positive, false positive,
        # true negative, and false negative predictions of the state of grid cells
        # in the testing dataset
        cmList = []
        cmAppend = cmList.append
    
        # The number of degrees of freedom that the model possesses
        degrees = []
        degreesAppend = degrees.append
        
        # The count and pvalues must also be saved from running the function
        countList = []
        countAppend = countList.append
        pValCountList = []
        pValCountAppend = pValCountList.append
        
        print("\n" + configList[g] + " model training and testing in progress...")
                
        # The logistic regression model is run 30 times, in order to account for 
        # different combinations of training and testing grid cells and to diagnose
        # an average model performance
        def trainingTestingModel(allData):
            
            # This count variable will keep track of how frequently the model
            # outperforms the null model (intercept-only) for different training
            # grid cell combinations
            count = 0
        
            # This count variable will keep track of how many times the model's
            # outperformance of the null model is statistically significant (p < 0.05)
            pValCount = 0
            
            # 25% of the grid cells train the model, and 75% test it. The datasets
            # are stratified such that equal numbers of grid cells containing wind
            # turbines exist in the training and testing datasets
            X_train, X_test, y_train, y_test = train_test_split(allData, dfy, train_size = 0.75, stratify = dfy)
            
            # The logistic regression model is constructed. A low value for C prevents
            # overfitting. A balanced class weight prevents a unit weight of 1 being
            # applied to each variable.
            model = LogisticRegression(solver = "liblinear", C = 1, class_weight = "balanced")
            
            # The model is fitted to the training grid cells. The training grid 
            # cells define the model's coefficients, i.e., the associations that 
            # will predict whether a grid cell contains a wind turbine or not.
            model.fit(X_train,y_train)
    
            # The coefficients from the model run are saved to the empty list
            coefAppend(model.coef_)
                
            # The intercept from this trained model defines the null model. The 
            # intercept is extended to the same length as the training dataset
            intercept = model.intercept_.tolist()
            intercept = [i for i in intercept for r in range(len(y_train))]
            interceptAppend(intercept)

            # A likelihood-ratio test for performance against the null model is 
            # needed to determine how much better this trained model can correctly
            # predict the existence of wind farms in the training grid cells.
            # The log-likelihood of the trained model is calculated first
            X_train = sm.add_constant(X_train)
            trained_model = OLS(y_train,X_train).fit()
            trained_ll = trained_model.llf
            
            # Next the log-likelihood of the null model is computed
            null_model = OLS(y_train,intercept).fit()
            null_ll = null_model.llf

            # These two log-likelihood scores are saved to empty lists
            trainedModelAppend(trained_ll)
            nullModelAppend(null_ll)
        
            # The goodness-of-fit of the trained model is computed in terms of 
            # McFadden's Adjusted Pseudo R-squared, and then added to its
            # respective list. The statistic's numerator is penalized for its large 
            # number of predictors.
            trained_rsquared = 1 - (trained_ll - X_train.shape[1])/null_ll
            rSquaredAppend(trained_rsquared)
        
            # The count keeps track of how many times the trained model has better
            # goodness-of-fit than the null model to the training grid cells.
            if trained_ll > null_ll:
                count = count + 1
        
            # The likelihood ratio of the null and trained models' likelihood scores 
            # is computed, and the associated p-value based on a chi-square test.
            # The number of predictors determines the number of degrees of freedom.
            LR_trained_vs_null = -2*(null_ll-trained_ll)
            p_value = chi2.sf(LR_trained_vs_null, len(X_train[0]))
    
            # The p-value count variable is now updated
            if p_value < 0.05:
                pValCount = pValCount + 1
        
            # A Receiver Operating Characteristic is calculated to illustrate how
            # effective the trained model is at correctly classifying the testing grid 
            # cells as containing wind farms (true positive rate versus false
            # positive rate).
            y_pred_proba = model.predict_proba(X_test)[::,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            thresh = thresholds[np.argmax(tpr-fpr)]

            # The true/false positive rates, the area under curve and the optimal 
            # classification thresholds are saved to their respective lists
            fprAppend(fpr)
            tprAppend(tpr)
            aucAppend(auc)    
            thresholdAppend(thresh)
        
            # The model is used to predict whether testing grid cells should contain 
            # wind turbines, based on the predictor values in the testing grid cells and
            # the optimal classification threshold from the ROC curve.
            y_pred = (model.predict_proba(X_test)[:, 1] > thresh).astype('float')
        
            # The actual and predicted model performance for each model run, as well
            # as a confusion matrix, are saved.
            cm = confusion_matrix(y_test, y_pred)
            cmAppend(cm)
            
            # The number of degrees of freedom will be needed to assess performance
            degreesAppend(len(X_train[0]))
            
            countAppend(count)
            pValCountAppend(pValCount)
            
        # The trained and tested model is iterated 30 times
        [trainingTestingModel(dfxArray) for i in tqdm(range(30))]
        
        # The Reduced predictor configuration first requires determining
        # which combination of predictors maximizes the model's predictive power
        if configList[g] == "Reduced":
        
            print("\nPlease wait while the model determines the importance of each predictor...\n")
            
            # These empty lists hold the Reduced model configuration's performance
            # compared to the Full configuration (across 30 runs), and the
            # statistical significance of any reduction in performance
            likelihoodRatioList = []
            likelihoodRatioAppend = likelihoodRatioList.append
            pValueList = []
            pValueAppend = pValueList.append

            # Removal of predictors is performed 30 times in order to obtain a median
            # reduction of model performance that accounts for randomness
            def iteration():
                
                # This list will hold the log likelihood scores of the model as predictors
                # are removed with replacement
                reducedLogLikelihoods = []
                reducedLogAppend = reducedLogLikelihoods.append
                
                # The effect of removing each predictor in turn on the model's output
                # needs to be determined
                def predictorRemoval(reducedData):
                    
                    # A version of the training and testing datasets having been built with
                    # a predictor excluded
                    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(reducedData, dfy, train_size = 0.75, stratify = dfy)

                    # The logistic regression model is refitted using these new training
                    # and testing datasets
                    model = LogisticRegression(solver = "liblinear", C = 1, class_weight = "balanced")
                    model.fit(X_train_red,y_train_red)
                    
                    # The log-likelihood of the Reduced model's goodness-of-fit
                    # is computed in the same way that it was for the other 
                    # predictor configurations
                    red_model = OLS(y_train_red,X_train_red).fit()
                    red_ll = red_model.llf
                    reducedLogAppend(red_ll)
                    
                # The log likelihood scores and likelihood ratios obtained by removing
                # each predictor with replacement are generated
                [predictorRemoval(np.delete(dfxArray,h,1)) for h in range(len(dfxArray[0]))]
                    
                # Likelihood ratios between the Reduced log likelihoods and the median 
                # log likelihood score from the Full predictor configuration are computed. 
                # Quantization of log likelihood scores in this manner is common 
                # in signal processing (Liu et al., 2010)
                fullVersusReducedLRs = -2*(reducedLogLikelihoods - np.median(trainedModelScores))
                
                # The statistical significance of the changes in the models likelihood ratio
                # after removing each predictor with replacement are computed using a
                # chi-square test
                pValues = chi2.sf(fullVersusReducedLRs, len(fullVersusReducedLRs)-1)
                
                # The computed likelihood ratios and p-values are saved
                likelihoodRatioAppend(fullVersusReducedLRs)
                pValueAppend(pValues)
            
            # The removal of predictors is iterated 30 times
            [iteration() for i in tqdm(range(30))]
            
            # The number of times the model's performance was worsened by removing each
            # predictor with replacement is computed.
            likelihoodRatioList = np.asarray(likelihoodRatioList)
            likelihoodRatioCounts = np.count_nonzero(likelihoodRatioList >= 0, axis = 0)
            
            # Same as above, but this time for the number of tiems the worsened model
            # performance exceeded the stopping criterion (p < 0.5). Used to inform
            # predictor removal, with value derived from Mahmood et al. (2016).
            pValueList = np.asarray(pValueList)
            pValueCounts = np.count_nonzero(pValueList < 0.5, axis = 0)
            
            # A dataframe is constructed that summarizes the lower goodness-of-fit
            # of each version of the Reduced precitor configuration when compared
            # to the Full predictor configuration.
            dfReducedModel = DataFrame()
            dfReducedModel["Predictors"] = dfxConfig.columns.tolist()
            dfReducedModel["Reduced_Fit"] = likelihoodRatioCounts
            dfReducedModel["Stop_Criterion"] = pValueCounts
            dfReducedModel = dfReducedModel.sort_values(by = ["Reduced_Fit","Stop_Criterion"], ascending = False, ignore_index = True)
            
            # The predictors ordered from most to least impactful upon their 
            # removal are assigned to a list            
            finalPredictors = dfReducedModel["Predictors"].tolist()
            
            # This dataframe is written to the console output.
            dfReducedModel = dfReducedModel.to_string(justify = 'center', col_space = 25, index = False)
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nDataframe showing the lowered goodness-of-fit caused by removing each predictor"
                          +"\n"+"with replacement over 30 model runs. The columns show the number of times removal of "
                          +"\n"+"each predictor reduced the model's goodness-of-fit, and the number of times this "
                          +"\n"+"reduction exceeded a p < 0.5 stopping criterion:\n\n",border=0)  
            pdf.multi_cell(w=0, h=5.0, align='R',txt="\n"+ dfReducedModel, border=0)            

            # Lists that will be used to create a dataframe summarizing the predictive
            # power of each set of model predictors
            numPredictors = list(range(1,len(finalPredictors)+1))
            medAccuracyList = []
            medAccuracyAppend = medAccuracyList.append
            trueVersusFalseList = []
            trueVersusFalseAppend = trueVersusFalseList.append
            
            # Predictors are inserted into the Reduced model from the most to least
            # impactful on the full model's goodness of fit, in order to determine the 
            # number of predictors that maximizes (minimizes) the model's true positive
            # (false positive) rate.
            print("\nPlease wait while the model's accuracy with different predictor combinations is assessed...\n")
            
            def predictorCombos():
                for h in tqdm(range(len(numPredictors))):
                    
                    # Dataframe is sliced to contain only the predictors of interest
                    dfReduced = dfxConfig[finalPredictors[0:h+1]]
                    
                    # Arrays for each Reduced set of predictors are constructed.
                    predictorArrays = [DataFrame(dfReduced.iloc[:,i]).to_numpy().reshape(-1,1) for i in range(len(dfReduced.columns))]
                    dfx = np.concatenate((predictorArrays[0:len(predictorArrays)]), axis = 1)
                
                    # A confusion matrix will summarize the model performance for each
                    # number of events per variable
                    cmList = []
                    cmListAppend = cmList.append
                    
                    # The proportion of grid cell states that are correctly predicted using
                    # each set of predictors
                    accuracyList = []
                    accuracyListAppend = accuracyList.append
                
                    # The model is run for 30 different training and testing grid cell 
                    # combinations for each set of refined predictors to obtain
                    # an average model performance
                    def iteration(allData):
                        # Datasets are defined and the model is fitted
                        X_train, X_test, y_train, y_test = train_test_split(allData, dfy, train_size = 0.75, stratify = dfy)
                        model = LogisticRegression(solver = "liblinear", C = 1, class_weight = "balanced")
                        model.fit(X_train,y_train)
                
                        # The ROC curve function is used to determine the optimal classification
                        # threshold for the testing grid cells.
                        y_pred_proba = model.predict_proba(X_test)[::,1]
                        fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
                        thresh = thresholds[np.argmax(tpr-fpr)]
                        
                        # Predicted state of the grid cells in the testing dataset.
                        y_pred = (model.predict_proba(X_test)[:, 1] > thresh).astype('float')
                        
                        # The actual and predicted model performance for each model run, as well
                        # as a confusion matrix, are saved.
                        cm = confusion_matrix(y_test, y_pred)
                        cmListAppend(cm)
                        
                        # Proportion of grid cells that are correctly predicted as containing
                        # or not containing wind farms
                        accuracy = accuracy_score(y_test, y_pred)
                        accuracyListAppend(accuracy)
                    
                    [iteration(dfx) for i in range(30)]
                
                    # The median of the model's accuracy with each set of predictors
                    accuracyMed = np.median(accuracyList, axis = 0)
                    # The value is added to a list
                    medAccuracyAppend(accuracyMed)
                        
                    # The median confusion matrix for the 30 model runs produced using
                    # each set of predictors
                    cmMed = np.median(cmList, axis = 0).round().astype(int)
                
                    # The ratio of true to false positive wind farm predictions based
                    # on the median confusion matrix
                    trueVersusFalsePositive = cmMed[1][1]/cmMed[0][1]
                    # The value is added to a list
                    trueVersusFalseAppend(trueVersusFalsePositive)
            
            predictorCombos()
            
            # A dataframe is created that summarizes the accuracy of the model's predictions
            # of wind farm locations for each set of predictors
            dfReduced = DataFrame()
            dfReduced["Num_Pred"] = numPredictors
            dfReduced["Accuracy"] = medAccuracyList
            dfReduced["True_False"] = trueVersusFalseList
            dfReduced = dfReduced.sort_values(by = ["Accuracy","True_False"], ascending = False, ignore_index = True)
            # A set of predictors that does not predict any of the grid cells as containing
            # wind farms (true to false positive ratio = NaN) is useless, because some
            # wind farms do exist. These rows are omitted from the dataframe
            dfReduced = dfReduced.dropna().reset_index()
            
            # The number of predictors that maximizes the model's accuracy and the true to
            # false positive ratio
            finalNumber = dfReduced["Num_Pred"][0]
            
            # This dataframe is written to the console output.
            dfReduced = dfReduced.to_string(justify = 'center', col_space = 25, index = False)
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nDataframe of model performance for each set of predictors, showing the "
                          +"\n"+"Number of predictors in each combination, the median number of accurately "
                          +"\n"+"predicted grid cell states, and the ratio of true-to-false positive predictions: \n", border=0)
            pdf.multi_cell(w=0, h=5.0, align='R',txt="\n"+ dfReduced, border=0)            
            
            # The set of predictors comprised of this number of them is identified as 
            # the set to used in the Reduced predictor configuration for this model.
            finalPredictors = finalPredictors[0:finalNumber]
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nSet of predictors (" + str(finalNumber) + " total) to be used in the Reduced Model:"
                          +"\n\n"+str(finalPredictors), border=0)
            
            # This line reduces the number of predictors to the refined set above
            dfxConfig = dfxConfig[finalPredictors]
            
            # The names of the predictors retained by the model are added to a 
            # separate list, which will be used for holding the coefficients should
            # the user wish to employ a cellular automaton in the model's second
            # script
            predictorCodeNames = dfxConfig.columns.tolist()
            
            # Numpy arrays are constructed for each of the predictor that will inform
            # the logistic regression model.
            predictorArrays = [DataFrame(dfxConfig.iloc[:,i]).to_numpy().reshape(-1,1) for i in range(len(dfxConfig.columns))]
            
            # The predictors are concatenated into the same array
            dfxArray = np.concatenate((predictorArrays[0:len(predictorArrays)]), axis = 1)    
            
            # The lists holding the outputs from the initial run of the model
            # are emptied, now that the Reduced predictors have been identified
            trainedModelScores.clear()
            nullModelScores.clear()
            interceptList.clear()
            rSquaredList.clear()
            coefList.clear()
            aucList.clear()
            thresholdList.clear()
            fprList.clear()
            tprList.clear()
            cmList.clear()
            degrees.clear()
            countList.clear()
            pValCountList.clear()
            
            # The logistic regression model is run 30 times, in order to account for 
            # different combinations of training and testing grid cells and to diagnose
            # an average model performance
            def trainingTestingModel(allData):
                
                # This count variable will keep track of how frequently the model
                # outperforms the null model (intercept-only) for different training
                # grid cell combinations
                count = 0
            
                # This count variable will keep track of how many times the model's
                # outperformance of the null model is statistically significant (p < 0.05)
                pValCount = 0
                
                # 25% of the grid cells train the model, and 75% test it. The datasets
                # are stratified such that equal numbers of grid cells containing wind
                # turbines exist in the training and testing datasets
                X_train, X_test, y_train, y_test = train_test_split(allData, dfy, train_size = 0.75, stratify = dfy)
                
                # The logistic regression model is constructed. A low value for C prevents
                # overfitting. A balanced class weight prevents a unit weight of 1 being
                # applied to each variable.
                model = LogisticRegression(solver = "liblinear", C = 1, class_weight = "balanced")
                
                # The model is fitted to the training grid cells. The training grid 
                # cells define the model's coefficients, i.e., the associations that 
                # will predict whether a grid cell contains a wind turbine or not.
                model.fit(X_train,y_train)
        
                # The coefficients from the model run are saved to the empty list
                coefAppend(model.coef_)
                    
                # The intercept from this trained model defines the null model. The 
                # intercept is extended to the same length as the training dataset
                intercept = model.intercept_.tolist()
                intercept = [i for i in intercept for r in range(len(y_train))]
                interceptAppend(intercept)
                
                # A likelihood-ratio test for performance against the null model is 
                # needed to determine how much better this trained model can correctly
                # predict the existence of wind farms in the training grid cells.
                # The log-likelihood of the trained model is calculated first
                trained_model = OLS(y_train,X_train).fit()
                trained_ll = trained_model.llf
                
                # Next the log-likelihood of the null model is computed
                null_model = OLS(y_train,intercept).fit()
                null_ll = null_model.llf
                
                # These two log-likelihood scores are saved to empty lists
                trainedModelAppend(trained_ll)
                nullModelAppend(null_ll)
            
                # The goodness-of-fit of the trained model is computed in terms of 
                # McFadden's Adjusted Pseudo R-squared, and then added to its
                # respective list. The statistic's numerator is penalized for its large 
                # number of predictors.
                trained_rsquared = 1 - (trained_ll - X_train.shape[1])/null_ll
                rSquaredAppend(trained_rsquared)
            
                # The count keeps track of how many times the trained model has better
                # goodness-of-fit than the null model to the training grid cells.
                if trained_ll > null_ll:
                    count = count + 1
            
                # The likelihood ratio of the null and trained models' likelihood scores 
                # is computed, and the associated p-value based on a chi-square test.
                # The number of predictors determines the number of degrees of freedom.
                LR_trained_vs_null = -2*(null_ll-trained_ll)
                p_value = chi2.sf(LR_trained_vs_null, len(X_train[0]))
        
                # The p-value count variable is now updated
                if p_value < 0.05:
                    pValCount = pValCount + 1
            
                # A Receiver Operating Characteristic is calculated to illustrate how
                # effective the trained model is at correctly classifying the testing grid 
                # cells as containing wind farms (true positive rate versus false
                # positive rate).
                y_pred_proba = model.predict_proba(X_test)[::,1]
                fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
                auc = metrics.roc_auc_score(y_test, y_pred_proba)
                thresh = thresholds[np.argmax(tpr-fpr)]

                # The true/false positive rates, the area under curve and the optimal 
                # classification thresholds are saved to their respective lists
                fprAppend(fpr)
                tprAppend(tpr)
                aucAppend(auc)    
                thresholdAppend(thresh)
            
                # The model is used to predict whether testing grid cells should contain 
                # wind turbines, based on the predictor values in the testing grid cells and
                # the optimal classification threshold from the ROC curve.
                y_pred = (model.predict_proba(X_test)[:, 1] > thresh).astype('float')
            
                # The actual and predicted model performance for each model run, as well
                # as a confusion matrix, are saved.
                cm = confusion_matrix(y_test, y_pred)
                cmAppend(cm)
                
                # The number of degrees of freedom will be needed to assess performance
                degreesAppend(len(X_train[0]))
                
                countAppend(count)
                pValCountAppend(pValCount)
                
            # The trained and tested model is iterated 30 times
            [trainingTestingModel(dfxArray) for i in tqdm(range(30))]
            
            
        print(configList[g] + " model training and testing complete.\n")

        #################### ASSESSMENT OF MODEL PERFORMANCE #####################
        
        # The performance of the null and trained logistic regression model due to 
        # changes in the training grid cells are summarized    
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nRange of log-likelihood scores from 30 training runs of the " + configList[g] + " model: "
                      +"\n"+ "Maximum Score: " + str(max(trainedModelScores))
                      +"\n"+ "Median Score: " + str(np.median(trainedModelScores))
                      +"\n"+ "Minimum Score: " + str(min(trainedModelScores))
                      +"\n\n"+ "Range of log-likelihood scores of the Null model: "
                      +"\n"+ "Maximum Score: " + str(max(nullModelScores))
                      +"\n"+ "Median Score: " + str(np.median(nullModelScores))
                      +"\n"+ "Minimum Score: " + str(min(nullModelScores))
                      +"\n\n"+ "Number of times (out of 30) the " + configList[g] + " model possesses a greater "
                      +"\n"+ "goodness-of-fit: " + str(sum(countList))
                      +"\n"+ "Number of times (out of 30) the " + configList[g] + " model's outperformance of the Null model "
                      +"\n"+ "is statistically significant: " + str(sum(pValCountList)), border=0)
            
        # The Median Log-Likelihood Ratio across all 30 training dataset combinations
        # can now be calculated. The median is calculated rather than the mean
        # because many of the model's predictors are categorical
        median_LR_trained_vs_null = -2*(np.median(nullModelScores)-np.median(trainedModelScores))
        # Statistical significance of the Median Log-Likelihood Ratio
        p_val_trained_vs_null = chi2.sf(median_LR_trained_vs_null, degrees[0]-1)
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nMedian Log-Likelihood Ratio, " + configList[g] + " model vs. Null model: " + str(median_LR_trained_vs_null)
                      +"\n"+ "p-value of the Median Log-Likelihood Ratio: " + str(p_val_trained_vs_null), border=0)
            
        # The median, maximum, and minimum McFadden Adjusted Pseudo R-squared values
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nRange of McFadden Adjusted Psuedo R-Squared statistics for the " + configList[g] + " model: "
                      +"\n"+ "Minimum Pseudo R-Squared: " + str(min(rSquaredList))
                      +"\n"+ "Median Pseudo R-Squared: " + str(np.median(rSquaredList))
                      +"\n"+ "Maximum Pseudo R-Squared: " + str(max(rSquaredList)), border=0)
            
        # The lower quartile, median, and upper quartile for the coefficients for 
        # each predictor are computed
        coef25 = np.percentile(coefList, 25, axis = 0).flatten().tolist()
        coefMedTrained = np.median(coefList, axis = 0).flatten().tolist()
        coef75 = np.percentile(coefList, 75, axis = 0).flatten().tolist()
        
        # The median coefficients and the predictor names are added to a new
        # pandas dataframe, which will be needed for the cellular automaton
        # script. Full names for each predictor are added to the dataframe
        # for clarity when constructing cellular automaton scenarios.
        predictors = []
        for i in range(len(predictorCodenames)):
            if predictorCodenames[i] == "Avg_25":
                predictors.append("Percent Under 25")
            if predictorCodenames[i] == "Avg_Elevat":
                predictors.append("Average Elevation")
            if predictorCodenames[i] == "Avg_Temp":
                predictors.append("Average Temperature")
            if predictorCodenames[i] == "Avg_Wind":
                predictors.append("Average Wind Speed")
            if predictorCodenames[i] == "Bat_Count":
                predictors.append("Bat Species Count")
            if predictorCodenames[i] == "Bird_Count":
                predictors.append("Bird Species Count")
            if predictorCodenames[i] == "Cost_15_19":
                predictors.append("Electricity Cost")
            if predictorCodenames[i] == "Critical":
                predictors.append("Critical Habitats")
            if predictorCodenames[i] == "Dem_Wins":
                predictors.append("Presidential Elections")
            if predictorCodenames[i] == "Dens_15_19":
                predictors.append("Population Density")
            if predictorCodenames[i] == "Farm_15_19":
                predictors.append("Farmland Value")
            if predictorCodenames[i] == "Farm_Year":
                predictors.append("Wind Farm Age")
            if predictorCodenames[i] == "Fem_15_19":
                predictors.append("Percent Female")
            if predictorCodenames[i] == "Foss_Lobbs":
                predictors.append("Fossil Fuel Lobbies")
            if predictorCodenames[i] == "Gree_Lobbs":
                predictors.append("Green Lobbies")
            if predictorCodenames[i] == "Hisp_15_19":
                predictors.append("Percent Hispanic")
            if predictorCodenames[i] == "Historical":
                predictors.append("Historical Landmarks")
            if predictorCodenames[i] == "Interconn":
                predictors.append("Interconnection Policy")
            if predictorCodenames[i] == "In_Tax_Cre":
                predictors.append("Investment Tax Credits")
            if predictorCodenames[i] == "ISO_YN":
                predictors.append("ISOs")
            if predictorCodenames[i] == "Military":
                predictors.append("Military Installations")
            if predictorCodenames[i] == "Mining":
                predictors.append("Mining Operations")
            if predictorCodenames[i] == "Nat_Parks":
                predictors.append("National Parks")
            if predictorCodenames[i] == "Near_Air":
                predictors.append("Nearest Airport")
            if predictorCodenames[i] == "Near_Hosp":
                predictors.append("Nearest_Hospital")
            if predictorCodenames[i] == "Near_Plant":
                predictors.append("Nearest Power Plant")
            if predictorCodenames[i] == "Near_Roads":
                predictors.append("Nearest Road")
            if predictorCodenames[i] == "Near_Sch":
                predictors.append("Nearest School")
            if predictorCodenames[i] == "Net_Meter":
                predictors.append("Net Metering Policy")
            if predictorCodenames[i] == "Near_Trans":
                predictors.append("Nearest Transmission Line")
            if predictorCodenames[i] == "Numb_Incen":
                predictors.append("Financial Incentives")
            if predictorCodenames[i] == "Numb_Pols":
                predictors.append("Political Legislations")
            if predictorCodenames[i] == "Plant_Year":
                predictors.append("Power Plant Age")
            if predictorCodenames[i] == "Prop_15_19":
                predictors.append("Property Value")
            if predictorCodenames[i] == "Prop_Rugg":
                predictors.append("Rugged Land")
            if predictorCodenames[i] == "Renew_Port":
                predictors.append("RPS Policy")
            if predictorCodenames[i] == "Renew_Targ":
                predictors.append("RPS Target")
            if predictorCodenames[i] == "Rep_Wins":
                predictors.append("Gubernatorial Elections")
            if predictorCodenames[i] == "supp_2018":
                predictors.append("RPS Support")
            if predictorCodenames[i] == "Tax_Prop":
                predictors.append("Property Tax Exemptions")
            if predictorCodenames[i] == "Tax_Sale":
                predictors.append("Sales Tax Abatements")
            if predictorCodenames[i] == "Trib_Land":
                predictors.append("Tribal Lands")
            if predictorCodenames[i] == "Type_15_19":
                predictors.append("Employment Type")
            if predictorCodenames[i] == "Undev_Land":
                predictors.append("Undevelopable Land")
            if predictorCodenames[i] == "Unem_15_19":
                predictors.append("Unemployment Rate")
            if predictorCodenames[i] == "Whit_15_19":
                predictors.append("Percent White")
            if predictorCodenames[i] == "Wild_Refug":
                predictors.append("Wildlife Refuges")

        # The dataframe is created and saved as a .csv file
        dfCoefficients = pd.DataFrame()
        dfCoefficients["Predictors"] = predictors
        dfCoefficients["Predictor_Codes"] = predictorCodenames
        dfCoefficients["Coefficients"] = coefMedTrained
        dfCoefficients = dfCoefficients.sort_values("Predictors", ignore_index = True)
        dfCoefficients.to_csv("".join([r"E:\Dissertation_Resources\Coefficients/", studyRegion.region, "/Coeffs_", configList[g], "_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, ".csv"]))

        # Median coefficients are ranked according to their magnitude, to convey
        # strength of association with the binary grid cell state
        rankedCoefficients = len(coefMedTrained) - rankdata([abs(-1 * i) for i in coefMedTrained]) + 1
        
        # The same is done for the odds ratios of each predictor
        odds25 = np.exp(coef25)
        oddsMed = np.exp(coefMedTrained)
        odds75 = np.exp(coef75)
        
        # This dataframe summarizes the coefficients and odds ratios for each predictor
        # in the trained model. A unit change in a predictor increases the likelihood
        # of a grid cell containing a wind farm by the odds ratio's predictor amount.
        dfTrained = DataFrame()
        dfTrained["Predictor"] = dfxConfig.columns
        dfTrained["Odds_Low"] = odds25
        dfTrained["Odds_Med"] = oddsMed
        dfTrained["Odds_Upp"] = odds75
        dfTrained["Coef_Med"] = coefMedTrained
        dfTrained["Rank"] = rankedCoefficients.astype(int)
        dfTrainedSorted = dfTrained.sort_values("Rank", ignore_index = True)
        dfTrainedSortedString = dfTrainedSorted.to_string(justify = 'center', col_space = 15, index = False)
        
        # The dataframe is saved to the console output
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nThe following dataframe summarizes the coefficients and odds ratios "
                      +"\n"+ "obtained from fitting the " + configList[g] + " model to the aggregated dataset. Predictors are "
                      +"\n"+ "ranked by the magnitude of their coefficients to convey strength of association: \n\n", border=0)
        pdf.multi_cell(w=0, h=5.0, align='R', txt= dfTrainedSortedString, border = 0)
        
        # An odds ratio chart is constructed to illustrate the association
        # between each predictor and the logit of the binary dependent variable       
        # This loop adds a column color to the odds ratio dataframe for coloring
        # the bars in the chart below
        for i in range(len(dfTrainedSorted)):
            # Colour of bar chart is set to red if the sales 
            # is < 60000 and green otherwise
            dfTrainedSorted['Colors'] = ['red' if float(
            x) < 1 else 'green' for x in dfTrainedSorted['Odds_Med']]
        
        # A bar chart summarizing the median and IQR of the odds ratios for each
        # predictor across all 30 model runs
        fig, ax = plt.subplots(figsize = (10,10))
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(270)
        # The loop below combines the IQR error bars for each predictor with the 
        # median odds ratio
        n = 0
        for index, i in dfTrainedSorted.iterrows():
            x = [n,n]
            y = [i["Odds_Low"], i["Odds_Upp"]]
            plt.errorbar(x,y,markersize = 10, marker = '|', transform = rot + base, color = "black")
            x = [n]
            y = [i["Odds_Med"]]
            plt.vlines(x, ymin=1, ymax = y, color=dfTrainedSorted["Colors"][n], linewidth = 15, transform = rot + base)
            n += 1
        # y-axis ticks are replaced with the predictor names
        ticks = list(range(0,n,1))
        ticks = [-x for x in ticks]
        plt.title(str("Odds Ratios - " + configList[g] + " Model \n" + studyRegion.region + "_" + farmDensity.density + "_acres_per_MW_" + farmCapacity.capacity + "th_percentile"), fontsize = 20)
        plt.yticks(ticks, labels = dfTrainedSorted["Predictor"].tolist(),fontsize = 14)
        plt.xticks(np.arange(0, max(dfTrainedSorted["Odds_Upp"]) + 0.5, 1), fontsize = 14)
        # A solid line at x=1 is added and the axes limits are set
        plt.axvline(x=1, color = 'black')
        plt.ylim(-n,1)
        plt.xlim(0,max(dfTrainedSorted["Odds_Upp"]) + 0.5)
        plt.xlabel("Odds Ratios", fontsize = 16, weight = "bold")
        # Legend for the bar chart
        green_square = Line2D([], [], color='green', marker='s', linestyle='None',
                                  markersize=16, label='Positive')
        red_square = Line2D([], [], color='red', marker='s', linestyle='None',
                                  markersize=16, label='Negative')
        plt.legend(handles = [red_square,green_square], prop={'size': 20})
        
        # Low-resolution version is created and saved to the console output
        oddsFilepath = "".join([directoryPlusFigures, "/OddsRatio_", configList[g] , "_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, ".png"])
        plt.tight_layout()
        plt.savefig(oddsFilepath, dpi = 50)
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nOdds Ratio chart generated from the 30 " + configList[g] + " model runs with the training data: \n" , border=0)
        pdf.image(oddsFilepath, w = 150, h = 150)
        # The figure is re-saved as a high-resolution version
        plt.tight_layout()
        plt.savefig(oddsFilepath, dpi = 300)
        plt.clf()
    
        ########################## MODEL VALIDATION ##########################
        
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n--------------- MODEL Validation (Testing Data): " + configList[g] + " Configuration ---------------", border=0)
        
        print("ROC construction in progress...")
    
        # First validation task is to construct the model's ROC, for which the
        # curves generated from each of the 30 runs with the testing data are laid
        # on top of each other
        for i in range(len(fprList)):
            plt.rcParams['figure.figsize'] = [10,10]
            plt.plot(fprList[i],tprList[i],label="AUC="+str(aucList[i]))
            plt.xlim(-0.01,1.01)
            plt.ylim(-0.01,1.01)
            plt.ylabel('True Positive Rate', weight = "bold", fontsize = 14)
            plt.xlabel('False Positive Rate', weight = "bold", fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)   
            
        # A straight line corresponding to AUC = 0.5 is added to the ROC plot
        plt.plot([-0.1,1.1], [-0.1,1.1], "--k", alpha = 0.5)
        # The minimum, median, and maximum area under curve statistics obtained from
        # the 30 ROC curves are assigned to variables and addded to the ROC plot
        aucMin = min(aucList)
        aucMed = np.median(aucList)
        aucMax = max(aucList)
        plt.title(str("ROC Curve - " + configList[g] + " Model \n" + studyRegion.region + "_" + farmDensity.density + "_acres_per_MW_" + farmCapacity.capacity + "th_percentile"), fontsize = 20)
        plt.text(0.6,0.21,'Maximum AUC: ' + str(aucMax)[:5], fontsize = 20)
        plt.text(0.6,0.17,'Median AUC: ' + str(aucMed)[:5], fontsize = 20)
        plt.text(0.6,0.13,'Minimum AUC: ' + str(aucMin)[:5], fontsize = 20)
        plt.text(0.6,0.06,'Median Thresh: ' + str(np.median(thresholdList))[:5], fontsize = 20)
        
        # Low-resolution version is created and saved to the console output
        rocFilepath = "".join([directoryPlusFigures,"/ROC_", configList[g], "_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, ".png"])
        plt.tight_layout()
        plt.savefig(rocFilepath, dpi = 50)
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nROC curves generated from the 30 " + configList[g] + " model runs with the testing data: \n" , border=0)
        pdf.image(rocFilepath, w = 150, h = 150)
        # The figure is re-saved as a high-resolution version
        plt.tight_layout()
        plt.savefig(rocFilepath, dpi = 300)
        plt.clf()
        
        # The Area Under Curve statistics obtained from the 30 tested model runs
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nRange of Area Under Curve (AUC) statistics for the " + configList[g] + " model: "
                      +"\n"+ "Minimum AUC: " + str(aucMin)
                      +"\n"+ "Median AUC: " + str(aucMed)
                      +"\n"+ "Maximum AUC: " + str(aucMax), border=0)
    
        # The range of optimal threshold classifications from these same 30 ROC curves
        threshMin = min(thresholdList)
        threshMed = np.median(thresholdList)
        threshMax = max(thresholdList)
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nRange of optimal threshold classifications for the " + configList[g] + " model: "
                      +"\n"+ "Minimum Threshold: " + str(threshMin)
                      +"\n"+ "Median Threshold: " + str(threshMed)
                      +"\n"+ "Maximum Threshold: " + str(threshMax), border=0)
        
        print("Confusion Matrix construction in progress...")

        # The median, lower quartile, and upper quartile confusion matrices across 
        # the 30 tested model runs are identified, first by computing the
        # prediction accuracy of each matrix ((true positive + true negative)/total)
        accuracyList = [(cmList[i][0][0]+cmList[i][1][1])/sum(cmList[i].flatten()) for i in range(len(cmList))]
           
        # Accuracies of the lower quartile, median, and upper quartile matrices
        lowerPerc = np.percentile(accuracyList, 25, interpolation = 'nearest')
        medianPerc = np.percentile(accuracyList, 50, interpolation = 'nearest')
        upperPerc = np.percentile(accuracyList, 75, interpolation = 'nearest')
        
        # Indexes of these three accuracies
        lowerIdx = accuracyList.index(lowerPerc)
        medianIdx = accuracyList.index(medianPerc)
        upperIdx = accuracyList.index(upperPerc)
        
        # Indexes are used to identify the respective confusion matrices
        cm25 = cmList[lowerIdx]
        cmMed = cmList[medianIdx]
        cm75 = cmList[upperIdx]
    
        # The confusion matrix averaged for all 30 tested model runs (median) is created
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cmMed)
        plt.title(str("Confusion Matrix - " + configList[g] + " Model \n" + studyRegion.region + "_" + farmDensity.density + "_acres_per_MW_" + farmCapacity.capacity + "th_percentile"), fontsize = 20, pad = 20)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('No Expected\nWind Farm', 'Expected\nWind Farm'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('No Observed\nWind Farm', 'Observed\nWind Farm'))
        ax.tick_params(labelsize = 14)
        ax.matshow(cmMed, cmap = "rainbow")
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cmMed[i, j], ha='center', va='center', color='white',fontsize=28,weight='bold')
        plt.text(-0.35,1.6, str(medianPerc*100)[:5] + "% of grid cell states were predicted correctly.", fontsize = 20)
        
        # A low-resolution version of the median confusion matrix is created and
        # saved to the console output
        matrixFilepath = "".join([directoryPlusFigures, "/Matrix_", configList[g], "_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, ".png"])
        plt.tight_layout()
        plt.savefig(matrixFilepath, dpi = 50)
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nMedian Confusion Matrix of the " + configList[g] + " model's predictive accuracy: \n" , border=0)    
        pdf.image(matrixFilepath, w = 150, h = 150)
        
        # The confusion matrix is re-saved as a high-resolution version
        plt.tight_layout()
        plt.savefig(matrixFilepath, dpi = 300)
        plt.clf()
    
        # The range of confusion matrices obtained from the 30 model runs with
        # the testing data is added to the console output.
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n\nBelow are the range of confusion matrix results from the "
                      +"\n"+ "30 " + configList[g] + " model runs with the testing data: "
                      +"\n\n"+ "Lower Quartile confusion matrix: \n" + str(cm25)
                      +"\n"+ "Lower Quartile proportion of correctly predicted grid cell states by the " + configList[g] + " model: " + str(lowerPerc)
                      +"\n\n"+ "Median confusion matrix: \n" + str(cmMed)
                      +"\n"+ "Median proportion of correctly predicted grid cell states by the " + configList[g] + " model: " + str(medianPerc)
                      +"\n\n"+ "Upper Quartile confusion matrix: \n" + str(cm75)
                      +"\n"+ "Upper Quartile proportion of correctly predicted grid cell states by the " + configList[g] + " model: " + str(upperPerc), border=0)    
        
        ########################## BOXPLOT CONSTRUCTION ##########################
        
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n--------------- BOXPLOT CONSTRUCTION (All Data): " + configList[g] + " Configuration ---------------", border=0)
        
        # The final part of the model's validation is boxplot construction, done to
        # validate that the grid cells correctly predicted to contain (not contain)
        # wind farms are indeed those that were assigned the highest (lowest) 
        # probabilities, applied to grid cells rather than just the training or
        # testing data
        
        # List to hold the probability that each grid cell contains a wind farm
        probabilityList = []
        probabilityAppend = probabilityList.append
        
        # List to hold the predicted binary wind farm existence
        cellStateList = []
        cellStateAppend = cellStateList.append
                
        # The loop iterates through every single grid cell
        for cell in range(len(dfy)):
            
            # Probability that each grid cell should contain a wind farm is computed
            # and retained
            prob = 1/(1+e**-(interceptList[0][0]+sum(coefMedTrained[0:len(dfxArray[0])]*dfxArray[cell][0:len(dfxArray[0])])))
            probabilityAppend(prob)
            
            # Probability is converted into a binary outcome, based on the median
            # classification threshold and the actual binary wind farm existence
            if prob >= threshMed and dfy[cell] == 1:
                cellStateAppend("True_Pos")
            elif prob >= threshMed and dfy[cell] == 0:
                cellStateAppend("False_Pos")
            elif prob < threshMed and dfy[cell] == 1:
                cellStateAppend("False_Neg")
            elif prob < threshMed and dfy[cell] == 0:
                cellStateAppend("True_Neg")
        
        # The results of executing the trained and tested model over all grid cells
        # in the study area are added to the console output
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nGrid cell classifications from executing the trained and tested " + configList[g] + " model "
                      +"\n"+ "over all grid cells in " + studyRegion.region + ":"
                      +"\n\n"+ "Number of True Positive Grid Cells: " + str(sum(i == "True_Pos" for i in cellStateList))
                      +"\n"+ "Number of False Positive Grid Cells: " + str(sum(i == "False_Pos" for i in cellStateList))
                      +"\n"+ "Number of True Negative Grid Cells: " + str(sum(i == "True_Neg" for i in cellStateList))
                      +"\n"+ "Number of False Negative Grid Cells: " + str(sum(i == "False_Neg" for i in cellStateList)), border=0)
     
        # The probabilities associated with each of the four grid cell states will
        # be saved to separate lists
        falsePositiveList = []
        falsePositiveAppend = falsePositiveList.append
        truePositiveList = []
        truePositiveAppend = truePositiveList.append
        falseNegativeList = []
        falseNegativeAppend = falseNegativeList.append
        trueNegativeList = []
        trueNegativeAppend = trueNegativeList.append
        
        # Assignment of probabilities to the lists above
        for i in range(len(probabilityList)):
            if cellStateList[i] == "False_Pos":
                falsePositiveAppend(probabilityList[i])
            if cellStateList[i] == "True_Pos":
                truePositiveAppend(probabilityList[i])
            if cellStateList[i] == "False_Neg":
                falseNegativeAppend(probabilityList[i])
            if cellStateList[i] == "True_Neg":
                trueNegativeAppend(probabilityList[i])
        
        print("Boxplot construction in progress...")

        # A boxplot is created for each of the four lists of probabilities
        plt.figure(figsize = (10,10))
        plt.boxplot([truePositiveList,falsePositiveList,trueNegativeList,falseNegativeList], showfliers = True, whis = 1.5,
                    boxprops = dict(linewidth = 2), medianprops = dict(linewidth = 2), whiskerprops = dict(linewidth = 2))
        plt.title("Boxplots - " + configList[g] + " Model \n" + studyRegion.region + "_" + farmDensity.density + "_acres_per_MW_" + farmCapacity.capacity + "th_percentile", fontsize = 20, pad = 20)
        plt.ylim(-0.1,1.1)
        plt.ylabel("Probability of Wind Farm Existence", fontsize = 14)
        plt.axhline(threshMed, linestyle = 'dashed', color = "blue", alpha = 0.7, linewidth = 4)
        plt.axvline(2.5, color = "black")
        plt.xticks([1,2,3,4],["".join(["True Positive\n(", str(len(truePositiveList)), ")"]),
                              "".join(["False Positive\n(", str(len(falsePositiveList)), ")"]),
                              "".join(["True Negative\n(", str(len(trueNegativeList)), ")"]),
                              "".join(["False Negative\n(", str(len(falseNegativeList)), ")"])], fontsize = 14)
        plt.yticks(fontsize = 14)
               
        # A Mann-Whitney U-test is performed to determine whether the difference in 
        # median probability of grid cells classed as positive, and those classed as
        # negative, is statistically significant (p < 0.05)
        mannWhitNegative = mannwhitneyu(falseNegativeList,trueNegativeList,alternative = "two-sided")
        mannWhitPositive = mannwhitneyu(falsePositiveList,truePositiveList,alternative = "two-sided")
        
        # Text to be added to the boxplot summarizing the medians and the Mann-Whitney
        # U-test results. The code will break if there were no grid cells in a 
        # particular category, so this is accounted for
        # False Positive
        if len(falsePositiveList) == 0:
            medFalsePos = "Median False Pos: N/A"
        else:
            medFalsePos = "Median False Pos: " + str(median(falsePositiveList))[:5]
        # True Positive
        if len(truePositiveList) == 0:
            medTruePos = "Median True Pos: N/A"
        else:    
            if mannWhitPositive[1] < 0.05:
                medTruePos = "Median True Pos: " + str(median(truePositiveList))[:5] + "*"
            else:
                medTruePos = "Median True Pos: " + str(median(truePositiveList))[:5]
        # False Negative
        if len(falseNegativeList) == 0:
            medFalseNeg = "Median False Neg: N/A"
        else:
            medFalseNeg = "Median False Neg: " + str(median(falseNegativeList))[:5]
        # True Negative
        if len(trueNegativeList) == 0:
            medTrueNeg = "Median True Neg: N/A"
        else:
            if mannWhitNegative[1] < 0.05:
                medTrueNeg = "Median True Neg: " + str(median(trueNegativeList))[:5] + "*"
            else:
                medTrueNeg = "Median True Neg: " + str(median(trueNegativeList))[:5]
        
        # The text is added to the boxplot as a legend
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        firstLegend = plt.legend([extra,extra,extra,extra],(medTruePos,medFalsePos,medTrueNeg,medFalseNeg), loc = 1, fontsize = 14, frameon = False)
        
        # The median threshold (dashed blue line) is also added as a legend
        blueDash = Line2D([], [], color='blue', linestyle='--', linewidth = 4,
                                  markersize=16, label='Med. Thresh: \n' + str(np.median(thresholdList))[:5])
        # The first legend is added to this one
        plt.gca().add_artist(firstLegend)
        plt.legend(handles = [blueDash], prop={'size': 20}, loc = "lower left", bbox_to_anchor = (-0.05,-0.2))
                   
        # A low-resolution version of the boxplot is saved to the console output
        boxplotFilepath = "".join([directoryPlusFigures, "/Boxplot_", configList[g], "_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, ".png"])
        plt.tight_layout()
        plt.savefig(boxplotFilepath, dpi = 50)
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n\nBoxplot of grid cell probabilities in each classification: \n", border=0)
        pdf.image(boxplotFilepath, w = 150, h = 150)
        
        # The boxplot is re-saved as a high-resolution version
        plt.tight_layout()
        plt.savefig(boxplotFilepath, dpi = 300)
        plt.clf()
        
        # Median probabilities of the four grid cell classifications 
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nMedian probabilities of wind farm existence for each grid cell classification."
                      +"\n"+ "An asterisk indicates a Mann-Whitney U-test result that is statistically significant (p<0.05): "
                      +"\n\n"+ medFalsePos
                      +"\n"+ medTruePos
                      +"\n"+ medFalseNeg
                      +"\n"+ medTrueNeg, border=0)
        
        # The results of the Mann-Whitney U-tests performed on the four grid cell classifications
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nMann-Whitney U-test results: "
                      +"\n\n"+ "Mann-Whitney Statistic - True Positive vs False Positive: " 
                      +"\n"+ "U-statistic = " + str(mannWhitPositive[0]) 
                      +"\n"+ "p-value = " + str(mannWhitPositive[1])
                      +"\n\n"+ "Mann-Whitney Statistic - True Negative vs False Negative: "
                      +"\n"+ "U-statistic = " + str(mannWhitNegative[0]) 
                      +"\n"+ "p-value = " + str(mannWhitNegative[1]), border=0)
        
        print("Plot construction complete.")

        ######################### MAP CONSTRUCTION ############################
        
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n\n------------------ MAP CONSTRUCTION: " + configList[g] + " Configuration ------------------", border=0)
    
        print("\nHexagonal grid map construction in progress...")
    
        # This conditional statement will create the map for the selected state
        if studyRegion.region != "CONUS": 
            # If the feature class and geodatabase created by this part of the code
            # already exist from a previous run, the geodatabase is emptied
            if os.path.exists("".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb"])) is True:
                arcpy.Delete_management("".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], "_Map"]))
                arcpy.Delete_management("".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb\Attribute_Table"]))
            # An empty geodatabase is created to hold the new map.
            # NOTE: Make sure a folder called "Wind_Farm_Predictor_Maps" has been
            # created in the directory before executing the model.
            else:
                arcpy.CreateFileGDB_management(directoryPlusSurfaces, "".join(["Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb"]))
            
            # The aggregated dataset is used to define the grid cell locations
            # for this map, first by adding the dataset to the new geodatabase
            inputFeature = "".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_Merged.gdb/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region + "_Merged"])
            outGDB = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb"])
            arcpy.FeatureClassToGeodatabase_conversion(inputFeature,outGDB)
                    
            # Centroids are created over the domain of the grid cells, which are clipped
            # to the shape of the state
            centroids = arcpy.FeatureToPoint_management(inputFeature, inputFeature + "_Centroids", "CENTROID")
            border = "".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_Merged.gdb/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region + "_Merged"])
            cellCentroids = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_Centroids"])
            arcpy.Clip_analysis(centroids,border,cellCentroids)
            
            # Empty probability and cell state fields are added to the centroid's
            # attribute table
            arcpy.AddField_management(cellCentroids, "Probab", "DOUBLE")
            arcpy.AddField_management(cellCentroids,"Cell_State", "TEXT")
            
            # The probability and cell state fields are filled
            fields = ["Probab","Cell_State"]
            iterator1 = iter(probabilityList)
            iterator2 = iter(cellStateList)
            with UpdateCursor(cellCentroids,fields) as cursor:
                for row in cursor:
                    try:
                        row[0] = next(iterator1)
                        row[1] = next(iterator2)
                        cursor.updateRow(row)
                    except:
                        continue
            # Filepath to the empty hexagonal grid
            gridCells = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region + "_Merged"])
            
            # Filepath to the hexgonal grid once it is filled with the data attached
            # to the centroids
            finalGrid = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_", configList[g], "_Map"])
            
            # Desired fields from combining the centroids and the hexagonal grid
            # are specified, and the two are spatially joined
            fmWindTurbines = arcpy.FieldMappings()
            fmWindTurbines.addTable(cellCentroids)
            windTurbineFields = ["Probab","Cell_State"]
            for field in fmWindTurbines.fields:
                if field.name not in windTurbineFields:
                    fmWindTurbines.removeFieldMap(fmWindTurbines.findFieldMapIndex(field.name))
            arcpy.SpatialJoin_analysis(gridCells,cellCentroids,finalGrid, field_mapping = (fmWindTurbines))
        
            # Rows with empty field values are deleted
            with UpdateCursor(finalGrid, "Probab") as cursor:
                for row in cursor:
                      if row[0] is None:
                        cursor.deleteRow()
        
            # Unwanted fields are deleted
            arcpy.DeleteField_management(finalGrid, ["Join_Count","TARGET_FID"])
            
            # The separate features used to perform the spatial join are deleted
            arcpy.Delete_management(cellCentroids)
            arcpy.Delete_management(gridCells)
            
            # Grid cell centroids for the aggregated dataset can also be deleted
            arcpy.Delete_management("".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_Merged.gdb/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region + "_Merged_Centroids"]))
            
            # Filepath to the constructed hexagonal grid map
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nFilepath to the constructed hexagonal grid map: "
                          +"\n\n"+finalGrid, border=0)
        
        # Same but for the CONUS
        else:
            # If the feature class and geodatabase created by this part of the code
            # already exist from a previous run, the geodatabase is emptied
            if os.path.exists("".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb"])) is True:
                arcpy.Delete_management("".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], "_Map"]))
                arcpy.Delete_management("".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb\Attribute_Table"]))

            # An empty geodatabase is created to hold the new map.
            # NOTE: Make sure a folder called "Wind_Farm_Predictor_Maps" has been
            # created in the directory before executing the model.
            else:
                arcpy.CreateFileGDB_management(directoryPlusSurfaces, "".join(["Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb"]))
            
            # The aggregated dataset is used to define the grid cell locations
            # for this map, first by adding the dataset to the new geodatabase
            inputFeature = "".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_Merged.gdb/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_Merged"])
            outGDB = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb"])
            arcpy.FeatureClassToGeodatabase_conversion(inputFeature,outGDB)
                    
            # Centroids are created over the domain of the grid cells, which are clipped
            # to the shape of the state
            centroids = arcpy.FeatureToPoint_management(inputFeature, inputFeature + "_Centroids", "CENTROID")
            border = "".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_", studyRegion.region, "_Merged.gdb/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_Merged"])
            cellCentroids = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_Centroids"])
            arcpy.Clip_analysis(centroids,border,cellCentroids)
            
            # Empty probability and cell state fields are added to the centroid's
            # attribute table
            arcpy.AddField_management(cellCentroids, "Probab", "DOUBLE")
            arcpy.AddField_management(cellCentroids,"Cell_State", "TEXT")

            # The probability and cell state fields are filled
            fields = ["Probab","Cell_State"]
            iterator1 = iter(probabilityList)
            iterator2 = iter(cellStateList)
            with UpdateCursor(cellCentroids,fields) as cursor:
                for row in cursor:
                    try:
                        row[0] = next(iterator1)
                        row[1] = next(iterator2)
                        cursor.updateRow(row)
                    except:
                        continue
            
            # Filepath to the empty hexagonal grid
            gridCells = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_Merged"])
            
            # Filepath to the hexgonal grid once it is filled with the data attached
            # to the centroids
            finalGrid = "".join([directoryPlusSurfaces, "/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], ".gdb\Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_", configList[g], "_Map"])
            
            # Desired fields from combining the centroids and the hexagonal grid
            # are specified, and the two are spatially joined
            fmWindTurbines = arcpy.FieldMappings()
            fmWindTurbines.addTable(cellCentroids)
            windTurbineFields = ["Probab","Cell_State"]
            for field in fmWindTurbines.fields:
                if field.name not in windTurbineFields:
                    fmWindTurbines.removeFieldMap(fmWindTurbines.findFieldMapIndex(field.name))
            arcpy.SpatialJoin_analysis(gridCells,cellCentroids,finalGrid, field_mapping = (fmWindTurbines))
            
            # Rows with empty field values are deleted
            with UpdateCursor(finalGrid, "Probab") as cursor:
                for row in cursor:
                      if row[0] is None:
                        cursor.deleteRow()
            
            # Unwanted fields are deleted
            arcpy.DeleteField_management(finalGrid, ["Join_Count","TARGET_FID"])
            
            # The separate features used to perform the spatial join are deleted
            arcpy.Delete_management(cellCentroids)
            arcpy.Delete_management(gridCells)
            
            # Grid cell centroids for the aggregated dataset can also be deleted
            arcpy.Delete_management("".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_Merged.gdb/Hexagon_Grid_", farmDensity.density, "_acres_per_MW_", farmCapacity.capacity, "th_percentile_CONUS_Merged_Centroids"]))
            
            # Filepath to the constructed hexagonal grid map
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nFilepath to the constructed hexagonal grid map: "
                          +"\n\n"+finalGrid, border=0)
        
        # The Getis-Ord (Gi*) Statistic are computed to identify statistically 
        # significant (p < 0.05) clusters of high-probability (true positive or false
        # positive) grid cells
        getisOrd = finalGrid + "_Getis_Ord"
        arcpy.HotSpots_stats(finalGrid, "Probab", getisOrd, "INVERSE_DISTANCE", "EUCLIDEAN_DISTANCE")
    
        # The z-score and p-value obtained for each grid cell using the Gi* statistic
        # is added to the final gridded surface
        arcpy.JoinField_management(finalGrid,"OBJECTID",getisOrd,"SOURCE_ID",fields = ["GiZScore","GiPValue"])
        
        # The total number of grid cells containing wind farms that exist in
        # statistically significant clusters (p < 0.05) are counted
        totalPosCount = 0
        truePosCount = 0
        falsePosCount = 0
        with SearchCursor(finalGrid,["Cell_State","GiPvalue"]) as cursor:
            for row in cursor:
                if row[1] < 0.05:
                    totalPosCount += 1
                if row[0] == "True_Pos" and row[1] < 0.05:
                    truePosCount += 1
                if row[0] == "False_Pos" and row[1] < 0.05:
                    falsePosCount += 1
    
        # Filepath to the constructed hexagonal grid map
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nTotal (Percentage) of all grid cells over " + studyRegion.region + " that exist in hotspots: "
                      +"\n"+ str(totalPosCount) + " (" + str(round(totalPosCount/len(cellStateList)*100,2)) + "%)"
                      +"\n"+ "Total (Percentage) True Positive grid cells over " + studyRegion.region + " that exist in hotspots: "
                      +"\n"+ str(truePosCount) + " (" + str(round(truePosCount/sum(i == "True_Pos" for i in cellStateList)*100,2)) + "%)"
                      +"\n"+ "Total (Percentage) False Positive grid cells over " + studyRegion.region + " that exist in hotspots: "
                      +"\n"+ str(falsePosCount) + " (" + str(round(falsePosCount/sum(i == "False_Pos" for i in cellStateList)*100,2)) + "%)", border=0)

        # A new field is added to hold the grid cell states and their existence in
        # statistically significant clusters within one variable
        arcpy.AddField_management(finalGrid,"State_Sig", "TEXT")
        
        # Field values are assigned
        with UpdateCursor(finalGrid,["Cell_State","GiPValue","State_Sig"]) as cursor:
            for row in cursor:
                if row[0] == "True_Pos" and row[1] < 0.05:
                    row[2] = "True_Pos_Clust"
                if row[0] == "True_Pos" and row[1] >= 0.05:
                    row[2] = "True_Pos"
                if row[0] == "False_Pos" and row[1] < 0.05:
                    row[2] = "False_Pos_Clust"
                if row[0] == "False_Pos" and row[1] >= 0.05:
                    row[2] = "False_Pos"
                if row[0] == "True_Neg":
                    row[2] = "True_Neg"
                if row[0] == "False_Neg":
                    row[2] = "False_Neg"
                cursor.updateRow(row) 
    
        # The getis-ord map is no longer needed since it has been combined with
        # the grid cell probabilities
        arcpy.Delete_management(getisOrd)
        
        # The attribute table is saved separately within the geodatabase
        arcpy.TableToTable_conversion(finalGrid, outGDB, "Attribute_Table")
        
        print("\nConstruction of the map using the " + configList[g] + " logistic regression model: Complete.")
    
    # Console output is written to a PDF
    pdf.output(directory + '/Console_Output.pdf', 'F')
    
LogisticRegressionModel()
