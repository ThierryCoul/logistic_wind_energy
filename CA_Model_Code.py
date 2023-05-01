# Script Name: Cellular_Automata
# Author: Joshua (Jay) Wimhurst
# Date created: 2/27/2023
# Date Last Edited: 4/30/2023

############################### DESCRIPTION ###################################
# This script constructs a cellular automaton for grid cell states predicted
# by a logistic regression model of wind farm site suitability across the CONUS
# (and individual states). The cellular automaton steps forward the grid cells
# in time (5-year timesteps from 2020 to 2050), in order to project grid cells
# that could gain a commercial wind farm over the next few decades. The fitted
# median coefficients and intercept from running the logistic regression model
# are used as a starting point, with the cellular automaton providing the
# option to modify coefficients for different scenarios of future wind farm
# development (e.g., wind speed increasing, more positive attitude toward
# wind energy technologies). The primary output is a map showing the location
# of current and future wind farm locations in a given state or across the
# CONUS. Four different versions of the model can be executed: all predictors,
# all predictors but wind speed, wind speed only, and a refined predictor set.
###############################################################################

import arcpy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from arcpy.da import TableToNumPyArray, UpdateCursor, SearchCursor
from fpdf import FPDF
from math import sin, radians, ceil, e
from numpy import sqrt
from pandas import DataFrame
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

######################### DIRECTORY CONSTRUCTION ##########################

# IMPORTANT - set up a file directory to place the Console Output, 
# generated figures, and constructed WiFSS surfaces BEFORE attempting to
# run this model
directory = r"D:\Dissertation_Resources\Model_Testing"

# IMPORTANT - make sure to also create a folder for the WiFSS surfaces
# created from applying the trained and tested model to all grid cells
directoryPlusSurfaces = directory + "\WiFSS_Surfaces"

# IMPORTANT - make sure to also create a folder for the WiFSS surfaces
# created from projecting the gridded surface out to the year 2050
directoryPlusFutureSurfaces = directory + "\WiFSS_Future_Surfaces"

# IMPORTANT - make sure to also create a folder for the QADI tables created
# from quantifying differences between projected grid cell surfaces (and a 
# sub-folder named after the state selected or CONUS)
directoryPlusQADI = directory + "\QADI_Tables"

# IMPORTANT - make sure to also create a folder for the coefficients
# obtained from fitting the model (and a sub-folder named after the state
# selected or CONUS), which will be needed for users running
# the cellular automaton portion of the model
directoryPlusCoefficients = directory + "\Coefficients"

# IMPORTANT - make sure to also create a folder for the intercepts
# obtained from fitting the model (and a sub-folder named after the state
# selected or CONUS), which will be needed for users running
# the cellular automaton portion of the model
directoryPlusIntercepts = directory + "\Intercepts"

# IMPORTANT - make sure to also create a folder for the constraints and 
# neighborhood effects that are defined by the model user 
directoryPlusConstraints = directory + "\Defined_Constraints"
directoryPlusNeighborhoods = directory + "\Defined_Neighborhoods"

# IMPORTANT - make sure to also create a folder that contains the results of 
# applying constraints and neighborhood effects to all grid cells in a 
# given study domain 
directoryPlusConstraintsAndNeighborhoods = directory + "\Constraints_and_Neighborhood_Effects"

# The script prints the console output to a text file, with a previous
# version deleted prior to the model run
if os.path.exists(directory + "\Cellular_Automata_Console_Output.txt"):
    os.remove(directory + "\Cellular_Automata_Console_Output.txt")

####################### DATASET SELECTION AND SETUP ###########################

# PDF file containing the console output is initiated and caveat is added
pdf = FPDF()
pdf.add_page()
pdf.set_xy(4, 4)
pdf.set_font(family = 'arial', size = 13.0)
pdf.multi_cell(w=0, h=5.0, align='R', txt="Console output", border = 0)
pdf.multi_cell(w=0, h=5.0, align='L', 
              txt="------------------ DATASET SELECTION AND SETUP ------------------"
                  +'\n\n'+ "NOTE: The Logistic_Regression_Model.py script MUST be executed "
                  +'\n'+ "prior to running the Cellular_Automata_Model.py script."
                  +'\n'+ "If one wishes to execute the model over states that "
                  +'\n'+ "contain zero commercial wind farms (Louisiana, Mississippi "
                  +'\n'+ "Alabama, Georgia, South Carolina, Kentucky), states that possess "
                  +'\n'+ "wind farms in only one grid cell at all but the highest spatial "
                  +'\n'+ "resolutions (Arkansas, Florida, Virginia, Delaware, Connecticut,"
                  +'\n'+ "New Jersey, Tennessee), or states at low spatial resolutions at which "
                  +'\n'+ "too many predictors were removed due to collinearity (Rhode Island "
                  +'\n'+ "at the 100th or 80th percentile), the Logistic_Regression_Model.py "
                  +'\n'+ "script must be executed for the CONUS.\n", border=0)

# Select the study region for which the cellular automaton will be constructed,
# CONUS or a single state.
def studyRegion(values, message):
    while True:
        x = input(message) 
        if x in values:
            studyRegion.region = x
            break
        else:
            print("Invalid value; options are " + str(values))
studyRegion(["CONUS","Alabama","Arizona","Arkansas","California","Colorado",
             "Connecticut","Delaware","Florida","Georgia","Idaho","Illinois",
             "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
             "Massachusetts","Michigan","Minnesota","Mississippi","Missouri",
             "Montana","Nebraska","Nevada","New_Hampshire","New_Mexico",
             "New_Jersey","New_York","North_Carolina","North_Dakota","Ohio",
             "Oklahoma","Oregon","Pennsylvania","Rhode_Island","South_Carolina",
             "South_Dakota","Tennessee","Texas","Utah","Vermont","Virginia",
             "Washington","West_Virginia","Wisconsin","Wyoming"], 
             '''Enter desired study region. \n(CONUS, Alabama, Arizona, Arkansas, '''
             '''California, Colorado, Connecticut, Delaware, Florida, Georgia, '''
             '''Idaho, Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, '''
             '''Maine, Maryland, Massachusetts, Michigan, Minnesota, Mississippi, '''
             '''Missouri, Montana, Nebraska, Nevada, New_Hampshire, New_Jersey, '''
             '''New_Mexico, New_York, North_Carolina, North_Dakota, Ohio, '''
             '''Oklahoma, Oregon, Pennsylvania, Rhode_Island, South_Carolina, '''
             '''South_Dakota, Tennessee, Texas, Utah, Virginia, Vermont, '''
             '''Washington, West_Virginia, Wisconsin, Wyoming):\n''')

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

# Based on selected percentile, the print-out to the console output changes
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

# User inputs are added to the console output
pdf.multi_cell(w=0, h=5.0, align='L', 
              txt="\nSpecified study region: " + str(studyRegion.region)
                  +'\n'+"Specified wind farm density: " + str(farmDensity.density) + " acres/MW"
                  +'\n'+"Specified wind power capacity: " + str(farmCapacity.capacity)  + "th percentile (" + power + ")", border=0)

############### SETTING OF CONSTRAINTS AND NEIGHBORHOOD EFFECTS ###############

# Function call for the computation of grid cells' constraints and 
# neighborhood effects
def ConstraintNeighborhood():
    
    ####################### SETTING UP THE GEODATABASE ########################
    
    # The study area, capacity, and density selected by the end-user are now 
    # defined as global variables to be used in this section of the code
    ConstraintNeighborhood.studyArea = studyRegion.region
    ConstraintNeighborhood.capacity = farmCapacity.capacity
    ConstraintNeighborhood.density = farmDensity.density

    # The computation of constraints and neighborhood effects may not be necessary
    # if the user does not wish to redefine those that have already been made.
    # This condition gives the user the option of skipping this function.
    if os.path.exists("".join([directoryPlusConstraintsAndNeighborhoods + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"])) is True:
        
        # If the filepath exists, then constraints and neighborhood effects 
        # have been previously computed. The user is notified of what they are.
        dfConstraints = pd.read_csv("".join([directoryPlusConstraints + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"]))
        dfNeighborhoods = pd.read_csv("".join([directoryPlusNeighborhoods + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"]))
        
        print("".join(["\nConstraints for ", ConstraintNeighborhood.studyArea, " at ", ConstraintNeighborhood.density, " acres per MW and for ", ConstraintNeighborhood.capacity, "th percentile wind farms have been defined previously:"]))
        for j in range(len(dfConstraints)):
            print("-", dfConstraints["Constraints"][j])
            
        print("".join(["\nA hexagonal neighborhood range of ", str(dfNeighborhoods["Neighborhood"][0]) ," grid cell(s) has also been previously defined."]))
             
        def remake(values, message):
            while True:
                x = input(message)
                if x in values:
                    remake.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
            # This will be needed for recalculating neighborhood effects after
            # each iteration of the cellular automaton
            ConstraintNeighborhood.remake = remake.YesOrNo
        remake(["Y","N"],"".join(["\nWould you like to redefine the constraints and/or neighborhood effects? Y or N:\n"]))

        # If the user does not wish to redo these computations, this
        # function is skipped
        if remake.YesOrNo == "N":
            ConstraintNeighborhood.neighborhoodSize = dfNeighborhoods["Neighborhood"][0]
            ConstraintNeighborhood.ConstYesNo = "N"
            ConstraintNeighborhood.NeighYesNo = "N"
            
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nThe constraints and neighborhood effects from the previous model run "
                              +'\n'+"have not been changed. A hexagonal neighborhood range of " + str(dfNeighborhoods["Neighborhood"][0]) 
                              +'\n'+"grid cell(s) is thus retained, along with the following constraints: "
                              +'\n'+ str(dfConstraints["Constraints"].tolist()), border=0)
            return
        # If constraints and neighborhood effects have not been made before
        # then the remake variable defaults to "Y" (yes)
        else:
            ConstraintNeighborhood.remake = "Y"       
            
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nThe constraints and neighborhood effects will be modified by the user.", border=0)
    
    # If the geodatabase does not exist yet, then the desire to reset the
    # constraints and neighborhood effects is automatically set to "N" (no)
    else:
        ConstraintNeighborhood.remake = "N"
        
    # Calculation of constraints and neighborhood effects if the user
    # selected CONUS
    if ConstraintNeighborhood.studyArea == "CONUS":
        # Filepath to the map containing predicted and actual wind farm locations,
        # used to check whether a logistic regression model run for the CONUS
        # has been done for at least one predictor configuration. Also acts as 
        # a dummy file for separating the predictors to be used as constraints 
        # for future wind farm sites
        if os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Full.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Full.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Full_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_No_Wind.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_No_Wind.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_No_Wind_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Wind_Only.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Wind_Only.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Wind_Only_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Reduced.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Reduced.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Reduced_Map"])
        else:
            print("\nA logistic regression model has not yet been fitted for the user's desired grid cell size and/or study area. The script is aborted.")
            sys.exit()
                
        # If the geodatabase for this particular study region and resolution exists,
        # its contents are emptied first
        if os.path.exists("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb"])) is True:
            arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Constraints_Neighborhoods"]))
            arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb\Attribute_Table"]))
        
        # If the geodatabase does not exist yet, it is made
        # NOTE: Make sure a folder called "Constraints_and_Neighborhood_Effects"
        # has been created in the directory before executing the model.
        else:
            arcpy.CreateFileGDB_management(directoryPlusConstraintsAndNeighborhoods, "".join(["Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb"]))
                
        # A copy of the predicted and actual wind farm locations is added 
        # to the geodatabase
        presentCopy = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Constraints_Neighborhoods_Interim"])
        arcpy.Copy_management(presentLocations, presentCopy)
        
        # The filepath to the aggregated data that constrain
        # wind farm development is set
        constraints = "".join([directory, "/", studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Merged.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Merged"]) 

        # The predicted wind farm locations and aggregated data must be saved
        # into the same gridded surface using a spatial join. The following
        # is the filepath to the constructed gridded surface
        preparedSurface = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Constraints_Neighborhoods"])
        
        # Field mapping is used to select the desired fields prior
        # to the spatial join
        fm = arcpy.FieldMappings()
        fm.addTable(presentCopy)
        fm.addTable(constraints)
        keepers = [# The fields acting as constraints
                   "Avg_Elevat","Avg_Temp","Avg_Wind","Critical","Historical",
                   "Military","Mining","Nat_Parks","Near_Air","Near_Hosp",
                   "Near_Plant","Near_Roads","Near_Sch","Near_Trans",
                   "Trib_Land","Wild_Refug",                   
                   # All other fields
                   "Avg_25","Bat_Count","Bird_Count","Cost_15_19","Dem_Wins",
                   "Dens_15_19","Farm_15_19","Farm_Year","Fem_15_19","Foss_Lobbs",
                   "Gree_Lobbs","Hisp_15_19","In_Tax_Cre","Interconn","ISO_YN",
                   "Net_Meter","Numb_Incen","Numb_Pols","Plant_Year","Prop_15_19",
                   "Prop_Rugg","Renew_Port","Renew_Targ","Rep_Wins","supp_2018",
                   "Tax_Prop","Tax_Sale","Type_15_19","Undev_Land","Unem_15_19",
                   "Whit_15_19",
                   # The wind turbine location field
                   "Wind_Turb"]
        for field in fm.fields:
            if field.name not in keepers:
                fm.removeFieldMap(fm.findFieldMapIndex(field.name))
        # The spatial join is performed
        arcpy.SpatialJoin_analysis(presentCopy,constraints,preparedSurface,field_mapping = (fm),match_option = "HAVE_THEIR_CENTER_IN")
        
        # The temporary files made can now be deleted
        arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Constraints_Neighborhoods_Interim"]))

    # Preparation of the gridded surface if the user did not select CONUS.
    # It was not possible to perform logistic regression in states containing
    # fewer than two grid cells with commercial wind farms in them, due to 
    # the model's training and testing requiring grid cells in both classes
    # of the dependent variable. Computation of constraints and neighborhood
    # effects for these states must be done by clipping predicted wind farm
    # locations out of the CONUS outputs.
    elif (ConstraintNeighborhood.studyArea == "Alabama" or ConstraintNeighborhood.studyArea == "Arkansas" or ConstraintNeighborhood.studyArea == "Connecticut"
    or ConstraintNeighborhood.studyArea == "Delaware" or ConstraintNeighborhood.studyArea == "Florida" or ConstraintNeighborhood.studyArea == "Georgia" 
    or ConstraintNeighborhood.studyArea == "Kentucky" or ConstraintNeighborhood.studyArea == "Louisiana" or ConstraintNeighborhood.studyArea == "Mississippi" 
    or ConstraintNeighborhood.studyArea == "New_Jersey" or ConstraintNeighborhood.studyArea == "Rhode_Island" or ConstraintNeighborhood.studyArea == "South_Carolina" 
    or ConstraintNeighborhood.studyArea == "Tennessee" or ConstraintNeighborhood.studyArea == "Virginia"):                
        # Filepath to the map containing predicted and actual wind farm locations,
        # used to check whether a logistic regression model run for the CONUS
        # has been done for at least one predictor configuration. Also acts as 
        # a dummy file for separating the predictors to be used as constraints 
        # for future wind farm sites
        if os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_85_acres_per_MW_100th_percentile_CONUS_Full.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Full.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Full_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_No_Wind.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_No_Wind.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_No_Wind_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Wind_Only.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Wind_Only.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Wind_Only_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Reduced.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Reduced.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Reduced_Map"])
        else:
            print("\nA logistic regression model has not yet been fitted for the user's desired grid cell size and/or study area. The script is aborted.")
            sys.exit()
         
        # If the geodatabase for this particular study region and resolution exists,
        # its contents are emptied first
        if os.path.exists("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"])) is True:
            arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods"]))
            arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Attribute_Table"]))

        # If the geodatabase does not exist yet, it is made
        # NOTE: Make sure a folder called "Constraints_and_Neighborhood_Effects"
        # has been created in the directory before executing the model.
        else:
            arcpy.CreateFileGDB_management(directoryPlusConstraintsAndNeighborhoods, "".join(["Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"]))
        
        # A copy of the predicted and actual wind farm locations is added 
        # to the geodatabase. Since state-level runs of the logistic regression
        # model were not possible for these states, wind farm locations
        # are derived from the CONUS as a whole
        presentCopy = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Interim"])
        arcpy.Copy_management(presentLocations, presentCopy)
        
        # The grid cells over the CONUS are transformed into points 
        pointsCONUS = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Points"])
        arcpy.FeatureToPoint_management(presentCopy,pointsCONUS)
        
        # State border used to clip the CONUS points
        stateBorder = "".join([directory + "\State_Borders/", ConstraintNeighborhood.studyArea, "/", ConstraintNeighborhood.studyArea])
    
        # Filepath for the clipped points, a copy of the pointsCONUS variable
        pointsCONUSClipped = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Points_Clipped"])
        # Grid cells are clipped
        arcpy.Clip_analysis(pointsCONUS,stateBorder,pointsCONUSClipped)
        
        # Hexagonal grid cells whose centroids are within the state border can
        # now be identified with a feature layer selection. If the feature
        # layers exist from a prior run of the script, they are deleted
        arcpy.Delete_management(["present_Copy_lyr","points_CONUS_Clipped_lyr"])
        arcpy.MakeFeatureLayer_management(presentCopy, "present_Copy_lyr")
        arcpy.MakeFeatureLayer_management(pointsCONUSClipped, "points_CONUS_Clipped_lyr")
        stateCells = arcpy.SelectLayerByLocation_management("present_Copy_lyr", "HAVE_THEIR_CENTER_IN", "points_CONUS_Clipped_lyr")
        
        # Selected features are saved for continued use
        selectedCells = arcpy.FeatureClassToFeatureClass_conversion(stateCells, "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"]), "".join(["Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Selected"]))

        # The filepath to the aggregated data that constrain
        # wind farm development is set. Note that the CONUS constraints
        # must still be used, rather than for the individual state
        constraints = "".join([directory + "/CONUS_Gridded_Surfaces/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Merged.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Merged"]) 
 
        # PresentCopy variable is reassigned
        presentCopy = selectedCells
        
        # The predicted wind farm locations and aggregated data must be saved
        # into the same gridded surface using a spatial join. The following
        # is the filepath to the constructed gridded surface
        preparedSurface = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods"])
        
        # Field mapping is used to select the desired fields prior
        # to the spatial join
        fm = arcpy.FieldMappings()
        fm.addTable(presentCopy)
        fm.addTable(constraints)
        keepers = [# The fields acting as constraints
                   "Avg_Elevat","Avg_Temp","Avg_Wind","Critical","Historical",
                   "Military","Mining","Nat_Parks","Near_Air","Near_Hosp",
                   "Near_Plant","Near_Roads","Near_Sch","Near_Trans",
                   "Trib_Land","Wild_Refug",                   
                   # All other fields
                   "Avg_25","Bat_Count","Bird_Count","Cost_15_19","Dem_Wins",
                   "Dens_15_19","Farm_15_19","Farm_Year","Fem_15_19","Foss_Lobbs",
                   "Gree_Lobbs","Hisp_15_19","In_Tax_Cre","Interconn","ISO_YN",
                   "Net_Meter","Numb_Incen","Numb_Pols","Plant_Year","Prop_15_19",
                   "Prop_Rugg","Renew_Port","Renew_Targ","Rep_Wins","supp_2018",
                   "Tax_Prop","Tax_Sale","Type_15_19","Undev_Land","Unem_15_19",
                   "Whit_15_19",
                   # The wind turbine location field
                   "Wind_Turb"]
        for field in fm.fields:
            if field.name not in keepers:
                fm.removeFieldMap(fm.findFieldMapIndex(field.name))
                # The spatial join is performed
        arcpy.SpatialJoin_analysis(presentCopy,constraints,preparedSurface,field_mapping = (fm),match_option = "HAVE_THEIR_CENTER_IN")

        # The temporary files made can now be deleted
        arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Constraints_Neighborhoods_Interim"]))
        arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_CONUS_Constraints_Neighborhoods_Points"]))
        arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Points_Clipped"]))
        arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Selected"]))

    # For all other states, it is possible to use the predicted wind farm locations
    # produced for them individually
    else:
        # Filepath to the map containing predicted and actual wind farm locations,
        # used to check whether a logistic regression model run for the state
        # has been done for at least one predictor configuration. Also acts as 
        # a dummy file for separating the predictors to be used as constraints 
        # for future wind farm sites
        if os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Full.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Full.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Full_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_No_Wind.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_No_Wind.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_No_Wind_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Wind_Only.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Wind_Only.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Wind_Only_Map"])
        elif os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Reduced.gdb"])) is True:
            presentLocations = "".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Reduced.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Reduced_Map"])
        else:
            print("\nA logistic regression model has not yet been fitted for the user's desired grid cell size and/or study area. The script is aborted.")
            sys.exit()

        # If the geodatabase for this particular study region and resolution exists,
        # its contents are emptied first
        if os.path.exists("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"])) is True:
            arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods"]))
            arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Attribute_Table"]))
        # If the geodatabase does not exist yet, it is made
        # NOTE: Make sure a folder called "Constraints_and_Neighborhood_Effects"
        # has been created in the directory before executing the model.
        else:
            arcpy.CreateFileGDB_management(directoryPlusConstraintsAndNeighborhoods, "".join(["Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"]))
     
        # A copy of the predicted and actual wind farm locations is added to the
        # same folder containing the aggregated data, for their combination
        presentCopy = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Interim"])
        arcpy.Copy_management(presentLocations, presentCopy)
        
        # The filepath to the aggregated data that constrain
        # wind farm development is set
        constraints = "".join([directory + "/" + studyRegion.region, "_Gridded_Surfaces/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Merged.gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Merged"]) 

        # The predicted wind farm locations and aggregated data must be saved
        # into the same gridded surface using a spatial join. The following
        # is the filepath to the constructed gridded surface
        preparedSurface = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods"])
        
        # Field mapping is used to select the desired fields prior
        # to the spatial join
        fm = arcpy.FieldMappings()
        fm.addTable(presentCopy)
        fm.addTable(constraints)
        keepers = [# The fields acting as constraints
                   "Avg_Elevat","Avg_Temp","Avg_Wind","Critical","Historical",
                   "Military","Mining","Nat_Parks","Near_Air","Near_Hosp",
                   "Near_Plant","Near_Roads","Near_Sch","Near_Trans",
                   "Trib_Land","Wild_Refug",                   
                   # All other fields
                   "Avg_25","Bat_Count","Bird_Count","Dem_Wins","Dens_15_19",
                   "Farm_Year","Fem_15_19","Hisp_15_19","ISO_YN","Plant_Year",
                   "Prop_Rugg","supp_2018","Type_15_19","Undev_Land",
                   "Unem_15_19","Whit_15_19",
                   # The wind turbine location field 
                   "Wind_Turb"]
        for field in fm.fields:
            if field.name not in keepers:
                fm.removeFieldMap(fm.findFieldMapIndex(field.name))
        # The spatial join is performed
        arcpy.SpatialJoin_analysis(presentCopy,constraints,preparedSurface,field_mapping = (fm),match_option = "HAVE_THEIR_CENTER_IN")
        
        # The temporary files made can now be deleted
        arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods_Interim"]))

    ######################## SETTING OF MODEL CONSTRAINTS #########################
        
    # The following are the cellular automaton's default constraints for
    # projection of future wind farm sites:
    # No wind farms within 2500 meters of an airport.
    # No wind farms within 10000 meters of an existing power plant.
    # No wind farms within 2500 meters of a hospital or school.
    # No wind farms within 500 meters of, or more than 10000 meters away from, a major road.
    # No wind farms within 250 meters of, or more than 10000 meters away from, a major transmission line.
    # No wind farms in grid cells that average more than 2000 meters above sea level.
    # No wind farms in grid cells with an average wind speed 80 meters above ground less than 4 meters per second.
    # No wind farms in grid cells with an average temperature below 0 degrees Celsius.
    # Wind farm development is prohibited in grid cells shared by
    # military bases, national parks, USFWS critical species habitats,
    # historical landmarks, mining operations, USFWS wildlife refuges,
    # and tribal lands.
    
    # Many wind farms already exist in grid cells that would be prohibited from
    # development based on the default constraints. The number of grid cells
    # that contain wind farms that violate these constraints is counted
    def constraints(airportProhibited, plantProhibited, hospProhibited, schProhibited,
                    roadMinProhibited, roadMaxProhibited, transMinProhibited,
                    transMaxProhibited, elevatProhibited, windProhibited,
                    tempProhibited, militProhibited, natParkProhibited,
                    criticalProhibited, historicProhibited, miningProhibited,
                    wildProhibited, tribalProhibited, windFarmCount,
                    allCellsProhibited, turbineCellsProhibited,
                    allCellsAllowed, turbineCellsAllowed):
        
        # The user is first asked if they wish to switch off the constraints,
        # meaning that nearby grid cells can gain wind farms even if the
        # default constraints are violated
        def constraintYesNo(values,message):
            while True:
                x = input(message) 
                if x in values:
                    constraintYesNo.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        constraintYesNo(["Y","N"], "\nWould you like to switch off the constraints? Y or N:\n")
        
        # If the user switches constraints off, this entire function is skipped
        if constraintYesNo.YesOrNo == "Y":
            # The user input is assigned to a global variable for later use
            ConstraintNeighborhood.ConstYesNo = "Y"
            
            # The total number of grid cells needs to be counted up, even when
            # not setting any constraints
            def totalCells(total):
                cursor = SearchCursor(preparedSurface, "Wind_Turb")
                for row in cursor:
                    total += 1
                totalCells.total = total
            
            # The function is called
            totalCells(0)

            # The total constrained grid cells are saved as a global variable
            constraints.total = totalCells.total
            
            return
        # Otherwise, the constraint setting works as before
        else:
            ConstraintNeighborhood.ConstYesNo = "N"
            # A cursor to sum these counts of prohibited grid cells is defined
            cursor = SearchCursor(preparedSurface,["Wind_Turb","Near_Air","Near_Plant","Near_Hosp", 
                                                   "Near_Sch","Near_Roads","Near_Trans","Avg_Elevat",
                                                   "Avg_Wind","Avg_Temp","Military","Nat_Parks",
                                                   "Critical","Historical","Mining","Wild_Refug",
                                                   "Trib_Land"])
            
            # If a grid cell contains a wind farm and also invalidates a constraint,
            # the count for that constraint is increased by 1
            for row in cursor:
                # Default nearest airport constraint
                if row[1] <= 2500:
                    if row[0] == "Y":
                        airportProhibited += 1
                # Default nearest power plant constraint
                if row[2] <= 10000:
                    if row[0] == "Y":
                        plantProhibited += 1
                # Default nearest hospital constraint
                if row[3] <= 2500:
                    if row[0] == "Y":
                        hospProhibited += 1
                # Default nearest school constraint
                if row[4] <= 2500:
                    if row[0] == "Y":
                        schProhibited += 1
                # Default minimum road distance constraint
                if row[5] <= 500:
                    if row[0] == "Y":
                        roadMinProhibited += 1
                # Default maximum road distance constraint
                if row[5] >= 10000:
                    if row[0] == "Y":
                        roadMaxProhibited += 1    
                # Default minimum transmission line distance constraint
                if row[6] <= 250:
                    if row[0] == "Y":
                        transMinProhibited += 1   
                # Default maximum transmission line distance constraint
                if row[6] >= 10000:
                    if row[0] == "Y":
                        transMaxProhibited += 1   
                # Default average elevation constraint
                if row[7] >= 2000:
                    if row[0] == "Y":
                        elevatProhibited += 1  
                # Default average wind speed constraint
                if row[8] <= 4:
                    if row[0] == "Y":
                        windProhibited += 1                 
                # Default average temperature constraint
                if row[9] <= 0:
                    if row[0] == "Y":
                        tempProhibited += 1    
                # Default military constraint
                if row[10] == "Y":
                    if row[0] == "Y":
                        militProhibited += 1  
                # Default national parks constraint
                if row[11] == "Y":
                    if row[0] == "Y":
                        natParkProhibited += 1  
                # Default critical habitats constraint
                if row[12] == "Y":
                    if row[0] == "Y":
                        criticalProhibited += 1  
                 # Default historical landmarks constraint 
                if row[13] == "Y":
                    if row[0] == "Y":
                        historicProhibited += 1  
                # Default mining constraint 
                if row[14] == "Y":
                    if row[0] == "Y":
                        miningProhibited += 1  
                #Default wildlife refuge constraint 
                if row[15] == "Y":
                    if row[0] == "Y" or row[0] == "Y":
                        wildProhibited += 1  
                # Default tribal land constraint        
                if row[16] == "Y":
                    if row[0] == "Y" or row[0] == "Y":
                        tribalProhibited += 1 
                # Count of the number of grid cells containing wind farms
                if row[0] == "Y":
                    windFarmCount += 1 
                    
                # Total count of the number of grid cells containing wind farms that
                # do and do not violate the default constraints
                if (row[1] <= 2500 or row[2] <= 10000 or row[3] <= 2500 or row[4] <= 2500 or row[5] <= 500 or row[5] >= 10000
                or row[6] <= 250 or row[6] >= 10000 or row[7] >= 2000 or row[8] <= 4 or row[9] <= 0 or row[10] == "Y"
                or row[11] == "Y" or row[12] == "Y" or row[13] == "Y" or row[14] == "Y" or row[15] == "Y" or row[16] == "Y"):
                    # Count of grid cells that invalidate at least one constraint
                    allCellsProhibited += 1
                    # Count of grid cells that contain a wind farm that invalidate
                    # at least one constraint
                    if row[0] == "Y":
                        turbineCellsProhibited += 1
                if (row[1] > 2500 and row[2] > 10000 and row[3] > 2500 and row[4] > 2500 and row[5] > 500 and row[5] < 10000
                and row[6] > 250 and row[6] < 10000 and row[7] < 2000 and row[8] > 4 and row[9] > 0 and row[10] == "N"
                and row[11] == "N" and row[12] == "N" and row[13] == "N" and row[14] == "N" and row[15] == "N" and row[16] == "N"):
                    # Count of grid cells that invalidate no constraints
                    allCellsAllowed += 1
                    # Count of grid cells that contain a wind farm that invalidate 
                    # no constraints
                    if row[0] == "Y":
                        turbineCellsAllowed += 1
                        
            # Counts are assigned for use in print statements below
            constraints.airport = airportProhibited
            constraints.plant = plantProhibited
            constraints.hospital = hospProhibited
            constraints.school = schProhibited
            constraints.roadMin = roadMinProhibited
            constraints.roadMax = roadMaxProhibited
            constraints.transMin = transMinProhibited
            constraints.transMax = transMaxProhibited
            constraints.elevation = elevatProhibited
            constraints.wind = windProhibited
            constraints.temp = tempProhibited
            constraints.milit = militProhibited
            constraints.natPark = natParkProhibited
            constraints.critical = criticalProhibited
            constraints.historic = historicProhibited
            constraints.mining = miningProhibited
            constraints.wild = wildProhibited
            constraints.tribal = tribalProhibited
            constraints.count = windFarmCount
            constraints.allProhibited = allCellsProhibited
            constraints.turbinesProhibited = turbineCellsProhibited
            constraints.allAllowed = allCellsAllowed
            constraints.turbinesAllowed = turbineCellsAllowed
        
            # Printouts should only be done for states that contain at least one wind farm
            if constraints.count > 0:
                # The number and percentage of existing grid cells containing wind farms that 
                # violate the default constraints are printed out
                print("\nNumber (percentage) of grid cells that contain wind farms but violate the default constraints:")
                print("".join(["- Wind farms within 2,500 meters of an airport: ", str(constraints.airport), " (", str(round(constraints.airport/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms within 10,000 meters of a power plant: ", str(constraints.plant), " (", str(round(constraints.plant/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms within 2,500 meters of a hospital: ", str(constraints.hospital), " (", str(round(constraints.hospital/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms within 2,500 meters of a school: ", str(constraints.school), " (", str(round(constraints.school/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms within 500 meters of a major road: ", str(constraints.roadMin), " (", str(round(constraints.roadMin/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms more than 10,000 meters from a major road: ", str(constraints.roadMax), " (", str(round(constraints.roadMax/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms within 250 meters of a major transmission line: ", str(constraints.transMin), " (", str(round(constraints.transMin/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms more than 10,000 meters from a major transmission line: ", str(constraints.transMax), " (", str(round(constraints.transMax/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in grid cells that average more than 2,000 meters above sea level: ", str(constraints.elevation), " (", str(round(constraints.elevation/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in grid cells with an average wind speed 80 meters above ground less than 5 meters per second: ", str(constraints.wind), " (", str(round(constraints.wind/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in grid cells with an average temperature below 0 degrees Celsius: ", str(constraints.temp), " (", str(round(constraints.temp/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in the presence of a military base or operations: ", str(constraints.milit), " (", str(round(constraints.milit/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in the presence of a National Park: ", str(constraints.natPark), " (", str(round(constraints.natPark/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in the presence of a USFWS critical species habitat: ", str(constraints.critical), " (", str(round(constraints.critical/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in the presence of a historical landmark: ", str(constraints.historic), " (", str(round(constraints.historic/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in the presence of mining operations: ", str(constraints.mining), " (", str(round(constraints.mining/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in the presence of a USFWS wildlife refuge: ", str(constraints.wild), " (", str(round(constraints.wild/constraints.count*100,2)), "%)"]))
                print("".join(["- Wind farms in the presence of tribal land: ", str(constraints.tribal), " (", str(round(constraints.tribal/constraints.count*100,2)), "%)"]))
                
                # These numbers and percentages are also represented cumulatively, since
                # grid cells containing wind farms may violate more than one constraint
                print("".join(["\nNumber of grid cells containing wind farms in ", ConstraintNeighborhood.studyArea, " that violate at least one default constraint: ", str(constraints.turbinesProhibited), " (", str(round(constraints.turbinesProhibited/constraints.count*100,2)), "%)"]))
                print("".join(["Total number of grid cells in ", ConstraintNeighborhood.studyArea, " that violate at least one default constraint: ", str(constraints.allProhibited), " (", str(round(constraints.allProhibited/(constraints.allProhibited+constraints.allAllowed)*100,2)), "%)"]))
        
                # Printout of the number of grid cells that do not violate any constraints
                print("".join(["\nNumber of grid cells containing wind farms in ", ConstraintNeighborhood.studyArea, " that do not violate any default constraints: ", str(constraints.turbinesAllowed), " (", str(round(constraints.turbinesAllowed/constraints.count*100,2)), "%)"]))        
                print("".join(["Total number of grid cells in ", ConstraintNeighborhood.studyArea, " that do not violate any default constraints: ", str(constraints.allAllowed), " (", str(round(constraints.allAllowed/(constraints.allProhibited+constraints.allAllowed)*100,2)), "%)"]))
                
                pdf.multi_cell(w=0, h=5.0, align='L', 
                              txt="\nThe number (percentage) of grid cells that contain wind farms but violate the default constraints are detailed below: "
                                  +'\n'+"- Wind farms within 2,500 meters of an airport: "+ str(constraints.airport)+ " ("+ str(round(constraints.airport/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms within 10,000 meters of a power plant: "+ str(constraints.plant)+ " ("+ str(round(constraints.plant/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms within 2,500 meters of a hospital: "+ str(constraints.hospital)+ " ("+ str(round(constraints.hospital/constraints.count*100,2))+ "%)"
                                  +'\n'+" Wind farms within 2,500 meters of a school: "+ str(constraints.school)+ " ("+ str(round(constraints.school/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms within 500 meters of a major road: "+ str(constraints.roadMin)+ " ("+ str(round(constraints.roadMin/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms more than 10,000 meters from a major road: "+ str(constraints.roadMax)+ " ("+ str(round(constraints.roadMax/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms within 250 meters of a major transmission line: "+ str(constraints.transMin)+ " ("+ str(round(constraints.transMin/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms more than 10,000 meters from a major transmission line: "+ str(constraints.transMax)+ " ("+ str(round(constraints.transMax/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in grid cells that average more than 2,000 meters above sea level: "+ str(constraints.elevation)+ " ("+ str(round(constraints.elevation/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in grid cells with an average wind speed 80 meters above ground less than 5 meters per second: "+ str(constraints.wind)+ " ("+ str(round(constraints.wind/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in grid cells with an average temperature below 0 degrees Celsius: "+ str(constraints.temp)+ " ("+ str(round(constraints.temp/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in the presence of a military base or operations: "+ str(constraints.milit)+ " ("+ str(round(constraints.milit/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in the presence of a National Park: "+ str(constraints.natPark)+ " ("+ str(round(constraints.natPark/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in the presence of a USFWS critical species habitat: "+ str(constraints.critical)+ " ("+ str(round(constraints.critical/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in the presence of a historical landmark: "+ str(constraints.historic)+ " ("+ str(round(constraints.historic/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in the presence of mining operations: "+ str(constraints.mining)+ " ("+ str(round(constraints.mining/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in the presence of a USFWS wildlife refuge: "+ str(constraints.wild)+ " ("+ str(round(constraints.wild/constraints.count*100,2))+ "%)"
                                  +'\n'+"- Wind farms in the presence of tribal land: "+ str(constraints.tribal)+ " ("+ str(round(constraints.tribal/constraints.count*100,2))+ "%)", border=0)
                
                # User is asked whether they wish to use the model's default constraints
                def defaultConstraints(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            defaultConstraints.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                defaultConstraints(["Y","N"], "\nDo you wish to use the model's default constraints? Y or N:\n")
                
            # If the state does not contain any wind farms, the following is printed out
            # instead
            if constraints.count == 0:
                # The total number of grid cells that do and do not violate the
                # default constraints
                print("".join(["\nTotal number of grid cells in ", ConstraintNeighborhood.studyArea, " that violate at least one default constraint: ", str(constraints.allProhibited), " (", str(round(constraints.allProhibited/(constraints.allProhibited+constraints.allAllowed)*100,2)), "%)"]))
                print("".join(["Total number of grid cells in ", ConstraintNeighborhood.studyArea, " that do not violate any default constraints: ", str(constraints.allAllowed), " (", str(round(constraints.allAllowed/(constraints.allProhibited+constraints.allAllowed)*100,2)), "%)"]))
        
                pdf.multi_cell(w=0, h=5.0, align='L', 
                              txt="\nSince the study area contains no commercial wind farms, there are no constraints to violate.", border=0)
                
                # User is asked whether they wish to use the model's default constraints
                def defaultConstraints(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            defaultConstraints.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                defaultConstraints(["Y","N"], "".join(["\n", ConstraintNeighborhood.studyArea, " does not contain any commercial wind farms. Do you wish to use the model's default constraints, Y or N? \n"]))
            
            # The cellular automaton's constraints are then set. The first condition
            # is for if the user wishes to use the default constraints.
            if defaultConstraints.YesOrNo == "Y":
                def airportConstraint():
                    airportConstraint.Value = 2500
                airportConstraint()
                def plantConstraint():
                    plantConstraint.Value = 10000
                plantConstraint()
                def hospConstraint():
                    hospConstraint.Value = 2500
                hospConstraint()
                def schConstraint():
                    schConstraint.Value = 2500
                schConstraint()
                def roadMinConstraint():
                    roadMinConstraint.Value = 500
                    roadMinConstraint.YesOrNo = "Y"
                roadMinConstraint()
                def roadMaxConstraint():
                    roadMaxConstraint.Value = 10000
                    roadMaxConstraint.YesOrNo = "Y"
                roadMaxConstraint()
                def transMinConstraint():
                    transMinConstraint.Value = 250
                    transMinConstraint.YesOrNo = "Y"
                transMinConstraint()
                def transMaxConstraint():
                    transMaxConstraint.Value = 10000
                    transMaxConstraint.YesOrNo = "Y"
                transMaxConstraint()
                def elevatConstraint():
                    elevatConstraint.Value = 2000
                elevatConstraint()
                def windConstraint():
                    windConstraint.Value = 4
                windConstraint()
                def tempConstraint():
                    tempConstraint.Value = 0
                tempConstraint()
                def militConstraint():
                    militConstraint.YesOrNo = "Y"
                militConstraint()
                def natParkConstraint():
                    natParkConstraint.YesOrNo = "Y"
                natParkConstraint()
                def criticalConstraint():
                    criticalConstraint.YesOrNo = "Y"
                criticalConstraint()
                def historicConstraint():
                    historicConstraint.YesOrNo = "Y"
                historicConstraint()
                def miningConstraint():
                    miningConstraint.YesOrNo = "Y"
                miningConstraint()
                def wildConstraint():
                    wildConstraint.YesOrNo = "Y"
                wildConstraint() 
                def tribalConstraint():
                    tribalConstraint.YesOrNo = "Y"
                tribalConstraint()
                
                # All constraints will be passed into the update cursor further down
                cursorList = ["Near_Air","Near_Plant","Near_Hosp","Near_Sch","Near_Roads",
                             "Near_Roads","Near_Trans","Near_Trans","Avg_Elevat","Avg_Wind",
                             "Avg_Temp","Military","Nat_Parks","Critical","Historical",
                             "Mining","Wild_Refug","Trib_Land"]
                
                # The default constraints are saved to a dataframe, to inform the user
                # of previously set constraints for this particular study area and
                # grid cell size if selected again
                dfConstraints = pd.DataFrame()
                dfConstraints["Constraints"] = ["No wind farms within 2500 meters of an airport.",
                                                "No wind farms within 10000 meters of a power plant.",
                                                "No wind farms within 2500 meters of a hospital.",
                                                "No wind farms within 2500 meters of a school.",
                                                "No wind farms within 500 meters of a major road.",
                                                "No wind farms more than 10000 meters from a major road.",
                                                "No wind farms within 250 meters of a major transmission line.",
                                                "No wind farms more than 10000 meters from a major transmission line.",
                                                "No wind farms in grid cells that average more than 2000 meters above sea level.",
                                                "No wind farms in grid cells with an average wind speed 80 meters above ground less than 5 meters per second.",
                                                "No wind farms in grid cells with an average temperature below 0 degrees Celsius.",
                                                "Wind farm development is prohibited in grid cells shared by military bases.",
                                                "Wind farm development is prohibited in grid cells shared by national parks.",
                                                "Wind farm development is prohibited in grid cells shared by USFWS critical habitats.",
                                                "Wind farm development is prohibited in grid cells shared by historical landmarks.",
                                                "Wind farm development is prohibited in grid cells shared by mining operations.",
                                                "Wind farm development is prohibited in grid cells shared by USFWS wildlife refuges.",
                                                "Wind farm development is prohibited in grid cells shared by tribal land."]
                dfConstraints.to_csv("".join([directoryPlusConstraints + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"]))        
        
            # If the user does not wish to use the default constraints, their own
            # constraints are set instead
            elif defaultConstraints.YesOrNo == "N":
                
                # Only the desired constraints will be passed into the update cursor
                cursorList = []
                cursorAppend = cursorList.append
                
                # The constraints and their respective values are saved to these lists
                constraintList = []
                constraintAppend = constraintList.append
        
                # Nearest airport constraint
                def airportConstraint(values,message):
                    while True:
                        # User specifies Y or N for whether they wish to use airports 
                        # as a constraint
                        x = input(message) 
                        if x in values:
                            airportConstraint.YesOrNo = x
                            break
                        # Invalid text entry resets the function
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            # If the user wishes to use airports, its value is set and
                            # appended to the lists above
                            setValue = input("Please set the distance to nearest airport constraint (in meters) of your choice:\n")
                            # The user input must be convertible into an integer or
                            # float
                            try:
                                int(setValue)
                                airportConstraint.Value = setValue
                                cursorAppend("Near_Air")
                                constraintAppend("".join(["No wind farms within ", airportConstraint.Value, " meters of an airport."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    airportConstraint.Value = setValue
                                    cursorAppend("Near_Air")
                                    constraintAppend("".join(["No wind farms within ", airportConstraint.Value, " meters of an airport."]))
                                    break
                                except:
                                    ValueError
                                    # If the user input cannot be converted into an
                                    # integer or float, the user is prompted to
                                    # re-enter the input
                                    print("Invalid value; please specify as an integer or float.")
                        else:
                            break            
                airportConstraint(["Y","N"], "\nWould you like to use airports as a constraint? Y or N:\n")
                
                # Nearest power plant constraint
                def plantConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            plantConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the distance to nearest power plant constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                plantConstraint.Value = setValue
                                cursorAppend("Near_Plant")
                                constraintAppend("".join(["No wind farms within ", plantConstraint.Value, " meters of a power plant."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    plantConstraint.Value = setValue
                                    cursorAppend("Near_Plant")
                                    constraintAppend("".join(["No wind farms within ", plantConstraint.Value, " meters of a power plant."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")       
                        else:
                            break            
                plantConstraint(["Y","N"], "\nWould you like to use power plants as a constraint? Y or N:\n")
                    
                # Nearest hospital constraint     
                def hospConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            hospConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the distance to nearest hospital constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                hospConstraint.Value = setValue
                                cursorAppend("Near_Hosp")
                                constraintAppend("".join(["No wind farms within ", hospConstraint.Value, " meters of a hospital."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    hospConstraint.Value = setValue
                                    cursorAppend("Near_Hosp")
                                    constraintAppend("".join(["No wind farms within ", hospConstraint.Value, " meters of a hospital."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")          
                        else:
                            break            
                hospConstraint(["Y","N"], "\nWould you like to use hospitals as a constraint? Y or N:\n")
                
                # Nearest school constraint
                def schConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            schConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the distance to nearest school constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                schConstraint.Value = setValue
                                cursorAppend("Near_Sch")
                                constraintAppend("".join(["No wind farms within ", schConstraint.Value, " meters of a school."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    schConstraint.Value = setValue
                                    cursorAppend("Near_Sch")
                                    constraintAppend("".join(["No wind farms within ", schConstraint.Value, " meters of a school."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")         
                        else:
                            break            
                schConstraint(["Y","N"], "\nWould you like to use schools as a constraint? Y or N:\n")
                
                # Minimum distance to road constraint
                def roadMinConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            roadMinConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the minimum distance to nearest road constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                roadMinConstraint.Value = setValue
                                cursorAppend("Near_Roads")
                                constraintAppend("".join(["No wind farms within ", roadMinConstraint.Value, " meters of a major road."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    roadMinConstraint.Value = setValue
                                    cursorAppend("Near_Roads")
                                    constraintAppend("".join(["No wind farms within ", roadMinConstraint.Value, " meters of a major road."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")     
                        else:
                            break            
                roadMinConstraint(["Y","N"], "\nWould you like to use minimum road distance as a constraint? Y or N:\n")
                
                # Maximum distance to road constraint
                def roadMaxConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            roadMaxConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the maximum distance to nearest road constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                roadMaxConstraint.Value = setValue
                                cursorAppend("Near_Roads")
                                constraintAppend("".join(["No wind farms more than ", roadMaxConstraint.Value, " meters from a major road."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    roadMaxConstraint.Value = setValue
                                    cursorAppend("Near_Roads")
                                    constraintAppend("".join(["No wind farms more than ", roadMaxConstraint.Value, " meters from a major road."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")          
                        else:
                            break            
                roadMaxConstraint(["Y","N"], "\nWould you like to use maximum road distance as a constraint? Y or N:\n")
                
                # Minimum distance to transmission line constraint
                def transMinConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            transMinConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the minimum distance to nearest transmission line constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                transMinConstraint.Value = setValue
                                cursorAppend("Near_Trans")
                                constraintAppend("".join(["No wind farms within ", transMinConstraint.Value, " meters of a major transmission line."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    transMinConstraint.Value = setValue
                                    cursorAppend("Near_Trans")
                                    constraintAppend("".join(["No wind farms within ", transMinConstraint.Value, " meters of a major transmission line."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")         
                        else:
                            break            
                transMinConstraint(["Y","N"], "\nWould you like to use minimum transmission line distance as a constraint? Y or N:\n")
                
                # Maximum distance to transmission line constraint
                def transMaxConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            transMaxConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the maximum distance to nearest transmission line constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                transMaxConstraint.Value = setValue
                                cursorAppend("Near_Trans")
                                constraintAppend("".join(["No wind farms more than ", transMaxConstraint.Value, " meters from a major transmission line."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    transMaxConstraint.Value = setValue
                                    cursorAppend("Near_Trans")
                                    constraintAppend("".join(["No wind farms more than ", transMaxConstraint.Value, " meters from a major transmission line."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")          
                        else:
                            break            
                transMaxConstraint(["Y","N"], "\nWould you like to use maximum transmission line distance as a constraint? Y or N:\n")
                
                # Average elevation constraint
                def elevatConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            elevatConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the maximum average elevation constraint (in meters) of your choice:\n")
                            try:
                                int(setValue)
                                elevatConstraint.Value = setValue
                                cursorAppend("Avg_Elevat")
                                constraintAppend("".join(["No wind farms in grid cells that average more than ", elevatConstraint.Value, " meters above sea level."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    elevatConstraint.Value = setValue
                                    cursorAppend("Avg_Elevat")
                                    constraintAppend("".join(["No wind farms in grid cells that average more than ", elevatConstraint.Value, " meters above sea level."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")         
                        else:
                            break            
                elevatConstraint(["Y","N"], "\nWould you like to use elevation as a constraint? Y or N:\n")
                
                # Average wind speed constraint
                def windConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            windConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))    
                    while True:
                        if x =="Y":
                            setValue = input("Please set the minimum average wind speed constraint (in meters per second) of your choice:\n")
                            try:
                                int(setValue)
                                windConstraint.Value = setValue
                                cursorAppend("Avg_Wind")
                                constraintAppend("".join(["No wind farms in grid cells with an average wind speed 80 meters above ground less than ", windConstraint.Value, " meters per second."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    windConstraint.Value = setValue
                                    cursorAppend("Avg_Wind")
                                    constraintAppend("".join(["No wind farms in grid cells with an average wind speed 80 meters above ground less than ", windConstraint.Value, " meters per second."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")              
                        else:
                            break            
                windConstraint(["Y","N"], "\nWould you like to use wind speed as a constraint? Y or N:\n")
                    
                # Average temperature constraint
                def tempConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            tempConstraint.YesOrNo = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                    while True:
                        if x =="Y":
                            setValue = input("Please set the minimum average temperature constraint (in degrees Celsius) of your choice:\n")
                            try:
                                int(setValue)
                                tempConstraint.Value = setValue
                                cursorAppend("Avg_Temp")
                                constraintAppend("".join(["No wind farms in grid cells with an average temperature below ", tempConstraint.Value, " degrees Celsius."]))
                                break
                            except:
                                ValueError
                                try:
                                    float(setValue)
                                    tempConstraint.Value = setValue
                                    cursorAppend("Avg_Temp")
                                    constraintAppend("".join(["No wind farms in grid cells with an average temperature below ", tempConstraint.Value, " degrees Celsius."]))
                                    break
                                except:
                                    ValueError
                                    print("Invalid value; please specify as an integer or float.")        
                        else:
                            break            
                tempConstraint(["Y","N"], "\nWould you like to use temperature as a constraint? Y or N:\n")
                    
                # Military base constraint
                def militConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            militConstraint.YesOrNo = x
                            if x == "Y":
                                cursorAppend("Military")
                                constraintAppend("".join(["Wind farm development is prohibited in grid cells shared by military bases."]))
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                militConstraint(["Y","N"], "\nWould you like to use military bases as a constraint, Y or N? \n")
                    
                # National park constraint
                def natParkConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            natParkConstraint.YesOrNo = x
                            if x == "Y":
                                cursorAppend("Nat_Parks")
                                constraintAppend("".join(["Wind farm development is prohibited in grid cells shared by national parks."]))
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                natParkConstraint(["Y","N"], "\nWould you like to use national parks as a constraint, Y or N? \n")
            
                # Critical habitats constraint
                def criticalConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            criticalConstraint.YesOrNo = x
                            if x == "Y":
                                cursorAppend("Critical")
                                constraintAppend("".join(["Wind farm development is prohibited in grid cells shared by USFWS critical habitats."]))
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                criticalConstraint(["Y","N"], "\nWould you like to use critical habitats as a constraint, Y or N? \n")
                
                # Historical landmarks constraint
                def historicConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            historicConstraint.YesOrNo = x
                            if x == "Y":
                                cursorAppend("Historical")
                                constraintAppend("".join(["Wind farm development is prohibited in grid cells shared by historical landmarks."]))
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                historicConstraint(["Y","N"], "\nWould you like to use historical landmarks as a constraint, Y or N? \n")
            
                # Mines and pits constraint
                def miningConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            miningConstraint.YesOrNo = x
                            if x == "Y":
                                cursorAppend("Mining")
                                constraintAppend("".join(["Wind farm development is prohibited in grid cells shared by mining operations."]))
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                miningConstraint(["Y","N"], "\nWould you like to use mines and pits as a constraint, Y or N? \n")
            
                # Wildlife refuges constraint
                def wildConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            wildConstraint.YesOrNo = x
                            if x == "Y":
                                cursorAppend("Wild_Refug")
                                constraintAppend("".join(["Wind farm development is prohibited in grid cells shared by USFWS wildlife refuges."]))
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                wildConstraint(["Y","N"], "\nWould you like to use wildlife refuges as a constraint, Y or N? \n")
                
                # Tribal land constraint
                def tribalConstraint(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            tribalConstraint.YesOrNo = x
                            if x == "Y":
                                cursorAppend("Trib_Land")
                                constraintAppend("".join(["Wind farm development is prohibited in grid cells shared by tribal land."]))
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                tribalConstraint(["Y","N"], "\nWould you like to use tribal land as a constraint, Y or N? \n")
                
                pdf.multi_cell(w=0, h=5.0, align='L', 
                              txt="\nThe following are the constraints that have been defined by the user: "
                                  +'\n'+ str(constraintList), border=0)
                
                print("\nThe following are the constraints that have been defined by the user:")    
                for item in constraintList:
                    print("- ", item)
                
                # The set constraints are saved to a dataframe, to inform the user
                # of previously set constraints for this particular study area and
                # grid cell size if selected again.
                # NOTE: Make sure that a folder named Defined_Constraints has been
                # added to the directory
                dfConstraints = pd.DataFrame()
                dfConstraints["Constraints"] = constraintList
                dfConstraints.to_csv("".join([directoryPlusConstraints + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"]))        
        
            # An empty field is added to the copy, this field will detail which grid cells
            # are prohibited from acquiring wind farms based on the model's constraints
            arcpy.AddField_management(preparedSurface, "Constraint", "DOUBLE")
            
            # The field name for the prohibited grid cells is added to the cursor list
            cursorList.append("Constraint")
            
            # This function fills rows in the empty field with 0 (yes) if at least one
            # constraint is violated or 1 (no) if none are violated
            def constrainedYesOrNo():
                # This new field is filled using the constraints defined above
                cursor = UpdateCursor(preparedSurface, cursorList)
                for row in cursor:
                    for i in range(len(cursorList)):
                        # Nearest Airport
                        if cursorList[i] == "Near_Air":
                            constraint = float(airportConstraint.Value)
                            if row[i] <= constraint:
                                row[len(cursorList)-1] = 0
                        # Nearest Power Plant
                        if cursorList[i] == "Near_Plant":
                            constraint = float(plantConstraint.Value)
                            if row[i] <= constraint:
                                row[len(cursorList)-1] = 0
                        # Nearest Hospital
                        if cursorList[i] == "Near_Hosp":
                            constraint = float(hospConstraint.Value)
                            if row[i] <= constraint:
                                row[len(cursorList)-1] = 0
                        # Nearest School
                        if cursorList[i] == "Near_Sch":
                            constraint = float(schConstraint.Value)
                            if row[i] <= constraint:
                                row[len(cursorList)-1] = 0
                        # Minimum Major Road Distance
                        if (cursorList[i] == "Near_Roads" and roadMinConstraint.YesOrNo == "Y"):
                            constraintMin = float(roadMinConstraint.Value)
                            if row[i] <= constraintMin:
                                row[len(cursorList)-1] = 0
                        # Maximum Major Road Distance
                        if (cursorList[i] == "Near_Roads" and roadMaxConstraint.YesOrNo == "Y"):
                            constraintMax = float(roadMaxConstraint.Value)
                            if row[i] >= constraintMax:
                                row[len(cursorList)-1] = 0       
                        # Minimum Major Transmission Line Distance
                        if (cursorList[i] == "Near_Trans" and transMinConstraint.YesOrNo == "Y"):
                            constraintMin = float(transMinConstraint.Value)
                            if row[i] <= constraintMin:
                                row[len(cursorList)-1] = 0
                        # Maximum Major Transmission Line Distance
                        if (cursorList[i] == "Near_Trans" and transMaxConstraint.YesOrNo == "Y"):
                            constraintMax = float(transMaxConstraint.Value)
                            if row[i] >= constraintMax:
                                row[len(cursorList)-1] = 0
                        # Average Elevation
                        if cursorList[i] == "Avg_Elevat":
                            constraint = float(elevatConstraint.Value)
                            if row[i] >= constraint:
                                row[len(cursorList)-1] = 0
                        # Average Wind Speed
                        if cursorList[i] == "Avg_Wind":
                            constraint = float(windConstraint.Value)
                            if row[i] <= constraint:
                                row[len(cursorList)-1] = 0
                        # Average Temperature
                        if cursorList[i] == "Avg_Temp":
                            constraint = float(tempConstraint.Value)
                            if row[i] <= constraint:
                                row[len(cursorList)-1] = 0            
                        # Military Base Present
                        if cursorList[i] == "Military":
                            constraint = militConstraint.YesOrNo
                            if row[i] == "Y":
                                row[len(cursorList)-1] = 0
                        # National Park Present
                        if cursorList[i] == "Nat_Parks":
                            constraint = natParkConstraint.YesOrNo
                            if row[i] == "Y":
                                row[len(cursorList)-1] = 0           
                        # USFWS Critical Habitat Present
                        if cursorList[i] == "Critical":
                            constraint = criticalConstraint.YesOrNo
                            if row[i] == "Y":
                                row[len(cursorList)-1] = 0
                        # Historical Landmark Present
                        if cursorList[i] == "Historical":
                            constraint = historicConstraint.YesOrNo
                            if row[i] == "Y":
                                row[len(cursorList)-1] = 0
                        # Mine and/or Pit Present
                        if cursorList[i] == "Mining":
                            constraint = miningConstraint.YesOrNo
                            if row[i] == "Y":
                                row[len(cursorList)-1] = 0
                        # USFWS Wildlife Refuge Present
                        if cursorList[i] == "Wild_Refug":
                            constraint = wildConstraint.YesOrNo
                            if row[i] == "Y":
                                row[len(cursorList)-1] = 0
                        # Tribal Land Present
                        if cursorList[i] == "Trib_Land":
                            constraint = tribalConstraint.YesOrNo
                            if row[i] == "Y":
                                row[len(cursorList)-1] = 0
                        # Grid cells not marked as prohibited are assigned 1
                        if cursorList[i] == "Constraint":
                            if row[i] is None:
                                row[len(cursorList)-1] = 1
                        
                        cursor.updateRow(row)
            
            # The function is called
            constrainedYesOrNo()     
            
            # The total number of grid cells that do and do not violate the set
            # constraints is counted
            def totalConstrained(constrainedCount, unconstrainedCount, total):
                cursor = SearchCursor(preparedSurface, "Constraint")
                for row in cursor:
                    if row[0] == 0:
                        constrainedCount += 1
                    else:
                        unconstrainedCount += 1
                    # Count of all grid cells
                    total += 1
                
                totalConstrained.constrained = constrainedCount
                totalConstrained.unconstrained = unconstrainedCount
                totalConstrained.total = total
            
            # The function is called
            totalConstrained(0,0,0)

            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nTotal number of grid cells in the study area that violate "
                              +'\n'+"at least one constraint: " + str(totalConstrained.constrained) + " (" + str(round(totalConstrained.constrained/totalConstrained.total*100,2)) + "%)"
                              +'\n'+"Total number of grid cells in the study area that do not violate "
                              +'\n'+"any set constraints: " + str(totalConstrained.unconstrained) + " (" + str(round(totalConstrained.unconstrained/totalConstrained.total*100,2)) + "%)", border=0)
            
            # The total constrained grid cells are saved as a global variable
            constraints.total = totalConstrained.total
            
    # The function is called
    constraints(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

    ########################### NEIGHBORHOOD EFFECTS ##############################
        
    # The neighborhood effect factor (from 0 to 1) for each cell of the gridded
    # surface is held by this list
    neighborhoodEffectList = []
    neighborhoodAppend = neighborhoodEffectList.append
    
    # A function is constructed to compute the neighborhood effects for
    # each grid cell
    def neighborhoodEffects():
        
        # The user is first asked if they wish to switch off the neighborhood
        # effects, meaning that nearby grid cells containing wind farms have
        # no effect on projected sites for new ones
        def neighborhoodYesNo(values,message):
            while True:
                x = input(message) 
                if x in values:
                    neighborhoodYesNo.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        neighborhoodYesNo(["Y","N"], "\nWould you like to switch off the neighborhood effects? Y or N:\n")
        
        # If the user switches neighborhood effects off, this entire function
        # is skipped
        if neighborhoodYesNo.YesOrNo == "Y":
            # The user input is assigned to a global variable for later use
            ConstraintNeighborhood.NeighYesNo = "Y"           
            
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\nThe user elected to switch off the neighborhood effects.", border=0)
            
        # Otherwise, the neighborhood effect setting works as before
        else:
            ConstraintNeighborhood.NeighYesNo = "N"
            # A search cursor is used to identify grid cells that neighbor each
            # cell of the gridded surface
            cursor =  SearchCursor(preparedSurface,["TARGET_FID"])
            
            # If the feature layer exists from a prior run of the script, it is deleted
            arcpy.Delete_management("present_Copy_lyr")
            
            # The gridded surface is saved as a feature layer
            arcpy.MakeFeatureLayer_management(preparedSurface, "present_Copy_lyr")
                    
            # If this filepath exists, then the hexagonal neighborhood range has
            # been set before. The user is given the option to redefine this range
            if os.path.exists("".join([directoryPlusNeighborhoods + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"])) is True:
                
                # Dataframe holding the previously defined neighborhood range
                dfNeighborhood = pd.read_csv("".join([directoryPlusNeighborhoods + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"]))
                
                # The user is asked if they wish to redefine their neighborhood
                # range of interest
                def oldRange(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            oldRange.range = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                oldRange(["Y","N"], "".join(["\nA prior model run for this study area and grid cell resolution used a hexagonal neighborhood range of ", str(dfNeighborhood["Neighborhood"][0]), ". Would you like to use this range again? Y or N:\n"]))
                
                # The former hexagonal neighborhood range is retained if the user
                # answered "Y" (yes)
                if oldRange.range == "Y":
                    ConstraintNeighborhood.neighborhoodSize = dfNeighborhood["Neighborhood"][0]
                    
                    pdf.multi_cell(w=0, h=5.0, align='L', 
                                  txt="\nThe user has retained the neighborhood size of " + str(dfNeighborhood["Neighborhood"][0])                                  
                                      +'\n'+"used in a prior model run." , border=0)
                    
                # If the user answered "N" (no), then a new range is defined
                else:
                    def newRange(values,message):
                        while True:
                            x = input(message) 
                            if x in values:
                                newRange.range = x
                                break
                            else:
                                print("Invalid value; options are " + str(values))
                    newRange(["1","2","3","4","5","6","7","8","9","10",
                               "11","12","13","14","15","16","17","18","19","20",
                               "21","22","23","24","25","26","27","28","29","30",
                               "31","32","33","34","35","36","37","38","39","40",
                               "41","42","43","44","45","46","47","48","49","50",
                               "51","52","53","54","55","56","57","58","59","60",
                               "61","62","63","64","65","66","67","68","69","70",
                               "71","72","73","74","75","76","77","78","79","80",
                               "81","82","83","84","85","86","87","88","89","90",
                               "91","92","93","94","95","96","97","98","99","100"], "\nEnter the desired range over which neighboring grid cells are defined and evaluated. Integers from 1 to 10 are recommended:\n")
                    
                    # The new hexagonal neighborhood range is assigned to a 
                    # global variable, and also saved as a dataframe for use
                    # in code runs using the same study area and grid cell size
                    ConstraintNeighborhood.neighborhoodSize = newRange.range    
                    dfNeighborhood = pd.DataFrame()
                    dfNeighborhood["Neighborhood"] = [ConstraintNeighborhood.neighborhoodSize]
                    dfNeighborhood.to_csv("".join([directoryPlusNeighborhoods + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"]))        
                                                            
                    pdf.multi_cell(w=0, h=5.0, align='L', 
                                  txt="\nBased on the desired range, grid cells that are up to " + str(ConstraintNeighborhood.neighborhoodSize) + " cells away from those that gained "
                                      +'\n'+"wind farms will have their neighborhood effect factors updated with each iteration.", border=0)
                    
            # If the filepath does not exist, then the same newRange function as 
            # above is again enlisted
            else:
                def newRange(values,message):
                    while True:
                        x = input(message) 
                        if x in values:
                            newRange.range = x
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                newRange(["1","2","3","4","5","6","7","8","9","10",
                           "11","12","13","14","15","16","17","18","19","20",
                           "21","22","23","24","25","26","27","28","29","30",
                           "31","32","33","34","35","36","37","38","39","40",
                           "41","42","43","44","45","46","47","48","49","50",
                           "51","52","53","54","55","56","57","58","59","60",
                           "61","62","63","64","65","66","67","68","69","70",
                           "71","72","73","74","75","76","77","78","79","80",
                           "81","82","83","84","85","86","87","88","89","90",
                           "91","92","93","94","95","96","97","98","99","100"], "\nEnter the desired range over which neighboring grid cells are defined and evaluated. Integers from 1 to 10 are recommended:\n")
                
                ConstraintNeighborhood.neighborhoodSize = newRange.range    
                dfNeighborhood = pd.DataFrame()
                dfNeighborhood["Neighborhood"] = [ConstraintNeighborhood.neighborhoodSize]
                dfNeighborhood.to_csv("".join([directoryPlusNeighborhoods + "/", ConstraintNeighborhood.studyArea, "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.csv"]))        
                
                pdf.multi_cell(w=0, h=5.0, align='L', 
                              txt="\nBased on the desired range, grid cells that are up to " + str(ConstraintNeighborhood.neighborhoodSize) + " cells away from those that gained "
                                  +'\n'+"wind farms will have their neighborhood effect factors updated with each iteration.", border=0)
                
            # In order to assess neighborhoods of different number of cells further away
            # from the cell of interest, search distance based on the grid cell size must
            # be specified. Grid cells are regular hexagons with resolutions based on
            # densities ranging from 25 acres/MW to 85 acres/MW and capacities ranging
            # from 30 MW (20th percentile) to 525 MW (100th percentile).
            if (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "100"): 
                # 44,625 acres, or 180,590,968 square meters
                area = 180590968
            elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "100"): 
                # 34,125 acres, or 138,098,975 square meters
                area = 138098975
            elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "100"): 
                # 23,625 acres, or 95,606,983 square meters
                area = 95606983
            elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "100"): 
                # 13,125 acres, or 53,114,991 square meters
                area = 53114991
            elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "80"): 
                # 17,127.5 acres, or 69,312,533 square meters
                area = 69312533
            elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "80"): 
                # 13,097.5 acres, or 53,003,702 square meters
                area = 53003702
            elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "80"): 
                # 9,067.5 acres, or 36,694,871 square meters
                area = 36694871
            elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "80"): 
                # 5,037.5 acres, or 20,386,039 square meters
                area = 20386039
            elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "60"): 
                # 12,750 acres, or 51,597,419 square meters
                area = 51597419
            elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "60"): 
                # 9,750 acres, or 39,456,850 square meters
                area = 39456850
            elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "60"): 
                # 6,750 acres, or 27,316,281 square meters
                area = 27316281
            elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "60"): 
                # 3,750 acres, or 15,175,712 square meters
                area = 15175712
            elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "40"): 
                # 7,650 acres, or 30,958,452 square meters
                area = 30958452
            elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "40"): 
                # 5,850 acres, or 23,674,110 square meters
                area = 23674110
            elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "40"): 
                # 4,050 acres, or 16,389,769 square meters
                area = 16389769
            elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "40"): 
                # 2,250 acres, or 9,105,427 square meters
                area = 9105427
            elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "20"): 
                # 2,550 acres, or 10,319,484 square meters
                area = 10319484
            elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "20"): 
                # 1,950 acres, or 7,891,370 square meters
                area = 7891370
            elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "20"): 
                # 1,350 acres, or 5,463,256 square meters
                area = 5463256
            elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "20"): 
                # 750 acres, or 3,035,142 square meters
                area = 3035142
                    
            # The length of one side of a hexagonal grid cell
            sideLength = sqrt(2*area/(3*sqrt(3)))
            # This length is doubled to compute the distance between two grid cell
            # centroids for grid range evaluation
            distance = sideLength*2*sin(radians(60))
            
            print("".join(["\nPlease wait while the neighboring cells around each of the ", str(constraints.total), " grid cells are identified...\n"]))
                    
            for row in tqdm(cursor):
                # The grid cell is identified
                gridCell = int(row[0])
                sql = "".join(["TARGET_FID = ", str(gridCell)])
                # Grid cells adjacent to the identified grid cell are selected. The product
                # of the input range and grid cell distance defines the area searched for
                # neighboring grid cells           
                adjacent = arcpy.SelectLayerByLocation_management("present_Copy_lyr", "HAVE_THEIR_CENTER_IN", arcpy.SelectLayerByAttribute_management("present_Copy_lyr", "New_Selection", sql), "".join([str(distance*int(ConstraintNeighborhood.neighborhoodSize)), " meters"]))
    
                # The number of neighboring grid cells is totaled by subtracting
                # the central grid cell
                totalNeighbors = int(adjacent[2]) - 1
                
                # A new cursor is used to sum the number of neighboring grid cells that
                # contain a wind farm
                windFarmCount = 0
                subCursor = SearchCursor(adjacent,["Wind_Turb","TARGET_FID"])
                for row in subCursor:
                    if row[0] == "Y": 
                        # The central grid cell should not be included in the sum
                        if row[1] != gridCell:
                            windFarmCount += 1
                
                # The neighborhood effect factor is computed and saved to its 
                # respective list.
                if totalNeighbors > 0:
                    neighborhoodEffect = windFarmCount/totalNeighbors
                # Grid cells isolated from other areas, e.g., on small islands, will have
                # zero total neighbors
                else:
                    neighborhoodEffect = 0
                neighborhoodAppend(neighborhoodEffect)
                    
            # An empty field is added to the gridded surface. This field will hold 
            # the neighborhood effect factor for each grid cell 
            arcpy.AddField_management(preparedSurface, "Neighborhood", "DOUBLE")
            
            # The cursor below will iteratively add the neighborhood effect factors to
            # the empty field
            iterator = iter(neighborhoodEffectList)
            
            # This new field is filled
            cursor = UpdateCursor(preparedSurface, "Neighborhood")
            for row in cursor:
                row[0] = next(iterator)
                cursor.updateRow(row)
        
    # The function is called
    neighborhoodEffects()
    
    print("\nComputation of each grid cell's neighborhood effect factor: Complete.\n")
    
# The function is called
ConstraintNeighborhood()   

##################### PREPARING THE CELLULAR AUTOMATON ########################

# Function call for the use of a cellular automaton to project future
# wind farm locations
def CellularAutomaton():
    
    #################### SELECTING PREDICTOR CONFIGURATIONS ###################
    
    # Since the logistic regression model could have been fitted to up to four
    # predictor configurations (Full, NoWind, WindOnly, and Reduced), the user
    # is asked for which predictor configurations they wish to run the
    # cellular automaton. The existence of an output for each configuration is
    # first checked. NOTE: Since logistic regression model runs for states
    # with less than 2 wind farms were not possible, the existence of each
    # configuration for the 14 states below is checked based on a CONUS 
    # model output.    
    if (ConstraintNeighborhood.studyArea == "Alabama" or ConstraintNeighborhood.studyArea == "Arkansas" or ConstraintNeighborhood.studyArea == "Connecticut"
    or ConstraintNeighborhood.studyArea == "Delaware" or ConstraintNeighborhood.studyArea == "Florida" or ConstraintNeighborhood.studyArea == "Georgia" 
    or ConstraintNeighborhood.studyArea == "Kentucky" or ConstraintNeighborhood.studyArea == "Louisiana" or ConstraintNeighborhood.studyArea == "Mississippi" 
    or ConstraintNeighborhood.studyArea == "New_Jersey" or ConstraintNeighborhood.studyArea == "Rhode_Island" or ConstraintNeighborhood.studyArea == "South_Carolina" 
    or ConstraintNeighborhood.studyArea == "Tennessee" or ConstraintNeighborhood.studyArea == "Virginia"):           
        studyAreaCheck = "CONUS"
    # Otherwise, the configurations used for model runs for an individual
    # state can be used
    else:
        studyAreaCheck = ConstraintNeighborhood.studyArea
    
    # This global variable is necessary when running the model for the states
    # derived from the CONUS model run, i.e. states with fewer than 2 existing
    # commercial wind farms
    CellularAutomaton.studyAreaCheck = studyAreaCheck
    
    # Empty list used to hold the names of the selected predictor
    # configurations for the console output
    configList = []
    
    # The existence of a model output that used the Full configuration
    # is first checked
    if os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, "_Full.gdb"])) is True:
        def fullConfiguration(values, message):
            while True:
                x = input(message) 
                if x in values:
                    fullConfiguration.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        fullConfiguration(["Y", "N"], "\nDo you wish to apply the 'Full' predictor configuration to the cellular automaton? Y or N:\n")
        
        # The user input result is saved for later use
        CellularAutomaton.full = fullConfiguration.YesOrNo
        
        # Selecting this configuration adds its name to the empty list above
        if fullConfiguration.YesOrNo == "Y":
            configList.append("Full")
        
    else:   
        CellularAutomaton.full = "N"
    
    # Same but for the No_Wind configuration
    if os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, "_No_Wind.gdb"])) is True:
        def noWindConfiguration(values, message):
            while True:
                x = input(message) 
                if x in values:
                    noWindConfiguration.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        noWindConfiguration(["Y", "N"], "\nDo you wish to apply the 'No_Wind' predictor configuration to the cellular automaton? Y or N:\n")
        
        # The user input result is saved for later use
        CellularAutomaton.noWind = noWindConfiguration.YesOrNo
        
        # Selecting this configuration adds its name to the empty list above
        if noWindConfiguration.YesOrNo == "Y":
            configList.append("No_Wind")
            
    else:   
        CellularAutomaton.noWind = "N"
        
    # Same but for the Wind_Only configuration
    if os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, "_Wind_Only.gdb"])) is True:
        def windOnlyConfiguration(values, message):
            while True:
                x = input(message) 
                if x in values:
                    windOnlyConfiguration.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        windOnlyConfiguration(["Y", "N"], "\nDo you wish to apply the 'Wind_Only' predictor configuration to the cellular automaton? Y or N:\n")
        
        # The user input result is saved for later use
        CellularAutomaton.windOnly = windOnlyConfiguration.YesOrNo
        
        # Selecting this configuration adds its name to the empty list above
        if windOnlyConfiguration.YesOrNo == "Y":
            configList.append("Wind_Only")
            
    else:   
        CellularAutomaton.windOnly = "N"
        
    # Same but for the Reduced configuration
    if os.path.exists("".join([directoryPlusSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, "_Reduced.gdb"])) is True:
        def reducedConfiguration(values, message):
            while True:
                x = input(message) 
                if x in values:
                    reducedConfiguration.YesOrNo = x
                    break
                else:
                    print("Invalid value; options are " + str(values))
        reducedConfiguration(["Y", "N"], "\nDo you wish to apply the 'Reduced' predictor configuration to the cellular automaton? Y or N:\n")
        
        # The user input result is saved for later use
        CellularAutomaton.reduced = reducedConfiguration.YesOrNo
        
        # Selecting this configuration adds its name to the empty list above
        if reducedConfiguration.YesOrNo == "Y":
            configList.append("Reduced")
            
    else:   
        CellularAutomaton.reduced = "N"
        
    # This list holds the results of the user inputs above. For each
    # result that equals "Y" (yes), the cellular automaton is executed
    # for the respective predictor configuration
    CellularAutomaton.configList = [CellularAutomaton.full,CellularAutomaton.noWind,
                                    CellularAutomaton.windOnly,CellularAutomaton.reduced]
        
    # If all elements in the list are "N", the user has made a mistake
    # in user inputs, and the script is aborted
    if all(element == "N" for element in CellularAutomaton.configList):
        print("\nThe user did not declare a predictor configuration for which "
              "\nthey wish to run the cellular automaton. The script is aborted.")
        sys.exit()
    
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\nThe following predictor configurations were selected by the user: "
                      +'\n'+str(configList), border=0)

    ################### DECIDING ON GAINED CAPACITY ########################
    
    # The cellular automaton needs to know how much wind power capacity
    # in a given state (or CONUS) will increase every 5 years according
    # to the Wind Vision report (between now and 2050), so it knows how 
    # many high-probability grid cells to convert to gaining a wind farm.
    # Unit of gained capacity: Megawatts (MW)
    if ConstraintNeighborhood.studyArea == "Alabama":
        gainedCapacity = 1211.6
    if ConstraintNeighborhood.studyArea == "Arizona":
        gainedCapacity = 604
    if ConstraintNeighborhood.studyArea == "Arkansas":
        gainedCapacity = 1211.6
    if ConstraintNeighborhood.studyArea == "California":
        gainedCapacity = 2036.6
    if ConstraintNeighborhood.studyArea == "Colorado":
        gainedCapacity = 435.5
    if ConstraintNeighborhood.studyArea == "Connecticut":
        gainedCapacity = 331.4
    if ConstraintNeighborhood.studyArea == "Delaware":
        gainedCapacity = 332.8
    if ConstraintNeighborhood.studyArea == "Florida":
        gainedCapacity = 333.2
    if ConstraintNeighborhood.studyArea == "Georgia":
        gainedCapacity = 414.3
    if ConstraintNeighborhood.studyArea == "Idaho":
        gainedCapacity = 544.8
    if ConstraintNeighborhood.studyArea == "Illinois":
        gainedCapacity = 10584.3
    if ConstraintNeighborhood.studyArea == "Indiana":
        gainedCapacity = 4872.2
    if ConstraintNeighborhood.studyArea == "Iowa":
        gainedCapacity = 8070.8
    if ConstraintNeighborhood.studyArea == "Kansas":
        gainedCapacity = 1189.7
    if ConstraintNeighborhood.studyArea == "Kentucky":
        gainedCapacity = 706.9
    if ConstraintNeighborhood.studyArea == "Louisiana":
        gainedCapacity = 1513.2
    if ConstraintNeighborhood.studyArea == "Maine":
        gainedCapacity = 654.8
    if ConstraintNeighborhood.studyArea == "Maryland":
        gainedCapacity = 1481.3
    if ConstraintNeighborhood.studyArea == "Massachusetts":
        gainedCapacity = 1191.3
    if ConstraintNeighborhood.studyArea == "Michigan":
        gainedCapacity = 62.4
    if ConstraintNeighborhood.studyArea == "Minnesota":
        gainedCapacity = 1854.6
    if ConstraintNeighborhood.studyArea == "Mississippi":
        gainedCapacity = 333.2
    if ConstraintNeighborhood.studyArea == "Missouri":
        gainedCapacity = 1867.1
    if ConstraintNeighborhood.studyArea == "Montana":
        gainedCapacity = 5214.6
    if ConstraintNeighborhood.studyArea == "Nebraska":
        gainedCapacity = 1070.5
    if ConstraintNeighborhood.studyArea == "Nevada":
        gainedCapacity = 307.9
    if ConstraintNeighborhood.studyArea == "New_Hampshire":
        gainedCapacity = 297.5
    if ConstraintNeighborhood.studyArea == "New_Jersey":
        gainedCapacity = 2612.9
    if ConstraintNeighborhood.studyArea == "New_Mexico":
        gainedCapacity = 1704.3
    if ConstraintNeighborhood.studyArea == "New_York":
        gainedCapacity = 2944.9
    if ConstraintNeighborhood.studyArea == "North_Carolina":
        gainedCapacity = 2375.7
    if ConstraintNeighborhood.studyArea == "North_Dakota":
        gainedCapacity = 949.1
    if ConstraintNeighborhood.studyArea == "Ohio":
        gainedCapacity = 3572.5
    if ConstraintNeighborhood.studyArea == "Oklahoma":
        gainedCapacity = 378.4
    if ConstraintNeighborhood.studyArea == "Oregon":
        gainedCapacity = 3927
    if ConstraintNeighborhood.studyArea == "Pennsylvania":
        gainedCapacity = 1433
    if ConstraintNeighborhood.studyArea == "Rhode_Island":
        gainedCapacity = 588.7
    if ConstraintNeighborhood.studyArea == "South_Carolina":
        gainedCapacity = 2410.3
    if ConstraintNeighborhood.studyArea == "South_Dakota":
        gainedCapacity = 592.4
    if ConstraintNeighborhood.studyArea == "Tennessee":
        gainedCapacity = 409.5
    if ConstraintNeighborhood.studyArea == "Texas":
        gainedCapacity = 10623.6
    if ConstraintNeighborhood.studyArea == "Utah":
        gainedCapacity = 349.2
    if ConstraintNeighborhood.studyArea == "Vermont":
        gainedCapacity = 308.3
    if ConstraintNeighborhood.studyArea == "Virginia":
        gainedCapacity = 704.9
    if ConstraintNeighborhood.studyArea == "Washington":
        gainedCapacity = 954
    if ConstraintNeighborhood.studyArea == "West_Virginia":
        gainedCapacity = 583.2
    if ConstraintNeighborhood.studyArea == "Wisconsin":
        gainedCapacity = 478.6
    if ConstraintNeighborhood.studyArea == "Wyoming":
        gainedCapacity = 2525.5
    if ConstraintNeighborhood.studyArea == "CONUS":
        gainedCapacity = 89190.1       
    
    # The number of grid cells to gain a wind farm every 5 years is 
    # calculated based on the gained capacity and the grid cell resolution.
    # Capacities are derived from commercial wind farm sizes in the
    # United States Wind Turbine Database, in Megawatts
    if ConstraintNeighborhood.capacity == "100":
        oneFarmCapacity = 525
    if ConstraintNeighborhood.capacity == "80":
        oneFarmCapacity = 202
    if ConstraintNeighborhood.capacity == "60":
        oneFarmCapacity = 150
    if ConstraintNeighborhood.capacity == "40":
        oneFarmCapacity = 90
    if ConstraintNeighborhood.capacity == "20":
        oneFarmCapacity = 30
        
    # The user may wish to customize a gained capacity, first with
    # a yes or no question
    def customCapacity(values, message):
        while True:
            x = input(message) 
            if x in values:
                customCapacity.YesOrNo = x
                break
            else:
                print("Invalid value; options are " + str(values))
        while True:
            if x == "Y":
                # The user sets their own gained capacity value
                setValue = input("\nPlease specify your desired gained capacity (in Megawatts):\n")
                # The custom value must be convertible into a float or
                # or integer to be valid
                try:
                    int(setValue)
                    customCapacity.capacity = int(setValue)
                    # The number of gained wind farms is rounded up to
                    # the next integer
                    customCapacity.gainedWindFarms = ceil(customCapacity.capacity/oneFarmCapacity)
                    # Console output
                    pdf.multi_cell(w=0, h=5.0, align='L', 
                               txt="\nThe user specified a custom gained wind farm capacity of " + str(customCapacity.capacity) + " MW every 5 years, "
                                   +'\n'+"which based on model resolution (" + str(ConstraintNeighborhood.capacity) + "th percentile, " + str(oneFarmCapacity) + "MW) translates to " + str(customCapacity.gainedWindFarms) + " new wind farms per model iteration.", border=0)
                    break
                except:
                    ValueError
                    try:
                        float(setValue)
                        customCapacity.capacity = float(setValue)
                        customCapacity.gainedWindFarms = ceil(customCapacity.capacity/oneFarmCapacity)
                        pdf.multi_cell(w=0, h=5.0, align='L', 
                                   txt="\nThe user specified a custom gained wind farm capacity of " + str(customCapacity.capacity) + " MW every 5 years, "
                                       +'\n'+"which based on model resolution (" + str(ConstraintNeighborhood.capacity) + "th percentile, " + str(oneFarmCapacity) + "MW) translates to " + str(customCapacity.gainedWindFarms) + " new wind farms per model iteration.", border=0)
                        break
                    except:
                        ValueError
                        print("Invalid value; please specify as an integer or float.")
                 
            # If the user does not wish to use a custom gained capacity
            # the default is used instead
            elif x == "N":
                customCapacity.gainedWindFarms = ceil(gainedCapacity/oneFarmCapacity)               
                # Console output
                pdf.multi_cell(w=0, h=5.0, align='L', 
                           txt="\nThe user specified a custom gained wind farm capacity of " + str(gainedCapacity) + " MW every 5 years, "
                               +'\n'+"which based on model resolution (" + str(ConstraintNeighborhood.capacity) + "th percentile, " + str(oneFarmCapacity) + "MW) translates to " + str(customCapacity.gainedWindFarms) + " new wind farms per model iteration.", border=0)
                break
    customCapacity(["Y", "N"], "".join(["\nThe default gained wind farm capacity for ", ConstraintNeighborhood.studyArea, " is ", str(gainedCapacity), " MW every 5 years. Would you like to set your own gain of capacity? Y or N:\n"]))
    
    # The number of gained wind farms represents the number of grid cells
    # whose state will change in each time step
    gridCellsChanged = customCapacity.gainedWindFarms

    ######################## SCENARIO CONSTRUCTION ############################
    pdf.multi_cell(w=0, h=5.0, align='L', 
                  txt="\n------------------ SCENARIO CONSTRUCTION ------------------", border=0)
                  
    # The scenario of interest is specified. The choice is
    # given to modify the coefficients of the model, with each
    # modification ascribing different influences on the growth
    # on the wind energy sector:
    # 1.  DEFAULT: The coefficients of predictors remain constant and changes are
    #              driven solely by neighborhood effects.
    # 2.  CLIMATE_CHANGE: Temperature and wind speed increase, and bird and bat 
    #              habitats are increasingly threatened.
    # 3.  DEMOGRAPHIC_CHANGES: Demographics that are statistically more supportive 
    #              of wind energy projects comprise a greater amount of local populations.
    # 4.  SOCIOPOLITICAL_LANDSCAPE: Support for wind energy development among 
    #              politicians and the electorate increases.
    # 5.  CHANGING_ENERGY_ECONOMIES: Older forms of energy generation age out, and a 
    #              demand grows for green energy and green jobs.
    # 6.  NEW_INFRASTRUCTURE: Roads and transmission lines are built to support
    #              development of new commercial wind farms.
    # 7.  NATURAL_AND_CULTURAL_PROTECTION: Protection of land that is historically, 
    #              culturally, or environmentally significant is prioritized as 
    #              commercial wind energy development continues.
    # 8.  URBAN_PROTECTION: Wind energy development continues at a distance set far 
    #              enough away from industrial and domestic activities.
    # 9.  CUSTOM: A unique set of changes to coefficients can be made by the user.
    # 10. NATIONWIDE: In model runs performed for the CONUS, predictors with effects 
    #              at a nationwide level, such as legislation in effect, lobbyism, 
    #              and land value can be implemented as an extra scenario.    
    
    scenarioList = []
    
    # Coefficients and intercepts derived from fitting the logistic regression
    # model using all four predictor configurations are opened
    if CellularAutomaton.full == "Y":
        dfCoefficientsFull = pd.read_csv("".join([directoryPlusCoefficients + "/", studyAreaCheck, "/Coeffs_Full_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
        dfInterceptFull = pd.read_csv("".join([directoryPlusIntercepts + "/", studyAreaCheck, "/Intercept_Full_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
    # If an output for the configuraton doesn't exist, then the dataframes 
    # don't exist either
    else:
        dfCoefficientsFull = None
        dfInterceptFull = None
    if CellularAutomaton.noWind == "Y":
        dfCoefficientsNoWind = pd.read_csv("".join([directoryPlusCoefficients + "/", studyAreaCheck, "/Coeffs_No_Wind_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
        dfInterceptNoWind = pd.read_csv("".join([directoryPlusIntercepts + "/", studyAreaCheck, "/Intercept_No_Wind_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
    else:
        dfCoefficientsNoWind = None
        dfInterceptNoWind = None
    if CellularAutomaton.windOnly == "Y":
        dfCoefficientsWindOnly = pd.read_csv("".join([directoryPlusCoefficients + "/", studyAreaCheck, "/Coeffs_Wind_Only_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
        dfInterceptWindOnly = pd.read_csv("".join([directoryPlusIntercepts + "/", studyAreaCheck, "/Intercept_Wind_Only_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
    else:
        dfCoefficientsWindOnly = None
        dfInterceptWindOnly = None 
    if CellularAutomaton.reduced == "Y":
        dfCoefficientsReduced = pd.read_csv("".join([directoryPlusCoefficients + "/", studyAreaCheck, "/Coeffs_Reduced_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
        dfInterceptReduced = pd.read_csv("".join([directoryPlusIntercepts + "/", studyAreaCheck, "/Intercept_Reduced_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]), index_col = 0)
    else:
        dfCoefficientsReduced = None
        dfInterceptReduced = None
        
    # If the default scenario is desired, no others can be chosen
    def defaultScenario(values, message):
        while True:
            x = input(message) 
            if x in values:
                defaultScenario.YesOrNo = x
                break
            else:
                print("Invalid value; options are " + str(values))
    defaultScenario(["Y", "N"], "".join(["\nThe DEFAULT scenario would keep the coefficients of all selected predictor configurations constant, meaning changes in grid cell states are driven only by neighborhood effects. \n"
                                         "Do you wish to apply the DEFAULT scenario? Y or N:\n"]))
    
    # The user input result is saved to a global variable
    CellularAutomaton.default = defaultScenario.YesOrNo
    
    # If the user says yes to the DEFAULT scenario,
    # it is saved to the scenario list
    if CellularAutomaton.default == "Y":
        scenarioList.append("DEFAULT")

        # Since the DEFAULT scenario was selected, the percent change
        # in all coefficients is zero percent, meaning the fitted 
        # coefficients will be used as unchanged in each iteration of the 
        # cellular automaton        
        if dfCoefficientsFull is not None:
            coefficientChangesFull = [0] * len(dfCoefficientsFull)
            predictorsFull = dfCoefficientsFull["Predictor_Codes"].tolist()
        if dfCoefficientsNoWind is not None:
            coefficientChangesNoWind = [0] * len(dfCoefficientsNoWind)
            predictorsNoWind = dfCoefficientsNoWind["Predictor_Codes"].tolist()
        if dfCoefficientsWindOnly is not None:
            coefficientChangesWindOnly = [0] * len(dfCoefficientsWindOnly)
            predictorsWindOnly = dfCoefficientsWindOnly["Predictor_Codes"].tolist()
        if dfCoefficientsReduced is not None:
            coefficientChangesReduced = [0] * len(dfCoefficientsReduced)
            predictorsReduced = dfCoefficientsReduced["Predictor_Codes"].tolist()
        
        pdf.multi_cell(w=0, h=5.0, align='L', 
                   txt="\nThe user selected the DEFAULT scenario, meaning no predictor coefficients change across the model's iterations.", border=0)

    # If the DEFAULT scenario is not selected, the user is asked
    # about using the CUSTOM scenario next
    if CellularAutomaton.default == "N":
        
        # The coefficient change lists are emptied for this new scenario
        if dfCoefficientsFull is not None:
            coefficientChangesFull = []
            predictorsFull = dfCoefficientsFull["Predictor_Codes"].tolist()
        if dfCoefficientsNoWind is not None:
            coefficientChangesNoWind = []
            predictorsNoWind = dfCoefficientsNoWind["Predictor_Codes"].tolist()
        if dfCoefficientsWindOnly is not None:
            coefficientChangesWindOnly = []
            predictorsWindOnly = dfCoefficientsWindOnly["Predictor_Codes"].tolist()
        if dfCoefficientsReduced is not None:
            coefficientChangesReduced = []
            predictorsReduced = dfCoefficientsReduced["Predictor_Codes"].tolist()
        
        # The CUSTOM scenario loops through all predictors and asks the user
        # for percent coefficient changes for each of them. Predictors that
        # feature in each of the four configurations will have the user-defined
        # percent changes added to the empty lists above
        if ConstraintNeighborhood.studyArea == "CONUS":
            predictorList = ["Average Elevation","Average Temperature","Average Wind Speed",
                             "Bat Species Count","Bird Species Count","Critical Habitats",
                             "Electricity Cost","Employment Type","Farmland Value",
                             "Financial Incentives","Fossil Fuel Lobbies","Green Lobbies",
                             "Gubernatorial Elections","Historical Landmarks","Interconnection Policy",
                             "ISOs","Investment Tax Credits","Military Installations",
                             "Mining Operations","National Parks","Nearest Airport",
                             "Nearest Power Plant","Nearest Road","Nearest School",
                             "Nearest Transmission Line","Nearest Hospital","Net Metering Policy",
                             "Percent Female","Percent Hispanic","Percent Under 25",
                             "Percent White","Political Legislations","Population Density",
                             "Power Plant Age","Presidential Elections","Property Tax Exemptions",
                             "Property Value","RPS Policy","RPS Support","RPS Target",
                             "Rugged Land","Sales Tax Abatements","Tribal Lands",
                             "Undevelopable Land","Unemployment Rate","Wildlife Refuges",
                             "Wind Farm Age"]
            predictorCodeList = ["Avg_Elevat","Avg_Temp","Avg_Wind","Bat_Count",
                                 "Bird_Count","Critical","Cost_15_19","Type_15_19",
                                 "Farm_15_19","Numb_Incen","Foss_Lobbs","Gree_Lobbs",
                                 "Rep_Wins","Historical","Interconn","ISO_YN",
                                 "In_Tax_Cre","Military","Mining","Nat_Parks",
                                 "Near_Air","Near_Plant","Near_Roads","Near_Sch",
                                 "Near_Trans","Near_Hosp","Net_Meter","Fem_15_19",
                                 "Hisp_15_19","Avg_25","Whit_15_19","Numb_Pols",
                                 "Dens_15_19","Plant_Year","Dem_Wins","Tax_Prop",
                                 "Prop_15_19","Renew_Port","supp_2018","Renew_Targ",
                                 "Prop_Rugg","Tax_Sale","Trib_Land","Undev_Land",
                                 "Unem_15_19","Wild_Refug","Farm_Year"]
        
        else:
            predictorList = ["Average Elevation","Average Temperature","Average Wind Speed",
                             "Bat Species Count","Bird Species Count","Critical Habitats",
                             "Employment Type","Historical Landmarks","ISOs","Military Installations",
                             "Mining Operations","National Parks","Nearest Airport",
                             "Nearest Power Plant","Nearest Road","Nearest School",
                             "Nearest Transmission Line","Nearest Hospital",
                             "Percent Female","Percent Hispanic","Percent Under 25",
                             "Percent White","Population Density","Power Plant Age",
                             "Presidential Elections","RPS Support","Rugged Land",
                             "Tribal Lands","Undevelopable Land","Unemployment Rate",
                             "Wildlife Refuges","Wind Farm Age"]
            
            predictorCodeList = ["Avg_Elevat","Avg_Temp","Avg_Wind","Bat_Count",
                                 "Bird_Count","Critical","Type_15_19","Historical",
                                 "ISO_YN","Military","Mining","Nat_Parks","Near_Air",
                                 "Near_Plant","Near_Roads","Near_Sch","Near_Trans",
                                 "Near_Hosp","Fem_15_19","Hisp_15_19","Avg_25","Whit_15_19",
                                 "Dens_15_19","Plant_Year","Dem_Wins","supp_2018",
                                 "Prop_Rugg","Trib_Land","Undev_Land",
                                 "Unem_15_19","Wild_Refug","Farm_Year"]
        
        # First asked whether the CUSTOM scenario is desired
        def customScenario(values, message):
            while True:
                x = input(message) 
                if x in values:
                    customScenario.YesOrNo = x
                    # If a CUSTOM scenario is requested, then the percent
                    # changes in each predictor are set
                    if x == "Y":   
                        
                        pdf.multi_cell(w=0, h=5.0, align='L', 
                                   txt="\nThe user selected the CUSTOM scenario, comprised of the following predictor coefficient changes:", border=0)
                        
                        # The coefficients whose values will be modified in the 
                        # CUSTOM scenario are asked of the user.
                        for i in range(len(predictorList)):
                            def customPredictors(values, message):
                                while True:
                                    x = input(message) 
                                    if x in values:
                                        customPredictors.YesOrNo = x
                                        break
                                    else:
                                        print("Invalid value; options are " + str(values))
                                while True:
                                    if x == "Y":
                                        # The user sets their own percent change in
                                        # coefficient value
                                        setValue = input("".join(["Please specify the change of the ", predictorList[i], " coefficient to happen every 5 years (in percent):\n"]))
                                        # The custom value must be convertible into a float or
                                        # or integer to be valid
                                        try:
                                            int(setValue)
                                            customPredictors.percent = int(setValue)
                                            print("".join(["The ", predictorList[i], " coefficient will change by ", str(customPredictors.percent), "% every 5 years."]))
                                            # The input percent change is saved to the coefficient
                                            # change lists of the selected predictor configurations
                                            if dfCoefficientsFull is not None:
                                                if predictorList[i] in dfCoefficientsFull["Predictors"].tolist():                                                    
                                                    coefficientChangesFull.append(customPredictors.percent)
                                            if dfCoefficientsNoWind is not None:
                                                if predictorList[i] in dfCoefficientsNoWind["Predictors"].tolist():
                                                    coefficientChangesNoWind.append(customPredictors.percent)
                                            if dfCoefficientsWindOnly is not None:
                                                if predictorList[i] in dfCoefficientsWindOnly["Predictors"].tolist():
                                                    coefficientChangesWindOnly.append(customPredictors.percent)
                                            if dfCoefficientsReduced is not None:
                                                if predictorList[i] in dfCoefficientsReduced["Predictors"].tolist():
                                                    coefficientChangesReduced.append(customPredictors.percent)                                            
                                            # Console output
                                            pdf.multi_cell(w=0, h=5.0, align='L', 
                                                       txt="The coefficient of " + str(predictorList[i]) + " changes by " + str(customPredictors.percent) + "% per model iteration.", border=0)
                                            break
                                        except:
                                            ValueError
                                            try:
                                                float(setValue)
                                                customPredictors.percent = float(setValue)
                                                print("".join(["The ", predictorList[i], " coefficient will change by ", str(customPredictors.percent), "% every 5 years."]))
                                                # The input percent change is saved to the coefficient
                                                # change lists of the selected predictor configurations
                                                if dfCoefficientsFull is not None:
                                                    if predictorList[i] in dfCoefficientsFull["Predictors"].tolist():                                                    
                                                        coefficientChangesFull.append(customPredictors.percent)
                                                if dfCoefficientsNoWind is not None:
                                                    if predictorList[i] in dfCoefficientsNoWind["Predictors"].tolist():
                                                        coefficientChangesNoWind.append(customPredictors.percent)
                                                if dfCoefficientsWindOnly is not None:
                                                    if predictorList[i] in dfCoefficientsWindOnly["Predictors"].tolist():
                                                        coefficientChangesWindOnly.append(customPredictors.percent)
                                                if dfCoefficientsReduced is not None:
                                                    if predictorList[i] in dfCoefficientsReduced["Predictors"].tolist():
                                                        coefficientChangesReduced.append(customPredictors.percent)                                                    
                                                pdf.multi_cell(w=0, h=5.0, align='L', 
                                                           txt="The coefficient of " + str(predictorList[i]) + " changes by " + str(customPredictors.percent) + "% per model iteration.", border=0)
                                                break
                                            except:
                                                ValueError
                                                print("Invalid value; please specify as an integer or float.")
                                    # The user may not wish to use a certain coefficient
                                    elif x == "N":
                                        print("".join(["The ", predictorList[i], " coefficient will not change every 5 years."]))
                                        # The input percent change is saved to the coefficient
                                        # change lists of the selected predictor configurations
                                        if dfCoefficientsFull is not None:
                                            if predictorList[i] in dfCoefficientsFull["Predictors"].tolist():                                                    
                                                coefficientChangesFull.append(0)
                                        if dfCoefficientsNoWind is not None:
                                            if predictorList[i] in dfCoefficientsNoWind["Predictors"].tolist():
                                                coefficientChangesNoWind.append(0)
                                        if dfCoefficientsWindOnly is not None:
                                            if predictorList[i] in dfCoefficientsWindOnly["Predictors"].tolist():
                                                coefficientChangesWindOnly.append(0)
                                        if dfCoefficientsReduced is not None:
                                            if predictorList[i] in dfCoefficientsReduced["Predictors"].tolist():
                                                coefficientChangesReduced.append(0)    
                                        break
                            customPredictors(["Y", "N"], "".join(["\nDo you wish to modify the ", predictorList[i], " (", predictorCodeList[i], ") coefficient? Y or N:\n"]))
                    break
                else:
                    print("Invalid value; options are " + str(values))
            
        customScenario(["Y", "N"], "".join(["\nThe CUSTOM scenario allows unique percentage changes to be applied to all predictors. "
                                            "Do you wish build a CUSTOM scenario for all selected predictor configurations? Y or N:\n"]))
        
        # The user input result is saved to a global variable
        CellularAutomaton.custom = customScenario.YesOrNo
        
        # If the user says said to the CUSTOM scenario,
        # "CUSTOM" is saved to the scenario list
        if CellularAutomaton.custom == "Y":
            scenarioList.append("CUSTOM")
                    
        # Desires for the seven remaining scenarios are 
        # requested if the user wishes not to use the DEFAULT
        # nor the CUSTOM scenario
        if CellularAutomaton.custom == "N":   
            
            # The coefficient change lists are replaced with zeros,
            # so that coefficients not selected by the user (as part of 
            # the scenarios) are assigned a value of zero rather than being
            # blank. Lists are also created of the predictor codes in the
            # four configurations
            if dfCoefficientsFull is not None:
                coefficientChangesFull = [0] * len(dfCoefficientsFull)
                predictorsFull = dfCoefficientsFull["Predictor_Codes"].tolist()
            if dfCoefficientsNoWind is not None:
                coefficientChangesNoWind = [0] * len(dfCoefficientsNoWind)
                predictorsNoWind = dfCoefficientsNoWind["Predictor_Codes"].tolist()
            if dfCoefficientsWindOnly is not None:
                coefficientChangesWindOnly = [0] * len(dfCoefficientsWindOnly)
                predictorsWindOnly = dfCoefficientsWindOnly["Predictor_Codes"].tolist()
            if dfCoefficientsReduced is not None:
                coefficientChangesReduced = [0] * len(dfCoefficientsReduced)
                predictorsReduced = dfCoefficientsReduced["Predictor_Codes"].tolist()

            # Starting with the CLIMATE_CHANGE scenario
            def climateScenario(values, message):
                while True:
                    x = input(message) 
                    if x in values:
                        climateScenario.YesOrNo = x
                        # Coefficient changes are filled into the list
                        if x == "Y":
                            # Wind Speed
                            if dfCoefficientsFull is not None:
                                if "Avg_Wind" in predictorsFull:
                                    index = predictorsFull.index("Avg_Wind")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Avg_Wind" in predictorsNoWind:
                                    index = predictorsNoWind.index("Avg_Wind")
                                    coefficientChangesNoWind[index] = 10
                            if dfCoefficientsWindOnly is not None:
                                if "Avg_Wind" in predictorsWindOnly:
                                    index = predictorsWindOnly.index("Avg_Wind")
                                    coefficientChangesWindOnly[index] = 10
                            if dfCoefficientsReduced is not None:
                                if "Avg_Wind" in predictorsReduced:
                                    index = predictorsReduced.index("Avg_Wind")
                                    coefficientChangesReduced[index] = 10                                
                            # Temperature
                            if dfCoefficientsFull is not None:
                                if "Avg_Temp" in predictorsFull:
                                    index = predictorsFull.index("Avg_Temp")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Avg_Temp" in predictorsNoWind:
                                    index = predictorsNoWind.index("Avg_Temp")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Avg_Temp" in predictorsReduced:
                                    index = predictorsReduced.index("Avg_Temp")
                                    coefficientChangesReduced[index] = 10                            
                            # Bat Species
                            if dfCoefficientsFull is not None:
                                if "Bat_Count" in predictorsFull:
                                    index = predictorsFull.index("Bat_Count")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Bat_Count" in predictorsNoWind:
                                    index = predictorsNoWind.index("Bat_Count")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Bat_Count" in predictorsReduced:
                                    index = predictorsReduced.index("Bat_Count")
                                    coefficientChangesReduced[index] = -10
                            # Bird Species
                            if dfCoefficientsFull is not None:
                                if "Bird_Count" in predictorsFull:
                                    index = predictorsFull.index("Bird_Count")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Bird_Count" in predictorsNoWind:
                                    index = predictorsNoWind.index("Bird_Count")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Bird_Count" in predictorsReduced:
                                    index = predictorsReduced.index("Bird_Count")
                                    coefficientChangesReduced[index] = -10                                 
                        break
                    else:
                        print("Invalid value; options are " + str(values))
            climateScenario(["Y", "N"], "".join(["\nThe CLIMATE_CHANGE scenario will increase the Temperature and Wind speed coefficients, "
                                                  "\nand decrease the Bat Species and Bird Species coefficients, by 10% in each 5-year timestep. "
                                                  "\nDo you wish to apply the model's CLIMATE_CHANGE scenario to the selected predictor configurations? Y or N:\n"]))
                                
            # The user input result is saved to a global variable
            CellularAutomaton.climate = climateScenario.YesOrNo
            
            # If the user says yes to the CLIMATE_CHANGE 
            # scenario, it is saved to the scenario list
            if CellularAutomaton.climate == "Y":
                scenarioList.append("CLIMATE_CHANGE")                    
    
            # Next the DEMOGRAPHIC_CHANGES scenario
            def demographicScenario(values, message):
                while True:
                    x = input(message) 
                    if x in values:
                        demographicScenario.YesOrNo = x
                        # Coefficient changes are filled into the list
                        if x == "Y":
                            # Age
                            if dfCoefficientsFull is not None:
                                if "Avg_25" in predictorsFull:
                                    index = predictorsFull.index("Avg_25")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Avg_25" in predictorsNoWind:
                                    index = predictorsNoWind.index("Avg_25")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Avg_25" in predictorsReduced:
                                    index = predictorsReduced.index("Avg_25")
                                    coefficientChangesReduced[index] = 10 
                            # Ethnicity
                            if dfCoefficientsFull is not None:
                                if "Hisp_15_19" in predictorsFull:
                                    index = predictorsFull.index("Hisp_15_19")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Hisp_15_19" in predictorsNoWind:
                                    index = predictorsNoWind.index("Hisp_15_19")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Hisp_15_19" in predictorsReduced:
                                    index = predictorsReduced.index("Hisp_15_19")
                                    coefficientChangesReduced[index] = 10 
                            # Gender
                            if dfCoefficientsFull is not None:
                                if "Fem_15_19" in predictorsFull:
                                    index = predictorsFull.index("Fem_15_19")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Fem_15_19" in predictorsNoWind:
                                    index = predictorsNoWind.index("Fem_15_19")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Fem_15_19" in predictorsReduced:
                                    index = predictorsReduced.index("Fem_15_19")
                                    coefficientChangesReduced[index] = 10                             
                            # Race
                            if dfCoefficientsFull is not None:
                                if "Whit_15_19" in predictorsFull:
                                    index = predictorsFull.index("Whit_15_19")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Whit_15_19" in predictorsNoWind:
                                    index = predictorsNoWind.index("Whit_15_19")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Whit_15_19" in predictorsReduced:
                                    index = predictorsReduced.index("Whit_15_19")
                                    coefficientChangesReduced[index] = -10                                 
                        break
                    else:
                        print("Invalid value; options are " + str(values))
            demographicScenario(["Y", "N"], "".join(["\nThe DEMOGRAPHIC_CHANGES scenario will increase the Age, Ethnicity, and Gender "
                                                      "\ncoefficients, and decrease the Race coefficient, by 10% in each 5-year timestep. "
                                                      "\nDo you wish to apply the model's DEMOGRAPHIC_CHANGES scenario to the selected predictor configurations? Y or N:\n"]))
            
            CellularAutomaton.demographics = demographicScenario.YesOrNo
            
            if CellularAutomaton.demographics == "Y":
                scenarioList.append("DEMOGRAPHIC_CHANGES")
            
            # Next the SOCIOPOLITICAL_LANDSCAPE scenario
            def politicsScenario(values, message):
                while True:
                    x = input(message) 
                    if x in values:
                        politicsScenario.YesOrNo = x
                        # Coefficient changes are filled into the list
                        if x == "Y":
                            # Presidential Elections
                            if dfCoefficientsFull is not None:
                                if "Dem_Wins" in predictorsFull:
                                    index = predictorsFull.index("Dem_Wins")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Dem_Wins" in predictorsNoWind:
                                    index = predictorsNoWind.index("Dem_Wins")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Dem_Wins" in predictorsReduced:
                                    index = predictorsReduced.index("Dem_Wins")
                                    coefficientChangesReduced[index] = 10 
                            # Public Opinion
                            if dfCoefficientsFull is not None:
                                if "supp_2018" in predictorsFull:
                                    index = predictorsFull.index("supp_2018")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "supp_2018" in predictorsNoWind:
                                    index = predictorsNoWind.index("supp_2018")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "supp_2018" in predictorsReduced:
                                    index = predictorsReduced.index("supp_2018")
                                    coefficientChangesReduced[index] = 10                             
                        break
                    else:
                        print("Invalid value; options are " + str(values))
            politicsScenario(["Y", "N"], "".join(["\nThe SOCIOPOLITICAL_LANDSCAPE scenario will increase the Presidential Elections "
                                                  "\nand Public Opinion coefficients by 10% in each 5-year timestep. "
                                                  "\nDo you wish to apply the model's SOCIOPOLITICAL_LANDSCAPE scenario to the selected predictor configurations? Y or N:\n"]))
            
            CellularAutomaton.politics = politicsScenario.YesOrNo
            
            if CellularAutomaton.politics == "Y":
                scenarioList.append("SOCIOPOLITICAL_LANDSCAPE")
            
            # Next the CHANGING_ENERGY_ECONOMIES scenario
            def economiesScenario(values, message):
                while True:
                    x = input(message) 
                    if x in values:
                        economiesScenario.YesOrNo = x
                        # Coefficient changes are filled into the list
                        if x == "Y":
                            # Employment Type
                            if dfCoefficientsFull is not None:
                                if "Type_15_19" in predictorsFull:
                                    index = predictorsFull.index("Type_15_19")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Type_15_19" in predictorsNoWind:
                                    index = predictorsNoWind.index("Type_15_19")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Type_15_19" in predictorsReduced:
                                    index = predictorsReduced.index("Type_15_19")
                                    coefficientChangesReduced[index] = 10 
                            # ISOs
                            if dfCoefficientsFull is not None:
                                if "ISO_YN" in predictorsFull:
                                    index = predictorsFull.index("ISO_YN")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "ISO_YN" in predictorsNoWind:
                                    index = predictorsNoWind.index("ISO_YN")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "ISO_YN" in predictorsReduced:
                                    index = predictorsReduced.index("ISO_YN")
                                    coefficientChangesReduced[index] = 10 
                            # Population Density
                            if dfCoefficientsFull is not None:
                                if "Dens_15_19" in predictorsFull:
                                    index = predictorsFull.index("Dens_15_19")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Dens_15_19" in predictorsNoWind:
                                    index = predictorsNoWind.index("Dens_15_19")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Dens_15_19" in predictorsReduced:
                                    index = predictorsReduced.index("Dens_15_19")
                                    coefficientChangesReduced[index] = 10 
                            # Power Station Age
                            if dfCoefficientsFull is not None:
                                if "Plant_Year" in predictorsFull:
                                    index = predictorsFull.index("Plant_Year")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Plant_Year" in predictorsNoWind:
                                    index = predictorsNoWind.index("Plant_Year")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Plant_Year" in predictorsReduced:
                                    index = predictorsReduced.index("Plant_Year")
                                    coefficientChangesReduced[index] = 10 
                            # Wind Farm Age
                            if dfCoefficientsFull is not None:
                                if "Farm_Year" in predictorsFull:
                                    index = predictorsFull.index("Farm_Year")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Farm_Year" in predictorsNoWind:
                                    index = predictorsNoWind.index("Farm_Year")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Farm_Year" in predictorsReduced:
                                    index = predictorsReduced.index("Farm_Year")
                                    coefficientChangesReduced[index] = 10                             
                            # Unemployment Rate
                            if dfCoefficientsFull is not None:
                                if "Unem_15_19" in predictorsFull:
                                    index = predictorsFull.index("Unem_15_19")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Unem_15_19" in predictorsNoWind:
                                    index = predictorsNoWind.index("Unem_15_19")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Unem_15_19" in predictorsReduced:
                                    index = predictorsReduced.index("Unem_15_19")
                                    coefficientChangesReduced[index] = -10                                
                        break
                    else:
                        print("Invalid value; options are " + str(values))
            economiesScenario(["Y", "N"], "".join(["\nThe CHANGING_ENERGY_ECONOMIES scenario will increase the Employment Type, "
                                                    "\nISOs, Population Density, Power Station Age, and Wind Farm Age"
                                                    "\ncoefficients, and decrease the Unemployment Rate coefficient, "
                                                    "\nby 10% in each 5-year timestep. "
                                                    "\nDo you wish to apply the model's CHANGING_ENERGY_ECONOMIES scenario to the selected predictor configurations? Y or N:\n"]))
            
            CellularAutomaton.economies = economiesScenario.YesOrNo
            
            if CellularAutomaton.economies == "Y":
                scenarioList.append("CHANGING_ENERGY_ECONOMIES")
                
            # Next the NEW_INFRASTRUCTURE scenario
            def infrastructureScenario(values, message):
                while True:
                    x = input(message) 
                    if x in values:
                        infrastructureScenario.YesOrNo = x
                        # Coefficient changes are filled into the list
                        if x == "Y":
                            # Nearest Road
                            if dfCoefficientsFull is not None:
                                if "Near_Roads" in predictorsFull:
                                    index = predictorsFull.index("Near_Roads")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Near_Roads" in predictorsNoWind:
                                    index = predictorsNoWind.index("Near_Roads")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Near_Roads" in predictorsReduced:
                                    index = predictorsReduced.index("Near_Roads")
                                    coefficientChangesReduced[index] = 10
                            # Nearest Transmission Line
                            if dfCoefficientsFull is not None:
                                if "Near_Trans" in predictorsFull:
                                    index = predictorsFull.index("Near_Trans")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Near_Trans" in predictorsNoWind:
                                    index = predictorsNoWind.index("Near_Trans")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Near_Trans" in predictorsReduced:
                                    index = predictorsReduced.index("Near_Trans")
                                    coefficientChangesReduced[index] = 10
                        break
                    else:
                        print("Invalid value; options are " + str(values))
            infrastructureScenario(["Y", "N"], "".join(["\nThe NEW_INFRASTRUCTURE scenario will increase the Nearest Road and "
                                                        "\nNearest Transmission Line coefficients by 10% in each 5-year timestep. "
                                                        "\nDo you wish to apply the model's NEW_INFRASTRUCTURE scenario to the selected predictor configurations? Y or N:\n"]))
            
            CellularAutomaton.infrastructure = infrastructureScenario.YesOrNo
            
            if CellularAutomaton.infrastructure == "Y":
                scenarioList.append("NEW_INFRASTRUCTURE")
                
            # Next the NATURAL_AND_CULTURAL_PROTECTION scenario
            def naturalCulturalScenario(values, message):
                while True:
                    x = input(message) 
                    if x in values:
                        naturalCulturalScenario.YesOrNo = x
                        # Coefficient changes are filled into the list
                        if x == "Y":
                            # Undevelopable Land
                            if dfCoefficientsFull is not None:
                                if "Undev_Land" in predictorsFull:
                                    index = predictorsFull.index("Undev_Land")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Undev_Land" in predictorsNoWind:
                                    index = predictorsNoWind.index("Undev_Land")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Undev_Land" in predictorsReduced:
                                    index = predictorsReduced.index("Undev_Land")
                                    coefficientChangesReduced[index] = -10                                  
                            # Critical Habitats
                            if dfCoefficientsFull is not None:
                                if "Critical" in predictorsFull:
                                    index = predictorsFull.index("Critical")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Critical" in predictorsNoWind:
                                    index = predictorsNoWind.index("Critical")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Critical" in predictorsReduced:
                                    index = predictorsReduced.index("Critical")
                                    coefficientChangesReduced[index] = -10  
                            # Historical Landmarks
                            if dfCoefficientsFull is not None:
                                if "Historical" in predictorsFull:
                                    index = predictorsFull.index("Historical")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Historical" in predictorsNoWind:
                                    index = predictorsNoWind.index("Historical")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Historical" in predictorsReduced:
                                    index = predictorsReduced.index("Historical")
                                    coefficientChangesReduced[index] = -10 
                            # National Parks
                            if dfCoefficientsFull is not None:
                                if "Nat_Parks" in predictorsFull:
                                    index = predictorsFull.index("Nat_Parks")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Nat_Parks" in predictorsNoWind:
                                    index = predictorsNoWind.index("Nat_Parks")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Nat_Parks" in predictorsReduced:
                                    index = predictorsReduced.index("Nat_Parks")
                                    coefficientChangesReduced[index] = -10 
                            # Tribal Land
                            if dfCoefficientsFull is not None:
                                if "Trib_Land" in predictorsFull:
                                    index = predictorsFull.index("Trib_Land")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Trib_Land" in predictorsNoWind:
                                    index = predictorsNoWind.index("Trib_Land")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Trib_Land" in predictorsReduced:
                                    index = predictorsReduced.index("Trib_Land")
                                    coefficientChangesReduced[index] = -10 
                            # Wildlife Refuges
                            if dfCoefficientsFull is not None:
                                if "Wild_Refug" in predictorsFull:
                                    index = predictorsFull.index("Wild_Refug")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Wild_Refug" in predictorsNoWind:
                                    index = predictorsNoWind.index("Wild_Refug")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Wild_Refug" in predictorsReduced:
                                    index = predictorsReduced.index("Wild_Refug")
                                    coefficientChangesReduced[index] = -10                                 
                        break
                    else:
                        print("Invalid value; options are " + str(values))
            naturalCulturalScenario(["Y", "N"], "".join(["\nThe NATURAL_AND_CULTURAL_PROTECTION scenario will decrease the "
                                                          "\nCritical Habitats, Historical Landmarks, National Parks, "
                                                          "\nTribal Land, Undevelopable Land, and Wildlife Refuges "
                                                          "\ncoefficients by 10% in each 5-year timestep. "
                                                          "\nDo you wish to apply the model's NATURAL_AND_CULTURAL_PROTECTION scenario to the selected predictor configurations? Y or N:\n"]))
            
            CellularAutomaton.naturalCultural = naturalCulturalScenario.YesOrNo
            
            if CellularAutomaton.naturalCultural == "Y":
                scenarioList.append("NATURAL_AND_CULTURAL_PROTECTION")
            
            # Finally the URBAN_PROTECTION scenario
            def urbanScenario(values, message):
                while True:
                    x = input(message) 
                    if x in values:
                        urbanScenario.YesOrNo = x
                        # Coefficient changes are filled into the list
                        if x == "Y":
                            # Nearest Airport
                            if dfCoefficientsFull is not None:
                                if "Near_Air" in predictorsFull:
                                    index = predictorsFull.index("Near_Air")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Near_Air" in predictorsNoWind:
                                    index = predictorsNoWind.index("Near_Air")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Near_Air" in predictorsReduced:
                                    index = predictorsReduced.index("Near_Air")
                                    coefficientChangesReduced[index] = 10 
                            # Nearest Hospital
                            if dfCoefficientsFull is not None:
                                if "Near_Hosp" in predictorsFull:
                                    index = predictorsFull.index("Near_Hosp")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Near_Hosp" in predictorsNoWind:
                                    index = predictorsNoWind.index("Near_Hosp")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Near_Hosp" in predictorsReduced:
                                    index = predictorsReduced.index("Near_Hosp")
                                    coefficientChangesReduced[index] = 10 
                            # Nearest Power Plant
                            if dfCoefficientsFull is not None:
                                if "Near_Plant" in predictorsFull:
                                    index = predictorsFull.index("Near_Plant")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Near_Plant" in predictorsNoWind:
                                    index = predictorsNoWind.index("Near_Plant")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Near_Plant" in predictorsReduced:
                                    index = predictorsReduced.index("Near_Plant")
                                    coefficientChangesReduced[index] = 10 
                            # Nearest School
                            if dfCoefficientsFull is not None:
                                if "Near_Sch" in predictorsFull:
                                    index = predictorsFull.index("Near_Sch")
                                    coefficientChangesFull[index] = 10
                            if dfCoefficientsNoWind is not None:
                                if "Near_Sch" in predictorsNoWind:
                                    index = predictorsNoWind.index("Near_Sch")
                                    coefficientChangesNoWind[index] = 10                            
                            if dfCoefficientsReduced is not None:
                                if "Near_Sch" in predictorsReduced:
                                    index = predictorsReduced.index("Near_Sch")
                                    coefficientChangesReduced[index] = 10                                  
                            # Active or Disused Mines
                            if dfCoefficientsFull is not None:
                                if "Mining" in predictorsFull:
                                    index = predictorsFull.index("Mining")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Mining" in predictorsNoWind:
                                    index = predictorsNoWind.index("Mining")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Mining" in predictorsReduced:
                                    index = predictorsReduced.index("Mining")
                                    coefficientChangesReduced[index] = -10 
                            # Military Bases
                            if dfCoefficientsFull is not None:
                                if "Military" in predictorsFull:
                                    index = predictorsFull.index("Military")
                                    coefficientChangesFull[index] = -10
                            if dfCoefficientsNoWind is not None:
                                if "Military" in predictorsNoWind:
                                    index = predictorsNoWind.index("Military")
                                    coefficientChangesNoWind[index] = -10                            
                            if dfCoefficientsReduced is not None:
                                if "Military" in predictorsReduced:
                                    index = predictorsReduced.index("Military")
                                    coefficientChangesReduced[index] = -10                         
                        break
                    else:
                        print("Invalid value; options are " + str(values))
            urbanScenario(["Y", "N"], "".join(["\nThe URBAN_PROTECTION scenario will increase the Nearest Airport, "
                                                "\nNearest Hospital, Nearest Power Plant, and Nearest School coefficients, "
                                                "\nand decrease the Active or Disused Mines and Military Bases coefficients, "
                                                "\nby 10% in each 5-year timestep. "
                                                "\nDo you wish to apply the model's URBAN_PROTECTION scenario to the selected predictor configurations? Y or N:\n"]))
            
            CellularAutomaton.urban = urbanScenario.YesOrNo
            
            if CellularAutomaton.urban == "Y":
                scenarioList.append("URBAN_PROTECTION")
            
            # If the model run is for the CONUS, then the 
            # predictors that are in effect at the nationwide
            # level can also be added as an extra scenario
            if ConstraintNeighborhood.studyArea == "CONUS":
                def nationwideScenario(values, message):
                    while True:
                        x = input(message) 
                        if x in values:
                            nationwideScenario.YesOrNo = x
                            # Coefficient changes are filled into the list
                            if x == "Y":
                                # Green Lobbies
                                if dfCoefficientsFull is not None:
                                    if "Gree_Lobbs" in predictorsFull:
                                        index = predictorsFull.index("Gree_Lobbs")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Gree_Lobbs" in predictorsNoWind:
                                        index = predictorsNoWind.index("Gree_Lobbs")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Gree_Lobbs" in predictorsReduced:
                                        index = predictorsReduced.index("Gree_Lobbs")
                                        coefficientChangesReduced[index] = 10  
                                # Interconnection
                                if dfCoefficientsFull is not None:
                                    if "Interconn" in predictorsFull:
                                        index = predictorsFull.index("Interconn")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Interconn" in predictorsNoWind:
                                        index = predictorsNoWind.index("Interconn")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Interconn" in predictorsReduced:
                                        index = predictorsReduced.index("Interconn")
                                        coefficientChangesReduced[index] = 10 
                                # Investment Tax Credits
                                if dfCoefficientsFull is not None:
                                    if "In_Tax_Cre" in predictorsFull:
                                        index = predictorsFull.index("In_Tax_Cre")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "In_Tax_Cre" in predictorsNoWind:
                                        index = predictorsNoWind.index("In_Tax_Cre")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "In_Tax_Cre" in predictorsReduced:
                                        index = predictorsReduced.index("In_Tax_Cre")
                                        coefficientChangesReduced[index] = 10 
                                # Net Metering
                                if dfCoefficientsFull is not None:
                                    if "Net_Meter" in predictorsFull:
                                        index = predictorsFull.index("Net_Meter")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Net_Meter" in predictorsNoWind:
                                        index = predictorsNoWind.index("Net_Meter")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Net_Meter" in predictorsReduced:
                                        index = predictorsReduced.index("Net_Meter")
                                        coefficientChangesReduced[index] = 10 
                                # Property Tax Exemptions
                                if dfCoefficientsFull is not None:
                                    if "Tax_Prop" in predictorsFull:
                                        index = predictorsFull.index("Tax_Prop")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Tax_Prop" in predictorsNoWind:
                                        index = predictorsNoWind.index("Tax_Prop")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Tax_Prop" in predictorsReduced:
                                        index = predictorsReduced.index("Tax_Prop")
                                        coefficientChangesReduced[index] = 10 
                                # RPS Policy
                                if dfCoefficientsFull is not None:
                                    if "Renew_Port" in predictorsFull:
                                        index = predictorsFull.index("Renew_Port")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Renew_Port" in predictorsNoWind:
                                        index = predictorsNoWind.index("Renew_Port")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Renew_Port" in predictorsReduced:
                                        index = predictorsReduced.index("Renew_Port")
                                        coefficientChangesReduced[index] = 10 
                                # RPS Target
                                if dfCoefficientsFull is not None:
                                    if "Renew_Targ" in predictorsFull:
                                        index = predictorsFull.index("Renew_Targ")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Renew_Targ" in predictorsNoWind:
                                        index = predictorsNoWind.index("Renew_Targ")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Renew_Targ" in predictorsReduced:
                                        index = predictorsReduced.index("Renew_Targ")
                                        coefficientChangesReduced[index] = 10 
                                # Sales Tax Abatements
                                if dfCoefficientsFull is not None:
                                    if "Tax_Sale" in predictorsFull:
                                        index = predictorsFull.index("Tax_Sale")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Tax_Sale" in predictorsNoWind:
                                        index = predictorsNoWind.index("Tax_Sale")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Tax_Sale" in predictorsReduced:
                                        index = predictorsReduced.index("Tax_Sale")
                                        coefficientChangesReduced[index] = 10  
                                # Total Incentives
                                if dfCoefficientsFull is not None:
                                    if "Numb_Incen" in predictorsFull:
                                        index = predictorsFull.index("Numb_Incen")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Numb_Incen" in predictorsNoWind:
                                        index = predictorsNoWind.index("Numb_Incen")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Numb_Incen" in predictorsReduced:
                                        index = predictorsReduced.index("Numb_Incen")
                                        coefficientChangesReduced[index] = 10 
                                # Total Legislation
                                if dfCoefficientsFull is not None:
                                    if "Numb_Pols" in predictorsFull:
                                        index = predictorsFull.index("Numb_Pols")
                                        coefficientChangesFull[index] = 10
                                if dfCoefficientsNoWind is not None:
                                    if "Numb_Pols" in predictorsNoWind:
                                        index = predictorsNoWind.index("Numb_Pols")
                                        coefficientChangesNoWind[index] = 10                            
                                if dfCoefficientsReduced is not None:
                                    if "Numb_Pols" in predictorsReduced:
                                        index = predictorsReduced.index("Numb_Pols")
                                        coefficientChangesReduced[index] = 10                                     
                                # Electricity Cost
                                if dfCoefficientsFull is not None:
                                    if "Cost_15_19" in predictorsFull:
                                        index = predictorsFull.index("Cost_15_19")
                                        coefficientChangesFull[index] = -10
                                if dfCoefficientsNoWind is not None:
                                    if "Cost_15_19" in predictorsNoWind:
                                        index = predictorsNoWind.index("Cost_15_19")
                                        coefficientChangesNoWind[index] = -10                            
                                if dfCoefficientsReduced is not None:
                                    if "Cost_15_19" in predictorsReduced:
                                        index = predictorsReduced.index("Cost_15_19")
                                        coefficientChangesReduced[index] = -10   
                                # Farmland Value
                                if dfCoefficientsFull is not None:
                                    if "Farm_15_19" in predictorsFull:
                                        index = predictorsFull.index("Farm_15_19")
                                        coefficientChangesFull[index] = -10
                                if dfCoefficientsNoWind is not None:
                                    if "Farm_15_19" in predictorsNoWind:
                                        index = predictorsNoWind.index("Farm_15_19")
                                        coefficientChangesNoWind[index] = -10                            
                                if dfCoefficientsReduced is not None:
                                    if "Farm_15_19" in predictorsReduced:
                                        index = predictorsReduced.index("Farm_15_19")
                                        coefficientChangesReduced[index] = -10 
                                # Fossil Fuel Lobbies
                                if dfCoefficientsFull is not None:
                                    if "Foss_Lobbs" in predictorsFull:
                                        index = predictorsFull.index("Foss_Lobbs")
                                        coefficientChangesFull[index] = -10
                                if dfCoefficientsNoWind is not None:
                                    if "Foss_Lobbs" in predictorsNoWind:
                                        index = predictorsNoWind.index("Foss_Lobbs")
                                        coefficientChangesNoWind[index] = -10                            
                                if dfCoefficientsReduced is not None:
                                    if "Foss_Lobbs" in predictorsReduced:
                                        index = predictorsReduced.index("Foss_Lobbs")
                                        coefficientChangesReduced[index] = -10 
                                # Governor Elections
                                if dfCoefficientsFull is not None:
                                    if "Rep_Wins" in predictorsFull:
                                        index = predictorsFull.index("Rep_Wins")
                                        coefficientChangesFull[index] = -10
                                if dfCoefficientsNoWind is not None:
                                    if "Rep_Wins" in predictorsNoWind:
                                        index = predictorsNoWind.index("Rep_Wins")
                                        coefficientChangesNoWind[index] = -10                            
                                if dfCoefficientsReduced is not None:
                                    if "Rep_Wins" in predictorsReduced:
                                        index = predictorsReduced.index("Rep_Wins")
                                        coefficientChangesReduced[index] = -10                                        
                                # Property Value
                                if dfCoefficientsFull is not None:
                                    if "Prop_15_19" in predictorsFull:
                                        index = predictorsFull.index("Prop_15_19")
                                        coefficientChangesFull[index] = -10
                                if dfCoefficientsNoWind is not None:
                                    if "Prop_15_19" in predictorsNoWind:
                                        index = predictorsNoWind.index("Prop_15_19")
                                        coefficientChangesNoWind[index] = -10                            
                                if dfCoefficientsReduced is not None:
                                    if "Prop_15_19" in predictorsReduced:
                                        index = predictorsReduced.index("Prop_15_19")
                                        coefficientChangesReduced[index] = -10    
                            break
                        else:
                            print("Invalid value; options are " + str(values))
                nationwideScenario(["Y", "N"], "".join(["\nThe NATIONWIDE scenario will increase the Green Lobbies, "
                                                        "\nInterconnection, Investment Tax Credits, Net Metering, "
                                                        "\nProperty Tax Exemptions, RPS Policy, RPS Target, "
                                                        "\nSales Tax Abatements, Total Incentives, and Total Legislation "
                                                        "\ncoefficients, and decrease the Electricity Cost, Farmland Value, Fossil Fuel Lobbies, "
                                                        "\nGovernor Elections, and Property Value coefficients, by 10% in each 5-year timestep. "
                                                        "\nDo you wish to use the model's NATIONWIDE scenario to the selected predictor configurations? Y or N:\n"]))
                
                CellularAutomaton.nationwide = nationwideScenario.YesOrNo
                
                if CellularAutomaton.nationwide == "Y":
                    scenarioList.append("NATIONWIDE")
                            
            # The scenarios selected by the user are saved to the console output
            pdf.multi_cell(w=0, h=5.0, align='L', 
                txt="\nThe following are the scenarios selected by the user "
                + "\n"+ "(see Model Instructions for scenario details):"
                +"\n"+ str(scenarioList), border=0)

    ########################## COEFFICIENT ITERATION ##########################
    
    # Based on the predictor configurations selected by the user, the coefficient
    # iteration and probability calculation will iterate through each one
    coefficientList = []
    coeffChangeList = []
    predictorList = []
    interceptList = []
    configList = []

    # The first entry in each of these lists always needs to be a null 
    # configuration, against which any of the four selected configurations
    # can be compared
    coefficientList.append([0])
    coeffChangeList.append([0])
    predictorList.append(["Null"])
    configList.append("Null")
    # Now the four configurations are addressed
    if dfCoefficientsFull is not None:
        coefficientList.append(dfCoefficientsFull["Coefficients"].tolist())
        coeffChangeList.append(coefficientChangesFull)
        predictorList.append(predictorsFull)        
        configList.append("Full")
        interceptList.append(dfInterceptFull["Intercept"][0])
    if dfCoefficientsNoWind is not None:
        coefficientList.append(dfCoefficientsNoWind["Coefficients"].tolist())
        coeffChangeList.append(coefficientChangesNoWind)
        predictorList.append(predictorsNoWind)
        configList.append("No_Wind")
        interceptList.append(dfInterceptNoWind["Intercept"][0])
    if dfCoefficientsWindOnly is not None:
        coefficientList.append(dfCoefficientsWindOnly["Coefficients"].tolist())
        coeffChangeList.append(coefficientChangesWindOnly)
        predictorList.append(predictorsWindOnly)
        configList.append("Wind_Only")
        interceptList.append(dfInterceptWindOnly["Intercept"][0])
    if dfCoefficientsReduced is not None:
        coefficientList.append(dfCoefficientsReduced["Coefficients"].tolist())
        coeffChangeList.append(coefficientChangesReduced)
        predictorList.append(predictorsReduced)
        configList.append("Reduced")
        interceptList.append(dfInterceptReduced["Intercept"][0])
    
    # Beginning of the iteration
    for g in range(len(coeffChangeList)):    
        
        # Subheading for console output
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\n------------------ MODEL PROJECTION: " + str(configList[g]) + " ------------------", border=0)
        
        if configList[g] == "Null":
            print("".join(["\nA version of the model with no predictors (a Null model) is run first..."]))
            # The null model uses only an intercept term, so the term of one of
            # the model configurations is used for it
            if configList[g] == "Null":
                interceptList.insert(0,interceptList[g])
        else:
            print("".join(["\nWind farm site projection using the ", configList[g], " configuration begins..."]))

        # A dataframe is created that will be filled with the coefficients
        # of all predictors for each five-year iteration period. The starting
        # columns contain predictor names, coefficient values from the fitted
        # model, and the change in coefficient values set above by the user
        dfCoeffs = pd.DataFrame()
        dfCoeffs["Predictors"] = predictorList[g]
        dfCoeffs["Coeff_Change_(%)"] = coeffChangeList[g]
        dfCoeffs["Coeff_2020"] = coefficientList[g]               

        # This list will be used for the column names added to the dataframe
        yearList = ["2020","2025","2030","2035","2040","2045","2050"]

        # Based on the scenarios selected by the model user, coefficient
        # values are updated accordingly every five years. The coefficients
        # must thus update themselves 6 times (2025, 2030, 2035, 2040,
        # 2045, 2050)
        for h in range(6):
              
            # This list holds the new coefficients computed for each
            # 5-year period
            newCoefficients = []
              
            # Coefficient updates must be done for each selected scenario
            for i in range(len(dfCoeffs["Coeff_2020"])):
                                
                # If the coefficient change is positive...
                if dfCoeffs["Coeff_Change_(%)"][i] > 0:
                    newCoefficients.append(dfCoeffs["".join(["Coeff_", yearList[h]])][i] + abs(dfCoeffs["".join(["Coeff_", yearList[h]])][i]*(dfCoeffs["Coeff_Change_(%)"][i]/100)))
                # If the coefficient change is negative...
                elif dfCoeffs["Coeff_Change_(%)"][i] < 0:
                    newCoefficients.append(dfCoeffs["".join(["Coeff_", yearList[h]])][i] - abs(dfCoeffs["".join(["Coeff_", yearList[h]])][i]*(dfCoeffs["Coeff_Change_(%)"][i]/100)))
                # If the coefficient was set by the user to not change...
                else:
                    newCoefficients.append(dfCoeffs["".join(["Coeff_", yearList[h]])][i])

            # The new coefficients are added to the dataframe    
            dfCoeffs["".join(["Coeff_", yearList[h+1]])] = newCoefficients
        
        # The dataframe is saved as a csv file
        dfCoeffs.to_csv("".join([directoryPlusCoefficients + "/", studyAreaCheck, "/Future_Coeffs_", configList[g], "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", studyAreaCheck, ".csv"]))
        
        # Write the dataframe to the console output
        if configList[g] != "Null":
            pdf.multi_cell(w=0, h=5.0, align='L', 
                       txt="\nCoefficient changes under the selected scenario when applying the " + str(configList[g]) + " predictor configuration: "
                       +"\n"+ str(dfCoeffs), border=0)
            
        # If the user has selected a state derived from a logistic 
        # regression model run over the CONUS, then the coefficients 
        # for predictors that matter at the nationwide level are
        # removed from the dataframe
        if (ConstraintNeighborhood.studyArea == "Alabama" or ConstraintNeighborhood.studyArea == "Arkansas" or ConstraintNeighborhood.studyArea == "Connecticut"
        or ConstraintNeighborhood.studyArea == "Delaware" or ConstraintNeighborhood.studyArea == "Florida" or ConstraintNeighborhood.studyArea == "Georgia" 
        or ConstraintNeighborhood.studyArea == "Kentucky" or ConstraintNeighborhood.studyArea == "Louisiana" or ConstraintNeighborhood.studyArea == "Mississippi" 
        or ConstraintNeighborhood.studyArea == "New_Jersey" or ConstraintNeighborhood.studyArea == "Rhode_Island" or ConstraintNeighborhood.studyArea == "South_Carolina" 
        or ConstraintNeighborhood.studyArea == "Tennessee" or ConstraintNeighborhood.studyArea == "Virginia"):           
            dfCoeffs = dfCoeffs[(dfCoeffs["Predictors"] != "Cost_15_19") & (dfCoeffs["Predictors"] != "Farm_15_19") &
                        (dfCoeffs["Predictors"] != "Prop_15_19") & (dfCoeffs["Predictors"] != "In_Tax_Cre") &
                        (dfCoeffs["Predictors"] != "Tax_Prop") & (dfCoeffs["Predictors"] != "Tax_Sale") &
                        (dfCoeffs["Predictors"] != "Numb_Incen") & (dfCoeffs["Predictors"] != "Rep_Wins") &
                        (dfCoeffs["Predictors"] != "Interconn") & (dfCoeffs["Predictors"] != "Net_Meter") &
                        (dfCoeffs["Predictors"] != "Renew_Port") & (dfCoeffs["Predictors"] != "Renew_Targ") &
                        (dfCoeffs["Predictors"] != "Numb_Pols") & (dfCoeffs["Predictors"] != "Foss_Lobbs") &
                        (dfCoeffs["Predictors"] != "Gree_Lobbs")].reset_index(drop = True)
        
        ##################### PROBABILITY CALCULATION #####################

        # Filepath to the gridded surface that contains the output from the
        # chosen constraints and neighborhood effects.
        constAndNeighbor = "".join([directoryPlusConstraintsAndNeighborhoods + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_Constraints_Neighborhoods"])
        
        # An attribute table is saved separately within the geodatabase.
        # The attribute table from a previous model run is deleted first
        arcpy.Delete_management("".join([directoryPlusConstraintsAndNeighborhoods + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb/Attribute_Table"]))     
        attTable = arcpy.TableToTable_conversion(constAndNeighbor, "".join([directoryPlusConstraintsAndNeighborhoods + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"]), "Attribute_Table")
        
        # The field names for the predictors held by the gridded surface
        # are assigned to a list
        fields = arcpy.ListFields(attTable)
        fieldList = []
        for field in fields:
            # Fields that do not represent predictors should not be added
            # to this list
            if ("{0}".format(field.name) != "OBJECTID" and "{0}".format(field.name) != "Shape"
            and "{0}".format(field.name) != "Join_Count" and "{0}".format(field.name) != "TARGET_FID"
            and "{0}".format(field.name) != "Wind_Turb" and "{0}".format(field.name) != "Shape_Length"
            and "{0}".format(field.name) != "Shape_Area" and "{0}".format(field.name) != "Constraint"
            and "{0}".format(field.name) != "Neighborhood" and "{0}".format(field.name) != "Neighb_Update"
            and "{0}".format(field.name) != "Probab"):
                fieldList.append("{0}".format(field.name))

        # Predictors that were removed while fitting the logistic regression
        # model should be removed from the gridded surface. Removal is based
        # on which predictors possess coefficients
        for field in fieldList:
            if field not in dfCoeffs["Predictors"].tolist():
                arcpy.DeleteField_management(attTable, [field])
                        
        # The attribute table is converted into a DataFrame, and the 
        # unwanted columns (i.e., columns that aren't predictors) are dropped
        df = DataFrame(TableToNumPyArray("".join([directoryPlusConstraintsAndNeighborhoods + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb/Attribute_Table"]), "*"))
        
        # The dataframe is a list of zeros if the Null configuration is running
        if configList[g] == "Null":
            dfNull = [0]*len(df)
            df = pd.DataFrame(data = dfNull, columns = ["Null"])
        
        # Otherwise, the data are normalized as below
        else:
            df.drop(["OBJECTID","Join_Count","TARGET_FID","Wind_Turb",
                      "Shape_Length","Shape_Area","Constraint","Neighborhood",
                      "Neighb_Update","Probab"], axis=1,inplace=True,errors="ignore")
            
            # Predictors whose values are constant in every grid cell prior
            # to normalization should be dropped
            constantDropped = []
            nunique = df.nunique()
            constantDropped = nunique[nunique == 1].index.tolist()
            # The names of the dropped predictors are written to the console output
            if len(constantDropped) == 0:
                pdf.multi_cell(w=0, h=5.0, align='L', 
                           txt="\nPredictors removed from the model based on having a constant value in all grid cells: None", border=0)
            else:
                pdf.multi_cell(w=0, h=5.0, align='L', 
                           txt="\nPredictors removed from the model based on having a constant value in all grid cells: " + str(constantDropped), border=0)

            # The respective columns are dropped from the dataset
            df = df.drop(columns = constantDropped)            
            
            # The coefficients for these predictors should also be dropped
            dfCoeffs = dfCoeffs[~dfCoeffs["Predictors"].isin(constantDropped)]
     
            # Predictors are split up such that the quantitative ones are normalized,
            # and the categorical ones are assigned a value of either 1 ("Y") or 0 ("N")
            columnNames = df.columns.tolist()
            dfQuantitative = pd.DataFrame()
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
                    dfCategorical[predictor] = df[predictor]
                else:
                    dfQuantitative[predictor] = df[predictor].astype(float)           
    
            # The normalization is executed using standard scores for each grid cell
            dfQuantitative = (dfQuantitative - dfQuantitative.mean())/dfQuantitative.std()
    
            # Categorical predictors are transformed as described above, with any fields
            # containing NaN (or other non-value) also dropped
            dfCategorical = dfCategorical.dropna().replace(to_replace = ["Other","N","Y"], value = [0,0,1])
    
            # The normalized and categorical columns can now be recombined
            df = pd.concat([dfQuantitative,dfCategorical], axis = 1)  
            
            # Both dataframes are alpabetized, necessary for computing
            # updated probabilities for the gridded surface
            df = df.reindex(sorted(df.columns), axis = 1)
            dfCoeffs = dfCoeffs.sort_values("Predictors", ignore_index=True)
                
        # The dataframe is transformed into a nested array
        predictorArrays = [DataFrame(df.iloc[:,i]).to_numpy().reshape(-1,1) for i in range(len(df.columns))]
        dfArray = np.concatenate((predictorArrays[0:len(predictorArrays)]), axis = 1)
                
        # A field is added to the gridded surfaces for updated neighborhood
        # effect factors, the grid cell probabilities, the cell states, and 
        # Getis-Ord statistic results from the logistic regression model run,
        # and the projected wind farm locations. These fields are deleted first
        # if they exist from a previous model run
        arcpy.DeleteField_management(constAndNeighbor, ["Probab","Cell_State","GiPValue","Wind_Turb_Fut"])
        arcpy.AddField_management(constAndNeighbor, "Probab", "DOUBLE")          
        arcpy.AddField_management(constAndNeighbor, "Wind_Turb_Fut", "TEXT")        
        # In states that have fewer than two grid cells with wind farms
        # in them, the Cell_State field amd GiPValue field are not of interest,
        # since they were not derived from state-specific model runs
        if (ConstraintNeighborhood.studyArea != "Alabama" and ConstraintNeighborhood.studyArea != "Arkansas" and ConstraintNeighborhood.studyArea != "Connecticut"
        and ConstraintNeighborhood.studyArea != "Delaware" and ConstraintNeighborhood.studyArea != "Florida" and ConstraintNeighborhood.studyArea != "Georgia" 
        and ConstraintNeighborhood.studyArea != "Kentucky" and ConstraintNeighborhood.studyArea != "Louisiana" and ConstraintNeighborhood.studyArea != "Mississippi" 
        and ConstraintNeighborhood.studyArea != "New_Jersey" and ConstraintNeighborhood.studyArea != "Rhode_Island" and ConstraintNeighborhood.studyArea != "South_Carolina" 
        and ConstraintNeighborhood.studyArea != "Tennessee" and ConstraintNeighborhood.studyArea != "Virginia"):      
            arcpy.AddField_management(constAndNeighbor, "Cell_State", "TEXT")   
            arcpy.AddField_management(constAndNeighbor, "GiPValue", "DOUBLE") 
        
        # If the user wishes to include neighborhood effects, then an updated
        # neighborhood effect factor field is also included
        if ConstraintNeighborhood.NeighYesNo == "N":
            arcpy.DeleteField_management(constAndNeighbor, ["Neighb_Update"])
            arcpy.AddField_management(constAndNeighbor, "Neighb_Update", "DOUBLE")

        # The probabilities computed for each grid cell from fitting the
        # logistic regression model are assigned to dataframes. The 
        # assigned array depends on the predictor configuration
        if configList[g] == "Full":
            wifssSurface = DataFrame(TableToNumPyArray("".join([directoryPlusSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", CellularAutomaton.studyAreaCheck, "_Full.gdb\Attribute_Table"]), "*", skip_nulls = True))
        if configList[g] == "No_Wind":
            wifssSurface = DataFrame(TableToNumPyArray("".join([directoryPlusSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", CellularAutomaton.studyAreaCheck, "_No_Wind.gdb\Attribute_Table"]), "*", skip_nulls = True))
        if configList[g] == "Wind_Only":
            wifssSurface = DataFrame(TableToNumPyArray("".join([directoryPlusSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", CellularAutomaton.studyAreaCheck, "_Wind_Only.gdb\Attribute_Table"]), "*", skip_nulls = True))
        if configList[g] == "Reduced":
            wifssSurface = DataFrame(TableToNumPyArray("".join([directoryPlusSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", CellularAutomaton.studyAreaCheck, "_Reduced.gdb\Attribute_Table"]), "*", skip_nulls = True))
        
        # No fields to add or update if running the null configuration
        if configList[g] != "Null":        
            # The cursor below will iteratively add the probabilities and 
            # predicted grid cell states to their respective empty fields
            iterator1 = iter(wifssSurface["Probab"].tolist())
            iterator2 = iter(wifssSurface["Cell_State"].tolist())
            iterator3 = iter(wifssSurface["GiPValue"].tolist())
                    
            # These new fields are filled, and the number of grid cells currently
            # containing a wind farm is counted. The existing wind farm locations
            # prior to running the cellular automaton are added to the empty
            # Wind_Turb_Fut field. The Cell_Stat and GiPValue fields are excluded
            # if running the model for a state with fewer than two grid cells 
            # containing wind farms
            if (ConstraintNeighborhood.studyArea != "Alabama" and ConstraintNeighborhood.studyArea != "Arkansas" and ConstraintNeighborhood.studyArea != "Connecticut"
            and ConstraintNeighborhood.studyArea != "Delaware" and ConstraintNeighborhood.studyArea != "Florida" and ConstraintNeighborhood.studyArea != "Georgia" 
            and ConstraintNeighborhood.studyArea != "Kentucky" and ConstraintNeighborhood.studyArea != "Louisiana" and ConstraintNeighborhood.studyArea != "Mississippi" 
            and ConstraintNeighborhood.studyArea != "New_Jersey" and ConstraintNeighborhood.studyArea != "Rhode_Island" and ConstraintNeighborhood.studyArea != "South_Carolina" 
            and ConstraintNeighborhood.studyArea != "Tennessee" and ConstraintNeighborhood.studyArea != "Virginia"):    
                fields = ["Probab","Wind_Turb_Fut","Wind_Turb","Cell_State","GiPValue"]
                cursor = UpdateCursor(constAndNeighbor, fields)
                count = 0
                for row in cursor:
                    # Probability, cell state, and Getis-Ord fields are filled
                    row[0] = next(iterator1)
                    row[3] = next(iterator2)
                    row[4] = next(iterator3)
                    # Empty future wind farm field is updated
                    row[1] = row[2]
                    cursor.updateRow(row)
                    # Wind farm count is modified
                    if row[2] == "Y":
                        count = count + 1          
            else:
                fields = ["Probab","Wind_Turb_Fut","Wind_Turb"]
                cursor = UpdateCursor(constAndNeighbor, fields)
                count = 0
                for row in cursor:
                    # Probability, cell state, and Getis-Ord fields are filled
                    row[0] = next(iterator1)                
                    # Empty future wind farm field is updated
                    row[1] = row[2]
                    cursor.updateRow(row)
                    # Wind farm count is modified
                    if row[2] == "Y":
                        count = count + 1
        
        # The null configuration run still needs to know how many grid cells
        # contain a wind farm, and the Wind_Turb_Fut field does need to be filled
        if configList[g] == "Null":
            fields = ["Wind_Turb","Wind_Turb_Fut"]
            cursor = UpdateCursor(constAndNeighbor,fields)
            count = 0
            for row in cursor:
                # Wind_Turb_Fut field is filled
                row[1] = row[0]
                cursor.updateRow(row)
                # Wind farm count is modified
                if row[0] == "Y":
                    count = count + 1
                
            # The cell state and Getis-Ord fields are also not needed
            arcpy.DeleteField_management(constAndNeighbor,["Cell_State","GiPValue"])
            
        # List of the years for which the cellular automaton is executed
        timeSteps = ["2025","2030","2035","2040","2045","2050"]

        # For loop of the cellular automaton that updates probabilities
        # and neighborhood effects
        for f in range(len(timeSteps)):
            
            # Data column from the compiled coefficients is called
            coeff = dfCoeffs["".join(["Coeff_", timeSteps[f]])].tolist()
        
            print("".join(["\nModel iteration for the year ", timeSteps[f], " in progress..."]))
        
            # The updated grid cell probabilities will be held in this list
            probabilityList = []
            probabilityAppend = probabilityList.append
        
            # Updated probabilities are computed in this loop
            for cell in range(len(dfArray)):
                # Probability that each grid cell should contain a wind farm is computed
                # and retained. 
                prob = 1/(1+e**-(interceptList[g] + sum(coeff[0:len(dfArray[0])]*dfArray[cell][0:len(dfArray[0])])))
                probabilityAppend(prob)
                            
            # The probability field of the gridded surface is refilled based on
            # the recomputed probabilities in the loop above
            iterator = iter(probabilityList)
            with UpdateCursor(constAndNeighbor,["Probab"]) as cursor:
                for row in cursor:
                    row[0] = next(iterator)
                    cursor.updateRow(row)

            # The product of the probabilities, the constraint, and the
            # neighborhood effect factor yields the probability of grid cells
            # gaining a wind farm. The name for this new field is defined.
            fieldName = "".join(["Probab_", timeSteps[f]])
            
            # The neighborhood effect factor is excluded from the first
            # iteration of field calculation for states containing fewer
            # than two grid cells with wind farms, for two reasons:
            # 1. The logistic regression model could not run for states
            # with fewer than two such grid cells.
            # 2. The projection of new wind farm locations is assumed to
            # depend on neighboring wind farms with favorable conditions
            # for installation. If wind farms do not yet exist, that
            # favorability does not yet exist either.            
            if timeSteps[f] == "2025" and count < 2:
                # The user make have asked to switch the constraints off
                if ConstraintNeighborhood.ConstYesNo == "Y":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Probab!",field_type = "DOUBLE")            
                if ConstraintNeighborhood.ConstYesNo == "N":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Constraint!*!Probab!",field_type = "DOUBLE")            
            # If there are 2 or more grid cells containing wind farms, 
            # then the neighborhood effects can be included
            elif timeSteps[f] == "2025" and count >= 2:
                # The user may have asked to switch the neighborhood effects
                # and/or the constraints off
                if ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "N":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Constraint!*!Probab!",field_type = "DOUBLE")            
                if ConstraintNeighborhood.NeighYesNo == "N" and ConstraintNeighborhood.ConstYesNo == "Y":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Neighborhood!*!Probab!",field_type = "DOUBLE")            
                if ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "Y":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Probab!",field_type = "DOUBLE")            
                if ConstraintNeighborhood.NeighYesNo == "N" and ConstraintNeighborhood.ConstYesNo == "N":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Constraint!*!Neighborhood!*!Probab!",field_type = "DOUBLE")            
            # After the first iteration, the updated neighborhood effect
            # factors are used instead
            else:
                # The user may have asked to switch the neighborhood effects
                # and/or the constraints off
                if ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "N":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Constraint!*!Probab!",field_type = "DOUBLE")            
                if ConstraintNeighborhood.NeighYesNo == "N" and ConstraintNeighborhood.ConstYesNo == "Y":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Neighb_Update!*!Probab!",field_type = "DOUBLE")            
                if ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "Y":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Probab!",field_type = "DOUBLE")            
                if ConstraintNeighborhood.NeighYesNo == "N" and ConstraintNeighborhood.ConstYesNo == "N":
                    arcpy.CalculateField_management(in_table = constAndNeighbor, field = fieldName, expression = "!Constraint!*!Neighb_Update!*!Probab!",field_type = "DOUBLE")  
                    
            # Grid cells that do and do not contain wind farms are selected
            noFarm = arcpy.SelectLayerByAttribute_management(constAndNeighbor, "New_Selection", where_clause = "Wind_Turb_Fut = 'N'")
            yesFarm = arcpy.SelectLayerByAttribute_management(constAndNeighbor, "New_Selection", where_clause = "Wind_Turb_Fut LIKE '%Y%'")
            
            # Grid cells without wind farms are sorted by the magnitude of 
            # computed probabilities, deleting the sorted field 
            # first if created in a previous model run
            arcpy.Delete_management(constAndNeighbor + "_Sorted")
            sortedCellsPath = constAndNeighbor + "_Sorted"
            sortedCells = arcpy.Sort_management(noFarm, sortedCellsPath, [[fieldName,"DESCENDING"]])
            
            # If grid cells are being projected for states created using
            # data from the CONUS, these states were combined by clipping
            # of the gridded surfaces re-expressed as points.
            if (ConstraintNeighborhood.studyArea == "Louisiana" or ConstraintNeighborhood.studyArea == "Mississippi" or ConstraintNeighborhood.studyArea == "Alabama"
            or ConstraintNeighborhood.studyArea == "Georgia" or ConstraintNeighborhood.studyArea == "South_Carolina" or ConstraintNeighborhood.studyArea == "Kentucky"
            or ConstraintNeighborhood.studyArea == "Arkansas" or ConstraintNeighborhood.studyArea == "Florida" or ConstraintNeighborhood.studyArea == "Virginia"
            or ConstraintNeighborhood.studyArea == "Delaware" or ConstraintNeighborhood.studyArea == "Connecticut" or ConstraintNeighborhood.studyArea == "New_Jersey"
            or ConstraintNeighborhood.studyArea == "Tennessee" or ConstraintNeighborhood.studyArea == "Rhode_Island"):
                # The highest probability grid cells are selected up to the number
                # that the user specified to gain a wind farm every five years
                cellsToConvert = arcpy.SelectLayerByAttribute_management(sortedCells, "New_Selection", where_clause = "".join(["OBJECTID <= ", str(gridCellsChanged)]))
                # The grid cells that are not to be converted are added to a 
                # separate selection
                doNotConvert = arcpy.SelectLayerByAttribute_management(sortedCells, "New_Selection", where_clause = "".join(["OBJECTID > ", str(gridCellsChanged)]))
            
            else:
                # The highest probability grid cells are selected up to the number
                # that the user specified to gain a wind farm every five years
                cellsToConvert = arcpy.SelectLayerByAttribute_management(sortedCells, "New_Selection", where_clause = "".join(["OBJECTID <= ", str(gridCellsChanged)]))
                # The grid cells that are not to be converted are added to a 
                # separate selection
                doNotConvert = arcpy.SelectLayerByAttribute_management(sortedCells, "New_Selection", where_clause = "".join(["OBJECTID > ", str(gridCellsChanged)]))
                
            # This list will hold the cell numbers of the grid cells that
            # gained a wind farm
            cellNoList = []
            
            # If there are no grid cells left to convert to gaining a wind
            # farm because of the model's constraints, this variable's 
            # assigned value will become True. This also depends on whether
            # the user asked to switch off the constraints.
            if ConstraintNeighborhood.ConstYesNo == "N":
                abortScript = None
                
                # Selected grid cells have their Wind_Farm field entry 
                # converted from N to Y(20__)
                with UpdateCursor(cellsToConvert,["Constraint","Wind_Turb_Fut","TARGET_FID"]) as cursor:
                    for row in cursor:
                        # The loop is aborted if there are no grid cells 
                        # left to convert
                        if row[0] == 0:
                            abortScript = True
                            break
                        else:
                            row[1] = "".join(["Y (", timeSteps[f], ")"])
                            cellNoList.append(row[2])
                            cursor.updateRow(row)
            
            # If constraints were switched off, then constrained grid cells
            # place no limit on the number of grid cells that can gain 
            # a wind farm in each iteration
            else:
                with UpdateCursor(cellsToConvert,["Wind_Turb_Fut","TARGET_FID"]) as cursor:
                    for row in cursor:                        
                        row[0] = "".join(["Y (", timeSteps[f], ")"])
                        cellNoList.append(row[1])
                        cursor.updateRow(row)
                            
            # The gridded surface containing the constraints and 
            # neighborhood effect factors is updated, based on the numbers
            # of grid cells that gained wind farms
            with UpdateCursor(constAndNeighbor,["TARGET_FID","Wind_Turb_Fut"]) as cursor:
                for row in cursor:
                    if row[0] in cellNoList:
                        row[1] = "".join(["Y (", timeSteps[f], ")"])
                        cellNoList.append(row[0])
                        cursor.updateRow(row)

            # Having completed this update, all grid cells are recombined,
            # first by creating a geodatabase to hold the updated gridded surface
            # NOTE: Make sure a folder called "Wind_Farm_Future_Locations"
            # has been created in the directory before executing the model.
            if os.path.exists("".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"])) is True:
                # The deleted features depend on whether the user switched off
                # the constraints and neighborhood effects or not
                if ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "N":             
                    arcpy.Delete_management("".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g], "_No_Neighb"]))
                elif ConstraintNeighborhood.NeighYesNo == "N" and ConstraintNeighborhood.ConstYesNo == "Y":             
                    arcpy.Delete_management("".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g], "_No_Const"]))
                elif ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "Y":             
                    arcpy.Delete_management("".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g], "_No_Const_No_Neighb"]))
                else:
                    arcpy.Delete_management("".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g]]))

            # An empty geodatabase is created to hold the new map
            else:
                arcpy.CreateFileGDB_management(directoryPlusFutureSurfaces + "", "".join(["Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb"]))
            
            # A copy of the projected grid cell states is added to a new 
            # folder, to be saved as the final version without predictors
            # in its attribute table. A copy of the final projected gridded
            # surface is also deleted if it exists from a previous run. Again,
            # the decision to switch of the constraints and neighborhood effects
            # alters the filepath
            if ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "N": 
                projectedCells = "".join([directoryPlusFutureSurfaces + "\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g], "_No_Neighb"])
            elif ConstraintNeighborhood.NeighYesNo == "N" and ConstraintNeighborhood.ConstYesNo == "Y":             
                projectedCells = "".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g], "_No_Const"])
            elif ConstraintNeighborhood.NeighYesNo == "Y" and ConstraintNeighborhood.ConstYesNo == "Y":             
                projectedCells = "".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g], "_No_Const_No_Neighb"])
            else:
                projectedCells = "".join([directoryPlusFutureSurfaces + "/Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, ".gdb\Hexagon_Grid_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile_", ConstraintNeighborhood.studyArea, "_", configList[g]])
            arcpy.Delete_management(projectedCells)

            # Only specific fields are wanted in the updated gridded surface,
            # i.e. the predictors are not needed
            fm = arcpy.FieldMappings()
            fm.addTable(constAndNeighbor)
            keepers = ["TARGET_FID","Wind_Turb","Constraint","Neighborhood",
                        "Neighb_Update","Probab","Wind_Turb_Fut","Probab_2025",
                        "Probab_2030","Probab_2035","Probab_2040","Probab_2045",
                        "Probab_2050","Cell_State","GiPValue"]
            # The mapped fields are retained for the merge of the selected
            # grid cells that contain wind farms and those to (not)
            # be converted
            for field in fm.fields:
                if field.name not in keepers:
                    fm.removeFieldMap(fm.findFieldMapIndex(field.name))
            mergedCells = arcpy.Merge_management([yesFarm,cellsToConvert,doNotConvert],projectedCells,field_mappings = (fm))
            
            # This step below is only done when not running the null configuration
            if configList[g] != "Null":            
                # Grid cells that gained wind farms but were initially classified 
                # as false positive or true negative need to be identified, as part
                # of following up on the model's cluster analysis. A new field is 
                # created to hold this classification, and deleted if it already
                # existed. As before, this isn't done if running the model
                # for a state with fewer than two grid cells containing wind farms
                if (ConstraintNeighborhood.studyArea != "Alabama" and ConstraintNeighborhood.studyArea != "Arkansas" and ConstraintNeighborhood.studyArea != "Connecticut"
                and ConstraintNeighborhood.studyArea != "Delaware" and ConstraintNeighborhood.studyArea != "Florida" and ConstraintNeighborhood.studyArea != "Georgia" 
                and ConstraintNeighborhood.studyArea != "Kentucky" and ConstraintNeighborhood.studyArea != "Louisiana" and ConstraintNeighborhood.studyArea != "Mississippi" 
                and ConstraintNeighborhood.studyArea != "New_Jersey" and ConstraintNeighborhood.studyArea != "Rhode_Island" and ConstraintNeighborhood.studyArea != "South_Carolina" 
                and ConstraintNeighborhood.studyArea != "Tennessee" and ConstraintNeighborhood.studyArea != "Virginia"):    
                    arcpy.DeleteField_management(projectedCells, ["Wind_Turb_Clu"])
                    arcpy.AddField_management(projectedCells, "Wind_Turb_Clu", "TEXT")
                    # A cursor is used to fill this field
                    with UpdateCursor(projectedCells,["Wind_Turb_Fut","Cell_State","Wind_Turb_Clu"]) as cursor:
                        for row in cursor:
                            if row[0] != "N":
                                if row[1] == "False_Pos":
                                    row[2] = "Y (False_Pos)"
                                if row[1] == "True_Neg":
                                    row[2] = "Y (True_Neg)"
                                if row[1] == "True_Pos" or row[1] == "False_Neg":
                                    row[2] = "Y"
                            else:
                                row[2] = "N"                                               
                            cursor.updateRow(row)
             
            # A function is constructed to update the neighborhood effects,
            # though there is no need to run it if on the final iteration. The
            # function doesn't run if the user switched the neighborhood
            # effects off
            if timeSteps[f] != "2050" and ConstraintNeighborhood.NeighYesNo == "N":
                def neighborhoodEffects():
                    # A search cursor is used to identify grid cells that neighbor 
                    # the cells that gained a wind farm
                    cursor = SearchCursor(mergedCells, ["Wind_Turb_Fut","TARGET_FID"])
                    
                    # If the feature layer exists from a prior run of the script, it is deleted
                    arcpy.Delete_management("merged_Cells_lyr")
                    
                    # The gridded surface is saved as a feature layer
                    arcpy.MakeFeatureLayer_management(mergedCells, "merged_Cells_lyr")
                    
                    # In order to assess neighborhoods of different number of cells further away
                    # from the cell of interest, search distance based on the grid cell size must
                    # be specified. Grid cells are regular hexagons with resolutions based on
                    # densities ranging from 25 acres/MW to 85 acres/MW and capacities ranging
                    # from 30 MW (20th percentile) to 525 MW (100th percentile). See supporting
                    # notes and the Grid_Cell_Construction script for further details.
                    if (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "100"): 
                        # 44,625 acres, or 180,590,968 square meters
                        area = 180590968
                    elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "100"): 
                        # 34,125 acres, or 138,098,975 square meters
                        area = 138098975
                    elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "100"): 
                        # 23,625 acres, or 95,606,983 square meters
                        area = 95606983
                    elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "100"): 
                        # 13,125 acres, or 53,114,991 square meters
                        area = 53114991
                    elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "80"): 
                        # 17,127.5 acres, or 69,312,533 square meters
                        area = 69312533
                    elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "80"): 
                        # 13,097.5 acres, or 53,003,702 square meters
                        area = 53003702
                    elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "80"): 
                        # 9,067.5 acres, or 36,694,871 square meters
                        area = 36694871
                    elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "80"): 
                        # 5,037.5 acres, or 20,386,039 square meters
                        area = 20386039
                    elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "60"): 
                        # 12,750 acres, or 51,597,419 square meters
                        area = 51597419
                    elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "60"): 
                        # 9,750 acres, or 39,456,850 square meters
                        area = 39456850
                    elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "60"): 
                        # 6,750 acres, or 27,316,281 square meters
                        area = 27316281
                    elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "60"): 
                        # 3,750 acres, or 15,175,712 square meters
                        area = 15175712
                    elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "40"): 
                        # 7,650 acres, or 30,958,452 square meters
                        area = 30958452
                    elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "40"): 
                        # 5,850 acres, or 23,674,110 square meters
                        area = 23674110
                    elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "40"): 
                        # 4,050 acres, or 16,389,769 square meters
                        area = 16389769
                    elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "40"): 
                        # 2,250 acres, or 9,105,427 square meters
                        area = 9105427
                    elif (ConstraintNeighborhood.density == "85" and ConstraintNeighborhood.capacity == "20"): 
                        # 2,550 acres, or 10,319,484 square meters
                        area = 10319484
                    elif (ConstraintNeighborhood.density == "65" and ConstraintNeighborhood.capacity == "20"): 
                        # 1,950 acres, or 7,891,370 square meters
                        area = 7891370
                    elif (ConstraintNeighborhood.density == "45" and ConstraintNeighborhood.capacity == "20"): 
                        # 1,350 acres, or 5,463,256 square meters
                        area = 5463256
                    elif (ConstraintNeighborhood.density == "25" and ConstraintNeighborhood.capacity == "20"): 
                        # 750 acres, or 3,035,142 square meters
                        area = 3035142
                        
                    # The length of one side of a hexagonal grid cell
                    sideLength = sqrt(2*area/(3*sqrt(3)))
                    # This length is doubled to compute the distance between two grid cell
                    # centroids for grid range evaluation
                    distance = sideLength*2*sin(radians(60))
                
                    print("".join(["\nPlease wait while the neighborhood effect factors of the grid cells surrounding the ", str(gridCellsChanged), " cell(s) that gained a wind farm are updated..."]))
                    
                    gridCellList = []
                    neighborhoodList = []
                    
                    # The cursor is used to iterate over all grid cells                f
                    for row in cursor:
                        
                        # If the cursor reaches a grid cell that gained a wind
                        # farm, its neighboring grid cells are identified
                        gridCell = int(row[1])

                        if row[0] == "".join(["Y (", timeSteps[f], ")"]):
                            sql = "".join(["Wind_Turb_Fut = '", row[0],"' AND TARGET_FID = ", str(gridCell)])
                            adjacent = arcpy.SelectLayerByLocation_management("merged_Cells_lyr", "HAVE_THEIR_CENTER_IN", arcpy.SelectLayerByAttribute_management("merged_Cells_lyr", "New_Selection", sql), "".join([str(distance*int(ConstraintNeighborhood.neighborhoodSize)), " meters"]))

                            # A feature layer composed of the identified grid cells
                            # is made, and deleted beforehand if previously created
                            arcpy.Delete_management("adjacent_lyr")
                            arcpy.MakeFeatureLayer_management(adjacent, "adjacent_lyr")
                                                            
                            # The neighborhood effect factors of these neighboring
                            # grid cells are then updated; on the first 
                            # timestep, the original neighborhood effects 
                            # are used
                            if timeSteps[f] == "2025":
                                subCursor = UpdateCursor(adjacent,["TARGET_FID","Neighborhood"])
                            # After the first time step, the updated ones
                            # are used insted
                            else:
                                subCursor = UpdateCursor(adjacent,["TARGET_FID","Neighb_Update"])
                            for row1 in tqdm(subCursor):
                                gridCell = int(row1[0])
                                sql = "".join(["TARGET_FID = ", str(gridCell)])
                                neighbors = arcpy.SelectLayerByLocation_management("adjacent_lyr", "HAVE_THEIR_CENTER_IN", arcpy.SelectLayerByAttribute_management("adjacent_lyr", "New_Selection", sql), "".join([str(distance*int(ConstraintNeighborhood.neighborhoodSize)), " meters"]))
                                
                                # The total number of neighboring grid cells is totaled by subtracting
                                # the central grid cell
                                totalNeighbors = int(neighbors[2]) - 1
                                # A new sub-cursor is used to sum the number of neighboring grid cells that
                                # contain a wind farm
                                windFarmCount = 0
                                subSubCursor = SearchCursor(neighbors,["Wind_Turb_Fut","TARGET_FID"])
                                for row2 in subSubCursor:
                                    if "Y" in row2[0]: 
                                        # The central grid cell should not be included in the sum
                                        if row2[1] != gridCell:
                                            windFarmCount += 1
                                # The updated neighborhood effect factor for the
                                # grid cell is computed and added to the 
                                # attribute table
                                row1[1] = windFarmCount/totalNeighbors
                                
                                # The grid cells whose neighborhood effect
                                # factors are updated are appended to 
                                # the lists above
                                gridCellList.append(int(row1[0]))
                                neighborhoodList.append(row1[1])
                    
                    # The updated neighborhood effect factors can now be
                    # added to the gridded surface
                    cursor = UpdateCursor(constAndNeighbor, ["TARGET_FID","Neighborhood","Neighb_Update"])
                    for row in cursor:
                        for j in range(len(gridCellList)):
                            if row[0] == gridCellList[j]:
                                row[2] = neighborhoodList[j]
                                cursor.updateRow(row)
                        # If the neighborhood effect factor didn't
                        # change, its original value is retained
                        if row[2] is None:
                              row[2] = row[1]
                              cursor.updateRow(row)                            

                # The function is called
                neighborhoodEffects()
            
            # Temporary files can be deleted
            arcpy.Delete_management(constAndNeighbor + "_Sorted")
            
            # The cellular automaton stops if there are no grid cells left
            # to gain a wind farm. This again depends on whether or not the
            # user switched off the constraints
            if ConstraintNeighborhood.ConstYesNo == "N":
                if abortScript == True:
                    print('''\nDue to the model's constraints, and lack of grid cells with neighboring '''
                          '''wind farms, no more grid cells can be projected to gain a wind farm.''')
                    
                    pdf.multi_cell(w=0, h=5.0, align='L', 
                                  txt="\nThe model iterations stopped in the " + str(f+1) + "th (" + timeSteps[f] + ") step due to the model's "
                                  +"\n"+ "constraints, and/or lack of grid cells with neighboring wind farms. ", border = 0)
                    break                    
            
        print("".join(["\nWind farm site projection using the ", configList[g], " configuration: Complete."]))
        
        # Filepath to the constructed hexagonal grid map is provided for the
        # console output
        pdf.multi_cell(w=0, h=5.0, align='L', 
                      txt="\nFilepath to the constructed hexagonal grid map: "
                      +"\n\n"+constAndNeighbor, border=0)    
        
        ################ QUANTITY AND ALLOCATION DISAGREEMENT #####################
        
        # The differences in the locations of grid cells projected to gain
        # wind farms can be quantified using quantity and allocation disagreement
        # Firstly, the number of grid cells that fall into each classification
        # when using the null configuration is counted
        if configList[g] == "Null":
            NCountNull = []
            YCount2025Null = []
            YCount2030Null = []
            YCount2035Null = []
            YCount2040Null = []
            YCount2045Null = []
            YCount2050Null = []
            cursor = SearchCursor(projectedCells, ["Wind_Turb_Fut","TARGET_FID"])
            for row in cursor:
                if row[0] == "N":
                    NCountNull.append(row[1])
                if row[0] == "Y (2025)":
                    YCount2025Null.append(row[1])
                if row[0] == "Y (2030)":
                    YCount2030Null.append(row[1])
                if row[0] == "Y (2035)":
                    YCount2035Null.append(row[1])
                if row[0] == "Y (2040)":
                    YCount2040Null.append(row[1])
                if row[0] == "Y (2045)":
                    YCount2045Null.append(row[1])
                if row[0] == "Y (2050)":
                    YCount2050Null.append(row[1])            
        # The process is repeated for the current configuration, and the 
        # disagreement tables can now be made
        else:
            NCount = []
            YCount2025 = []
            YCount2030 = []
            YCount2035 = []
            YCount2040 = []
            YCount2045 = []
            YCount2050 = []
            cursor = SearchCursor(projectedCells, ["Wind_Turb_Fut","TARGET_FID"])
            for row in cursor:
                if row[0] == "N":
                    NCount.append(row[1])
                if row[0] == "Y (2025)":
                    YCount2025.append(row[1])
                if row[0] == "Y (2030)":
                    YCount2030.append(row[1])
                if row[0] == "Y (2035)":
                    YCount2035.append(row[1])
                if row[0] == "Y (2040)":
                    YCount2040.append(row[1])
                if row[0] == "Y (2045)":
                    YCount2045.append(row[1])
                if row[0] == "Y (2050)":
                    YCount2050.append(row[1])        
            
            # For each iteration of the cellular automaton, the number of grid
            # cells that did and did not gain wind farms is counted. The purpose
            # is to contrast between two predictor configurations when and where
            # wind farms were gained, starting with those that did not gain one
            bothNo = 0
            noNullOther2025 = 0
            noNullOther2030 = 0
            noNullOther2035 = 0
            noNullOther2040 = 0
            noNullOther2045 = 0
            noNullOther2050 = 0
            for i in range(len(NCountNull)):
                if NCountNull[i] in NCount:
                    bothNo = bothNo + 1
                if NCountNull[i] in YCount2025:
                    noNullOther2025 = noNullOther2025 + 1
                if NCountNull[i] in YCount2030:
                    noNullOther2030 = noNullOther2030 + 1
                if NCountNull[i] in YCount2035:
                    noNullOther2035 = noNullOther2035 + 1
                if NCountNull[i] in YCount2040:
                    noNullOther2040 = noNullOther2040 + 1
                if NCountNull[i] in YCount2045:
                    noNullOther2045 = noNullOther2045 + 1
                if NCountNull[i] in YCount2050:
                    noNullOther2050 = noNullOther2050 + 1
            sumNoFarm = bothNo + noNullOther2025 + noNullOther2030 + noNullOther2035 + noNullOther2040 + noNullOther2045 + noNullOther2050
            listNo = [bothNo, noNullOther2025, noNullOther2030, noNullOther2035, noNullOther2040, noNullOther2045, noNullOther2050, sumNoFarm]
            
            # For grid cells that gained wind farms in 2025...
            both2025 = 0
            yes2025OtherNo = 0
            yes2025Other2030 = 0
            yes2025Other2035 = 0
            yes2025Other2040 = 0
            yes2025Other2045 = 0
            yes2025Other2050 = 0
            for i in range(len(YCount2025Null)):
                if YCount2025Null[i] in NCount:
                    yes2025OtherNo = yes2025OtherNo + 1
                if YCount2025Null[i] in YCount2025:
                    both2025 = both2025 + 1
                if YCount2025Null[i] in YCount2030:
                    yes2025Other2030 = yes2025Other2030 + 1
                if YCount2025Null[i] in YCount2035:
                    yes2025Other2035 = yes2025Other2035 + 1
                if YCount2025Null[i] in YCount2040:
                    yes2025Other2040 = yes2025Other2040 + 1
                if YCount2025Null[i] in YCount2045:
                    yes2025Other2045 = yes2025Other2045 + 1
                if YCount2025Null[i] in YCount2050:
                    yes2025Other2050 = yes2025Other2050 + 1
            sumYes2025 = yes2025OtherNo + both2025 + yes2025Other2030 + yes2025Other2035 + yes2025Other2040 + yes2025Other2045 + yes2025Other2050
            list2025 = [yes2025OtherNo, both2025, yes2025Other2030, yes2025Other2035, yes2025Other2040, yes2025Other2045, yes2025Other2050, sumYes2025]
            
            # For grid cells that gained wind farms in 2030...
            both2030 = 0
            yes2030OtherNo = 0
            yes2030Other2025 = 0
            yes2030Other2035 = 0
            yes2030Other2040 = 0
            yes2030Other2045 = 0
            yes2030Other2050 = 0
            for i in range(len(YCount2030Null)):
                if YCount2030Null[i] in NCount:
                    yes2030OtherNo = yes2030OtherNo + 1
                if YCount2030Null[i] in YCount2025:
                    yes2030Other2025 = yes2030Other2025 + 1
                if YCount2030Null[i] in YCount2030:
                    both2030 = both2030 + 1
                if YCount2030Null[i] in YCount2035:
                    yes2030Other2035 = yes2030Other2035 + 1
                if YCount2030Null[i] in YCount2040:
                    yes2030Other2040 = yes2030Other2040 + 1
                if YCount2030Null[i] in YCount2045:
                    yes2030Other2045 = yes2030Other2045 + 1
                if YCount2030Null[i] in YCount2050:
                    yes2030Other2050 = yes2030Other2050 + 1
            sumYes2030 = yes2030OtherNo + yes2030Other2025 + both2030 + yes2030Other2035 + yes2030Other2040 + yes2030Other2045 + yes2030Other2050
            list2030 = [yes2030OtherNo, yes2030Other2025, both2030, yes2030Other2035, yes2030Other2040, yes2030Other2045, yes2030Other2050, sumYes2030]
            
            # For grid cells that gained wind farms in 2035...
            both2035 = 0
            yes2035OtherNo = 0
            yes2035Other2025 = 0
            yes2035Other2030 = 0
            yes2035Other2040 = 0
            yes2035Other2045 = 0
            yes2035Other2050 = 0
            for i in range(len(YCount2035Null)):
                if YCount2035Null[i] in NCount:
                    yes2035OtherNo = yes2035OtherNo + 1
                if YCount2035Null[i] in YCount2025:
                    yes2035Other2025 = yes2035Other2025 + 1
                if YCount2035Null[i] in YCount2030:
                    yes2035Other2030 = yes2035Other2030 + 1
                if YCount2035Null[i] in YCount2035:
                    both2035 = both2035 + 1
                if YCount2035Null[i] in YCount2040:
                    yes2035Other2040 = yes2035Other2040 + 1
                if YCount2035Null[i] in YCount2045:
                    yes2035Other2045 = yes2035Other2045 + 1
                if YCount2035Null[i] in YCount2050:
                    yes2035Other2050 = yes2035Other2050 + 1
            sumYes2035 = yes2035OtherNo + yes2035Other2025 + yes2035Other2030 + both2035 + yes2035Other2040 + yes2035Other2045 + yes2035Other2050
            list2035 = [yes2035OtherNo, yes2035Other2025, yes2035Other2030, both2035, yes2035Other2040, yes2035Other2045, yes2035Other2050, sumYes2035]
            
            # For grid cells that gained wind farms in 2040...
            both2040 = 0
            yes2040OtherNo = 0
            yes2040Other2025 = 0
            yes2040Other2030 = 0
            yes2040Other2035 = 0
            yes2040Other2045 = 0
            yes2040Other2050 = 0
            for i in range(len(YCount2040Null)):
                if YCount2040Null[i] in NCount:
                    yes2040OtherNo = yes2040OtherNo + 1
                if YCount2040Null[i] in YCount2025:
                    yes2040Other2025 = yes2040Other2025 + 1
                if YCount2040Null[i] in YCount2030:
                    yes2040Other2030 = yes2040Other2030 + 1
                if YCount2040Null[i] in YCount2035:
                    yes2040Other2035 = yes2040Other2035 + 1
                if YCount2040Null[i] in YCount2040:
                    both2040 = both2040 + 1
                if YCount2040Null[i] in YCount2045:
                    yes2040Other2045 = yes2040Other2045 + 1
                if YCount2040Null[i] in YCount2050:
                    yes2040Other2050 = yes2040Other2050 + 1
            sumYes2040 = yes2040OtherNo + yes2040Other2025 + yes2040Other2030 + yes2040Other2035 + both2040 + yes2040Other2045 + yes2040Other2050
            list2040 = [yes2040OtherNo, yes2040Other2025, yes2040Other2030, yes2040Other2035, both2040, yes2040Other2045, yes2040Other2050, sumYes2040]
            
            # For grid cells that gained wind farms in 2045...
            both2045 = 0
            yes2045OtherNo = 0
            yes2045Other2025 = 0
            yes2045Other2030 = 0
            yes2045Other2035 = 0
            yes2045Other2040 = 0
            yes2045Other2050 = 0
            for i in range(len(YCount2045Null)):
                if YCount2045Null[i] in NCount:
                    yes2045OtherNo = yes2045OtherNo + 1
                if YCount2045Null[i] in YCount2025:
                    yes2045Other2025 = yes2045Other2025 + 1
                if YCount2045Null[i] in YCount2030:
                    yes2045Other2030 = yes2045Other2030 + 1
                if YCount2045Null[i] in YCount2035:
                    yes2045Other2035 = yes2045Other2035 + 1
                if YCount2045Null[i] in YCount2040:
                    yes2045Other2040 = yes2045Other2040 + 1
                if YCount2045Null[i] in YCount2045:
                    both2045 = both2045 + 1
                if YCount2045Null[i] in YCount2050:
                    yes2045Other2050 = yes2045Other2050 + 1
            sumYes2045 = yes2045OtherNo + yes2045Other2025 + yes2045Other2030 + yes2045Other2035 + yes2045Other2040 + both2045 + yes2045Other2050
            list2045 = [yes2045OtherNo, yes2045Other2025, yes2045Other2030, yes2045Other2035, yes2045Other2040, both2045, yes2045Other2050, sumYes2045]
            
            # For grid cells that gained wind farms in 2050...
            both2050 = 0
            yes2050OtherNo = 0
            yes2050Other2025 = 0
            yes2050Other2030 = 0
            yes2050Other2035 = 0
            yes2050Other2040 = 0
            yes2050Other2045 = 0        
            for i in range(len(YCount2050Null)):
                if YCount2050Null[i] in NCount:
                    yes2050OtherNo = yes2050OtherNo + 1
                if YCount2050Null[i] in YCount2025:
                    yes2050Other2025 = yes2050Other2025 + 1
                if YCount2050Null[i] in YCount2030:
                    yes2050Other2030 = yes2050Other2030 + 1
                if YCount2050Null[i] in YCount2035:
                    yes2050Other2035 = yes2050Other2035 + 1
                if YCount2050Null[i] in YCount2040:
                    yes2050Other2040 = yes2050Other2040 + 1
                if YCount2050Null[i] in YCount2045:
                    yes2050Other2045 = yes2050Other2045 + 1
                if YCount2050Null[i] in YCount2050:
                    both2050 = both2050 + 1
            sumYes2050 = yes2050OtherNo + yes2050Other2025 + yes2050Other2030 + yes2050Other2035 + yes2050Other2040 + yes2050Other2045 + both2050 
            list2050 = [yes2050OtherNo, yes2050Other2025, yes2050Other2030, yes2050Other2035, yes2050Other2040, yes2050Other2045, both2050, sumYes2050]
    
            # The number of grid cells in each group is counted to obtain a final total
            totalCells = sumNoFarm + sumYes2025 + sumYes2030 + sumYes2035 + sumYes2040 + sumYes2045 + sumYes2050
            # The final total and the number of grid cells that gained (or did not gain)
            # wind farms is used to define another list
            bottomRow = [(bothNo+yes2025OtherNo+yes2030OtherNo+yes2035OtherNo+yes2040OtherNo+yes2045OtherNo+yes2050OtherNo),
                         (noNullOther2025+both2025+yes2030Other2025+yes2035Other2025+yes2040Other2025+yes2045Other2025+yes2050Other2025),             
                         (noNullOther2030+yes2025Other2030+both2030+yes2035Other2030+yes2040Other2030+yes2045Other2030+yes2050Other2030),             
                         (noNullOther2035+yes2025Other2035+yes2030Other2035+both2035+yes2040Other2035+yes2045Other2035+yes2050Other2035),             
                         (noNullOther2040+yes2025Other2040+yes2030Other2040+yes2035Other2040+both2040+yes2045Other2040+yes2050Other2040),             
                         (noNullOther2045+yes2025Other2045+yes2030Other2045+yes2035Other2045+yes2040Other2045+both2045+yes2050Other2045),             
                         (noNullOther2050+yes2025Other2050+yes2030Other2050+yes2035Other2050+yes2040Other2050+yes2045Other2050+both2050),             
                         totalCells]            
            # A list of lists for all rows of the table produced below
            listOfLists = [listNo,list2025,list2030,list2035,list2040,list2045,list2050,bottomRow]
            # The list of lists is transposed for creating the final table
            transList = [list(i) for i in zip(*listOfLists)]
            
            # Columns, rows, and colour schemes of the table are defined
            columns = ("$\\bf{No Farm}$", "$\\bf{Y (2025)}$", "$\\bf{Y (2030)}$", "$\\bf{Y (2035)}$",
                       "$\\bf{Y (2040)}$", "$\\bf{Y (2045)}$", "$\\bf{Y (2050)}$", "$\\bf{Sum}$")
            rows = ["$\\bf{No Farm}$", "$\\bf{Y (2025)}$", "$\\bf{Y (2030)}$", "$\\bf{Y (2035)}$",
                    "$\\bf{Y (2040)}$", "$\\bf{Y (2045)}$", "$\\bf{Y (2050)}$", "$\\bf{Sum}$"]
            colors = [["w","w","w","w","w","w","w","#56b5fd"],["w","w","w","w","w","w","w","#56b5fd"],
                      ["w","w","w","w","w","w","w","#56b5fd"],["w","w","w","w","w","w","w","#56b5fd"],
                      ["w","w","w","w","w","w","w","#56b5fd"],["w","w","w","w","w","w","w","#56b5fd"],
                      ["w","w","w","w","w","w","w","#56b5fd"],["#56b5fd","#56b5fd","#56b5fd","#56b5fd",
                                                               "#56b5fd","#56b5fd","#56b5fd","#56b5fd"]]
            
            # The table is constructed
            fig,ax = plt.subplots(figsize =(10,10))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=transList,rowLabels=rows,cellLoc = 'center',cellColours =colors,
                                 colLabels=columns,loc='center',rowLoc ='center',colWidths = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
            ax.text(-0.04,0.047,"Grid Cell Projections (Null Configuration)",fontsize = 20)
            ax.text(-0.062,-0.045,"".join(["Grid Cell Projections (", configList[g], " Configuration)"]),fontsize = 20,rotation=90)
            table.set_fontsize(20)
            table.scale(1, 4)


            # Computation of quantity disagreement, using the formulae presented 
            # by Pontius and Millones (2011) and Feizizadeh et al. (2022)        
            quantDis = (abs(bottomRow[0]-transList[0][7]) + abs(bottomRow[1]-transList[1][7])
                      + abs(bottomRow[2]-transList[2][7]) + abs(bottomRow[3]-transList[3][7])
                      + abs(bottomRow[4]-transList[4][7]) + abs(bottomRow[5]-transList[5][7])
                      + abs(bottomRow[6]-transList[6][7]))/2
            # An alternative quantity disagreement to check for errors in its calculation
            quantDisStar = abs(((transList[0][0] + transList[0][1] + transList[0][2] + transList[0][3] + transList[0][4] + transList[0][5] + transList[0][6])
                              + (transList[1][0] + transList[1][1] + transList[1][2] + transList[1][3] + transList[1][4] + transList[1][5] + transList[1][6])
                              + (transList[2][0] + transList[2][1] + transList[2][2] + transList[2][3] + transList[2][4] + transList[2][5] + transList[2][6])
                              + (transList[3][0] + transList[3][1] + transList[3][2] + transList[3][3] + transList[3][4] + transList[3][5] + transList[3][6])
                              + (transList[4][0] + transList[4][1] + transList[4][2] + transList[4][3] + transList[4][4] + transList[4][5] + transList[4][6])
                              + (transList[5][0] + transList[5][1] + transList[5][2] + transList[5][3] + transList[5][4] + transList[5][5] + transList[5][6]))
                              - ((transList[0][0] + transList[1][0] + transList[2][0] + transList[3][0] + transList[4][0] + transList[5][0] + transList[6][0])
                              + (transList[0][1] + transList[1][1] + transList[2][1] + transList[3][1] + transList[4][1] + transList[5][1] + transList[6][1])
                              + (transList[0][2] + transList[1][2] + transList[2][2] + transList[3][2] + transList[4][2] + transList[5][2] + transList[6][2])
                              + (transList[0][3] + transList[1][3] + transList[2][3] + transList[3][3] + transList[4][3] + transList[5][3] + transList[6][3])
                              + (transList[0][4] + transList[1][4] + transList[2][4] + transList[3][4] + transList[4][4] + transList[5][4] + transList[6][4])
                              + (transList[0][5] + transList[1][5] + transList[2][5] + transList[3][5] + transList[4][5] + transList[5][5] + transList[6][5])))

            # Computation of allocation disagreement using the same references
            absoDis = (2*min(bottomRow[0]-transList[0][0],transList[0][7]-transList[0][0]) + 2*min(bottomRow[1]-transList[1][1],transList[1][7]-transList[1][1])
                    + 2*min(bottomRow[2]-transList[2][2],transList[2][7]-transList[2][2]) + 2*min(bottomRow[3]-transList[3][3],transList[3][7]-transList[3][3])
                    + 2*min(bottomRow[4]-transList[4][4],transList[4][7]-transList[4][4]) + 2*min(bottomRow[5]-transList[5][5],transList[5][7]-transList[5][5])
                    + 2*min(bottomRow[6]-transList[6][6],transList[6][7]-transList[6][6]))/2
            
            # If Q and Q* are not equal, then Q* is used instead of Q, and the 
            # value of allocation disagreement is amended
            if quantDis != quantDisStar:
                qFinal = quantDisStar
                aFinal = absoDis + abs(quantDis - quantDisStar)            
            else:
                qFinal = quantDis
                aFinal = absoDis
                
            # Quantity Allocation Disagreement Index is calculated. A value closer
            # to 1 suggests greater disagreement between projections obtained from
            # the predictor configurations
            qadiIndex = np.sqrt((aFinal/totalCells)**2+(qFinal/totalCells)**2)
            
            # Metrics are added to the table
            ax.text(-0.05,-0.05,"".join(["Quantity Disagreement: ", str(int(qFinal))]),fontsize = 14, color = "r", style = "italic")
            ax.text(0,-0.05,"".join(["Allocation Disagreement: ", str(int(aFinal))]),fontsize = 14, color = "r", style = "italic")
            ax.text(-0.02,-0.055,"".join(["QADI Index: ", str(round(qadiIndex,3))]),fontsize = 14, color = "r", style = "italic")
            
            # A low-resolution version of the QADI table is saved to the console output
            qadiFilepath = "".join([directoryPlusQADI + "/", ConstraintNeighborhood.studyArea, "/", ConstraintNeighborhood.studyArea, "_", configList[g], "_", ConstraintNeighborhood.density, "_acres_per_MW_", ConstraintNeighborhood.capacity, "th_percentile.png"])
            fig.tight_layout()
            fig.savefig(qadiFilepath, dpi = 50, bbox_inches = 'tight')
            pdf.multi_cell(w=0, h=5.0, align='L', 
                          txt="\n\nQADI table produced by comparing projections forced by the "
                          +"\n"+"Null versus " + str(configList[g]) + " predictor configurations: ", border=0)
            pdf.image(qadiFilepath, w = 150, h = 150)
            
            # The QADI table is re-saved as a high-resolution version
            plt.tight_layout()
            plt.savefig(qadiFilepath, dpi = 300)
            plt.clf()
            
    # Console output is written to a PDF
    pdf.output(directory + '/Cellular_Automata_Console_Output.pdf', 'F')
    
CellularAutomaton()