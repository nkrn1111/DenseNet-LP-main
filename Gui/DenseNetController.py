import joblib
import pandas as pd
from DenseNetGui import *
import PreProcess

def init(top_level, w, *args, **kwargs):
    global gui, top, root
    gui = w
    top = top_level
    root = top_level


# ==============================================================================
def OpenTestDataFile():
    ''' function to select, upload and read test data file '''
    data_file_name = ""
    read_file = ""
    print("================================================")

    # upload csv and excel (xlsx) dependent data file
    file_opt = options = {}
    options['filetypes'] = [("CSV files", "*.csv")]

    # open dialog box - opne and read selected file
    test_data_file_dialogbox = filedialog.askopenfile(mode="rb", **file_opt)

    if test_data_file_dialogbox == None:
        data_file_name = "empty"
        read_file = "empty"

    if test_data_file_dialogbox != None:

        data_file_name = test_data_file_dialogbox.name
        print(data_file_name)
        try:
            # Use index_col = [0] to add column to use as the row labels of
            # the DataFrame and then transpose
            read_file = pd.read_csv(
                test_data_file_dialogbox, index_col=[0]).transpose()
        except:
            read_file = "empty"

    return data_file_name, read_file


# ==============================================================================
# Function UploadAlreadyTrainedModel: used to upload PKL based file.
# PKL file include:
# index [0]: Features for Data Training
# index [1]: Classifier
# index [2]: Classifier confusion matrix, accuracy and error

# ==============================================================================
def UploadAlreadyTrainedModelWindowTwo():
    # globalize both pkl data file and its name, the file name will show in user
    # "current data upload" window which show user Log History for any file upload.

    AlreadyTrainedModel_name = ""
    AlreadyTrainedModel_joblib = ""

    file_opt = options = {}
    options['filetypes'] = [("PKL files", "*.txt")]

    # open dialog box - opne and read selected file
    AlreadyTrainedModel_ReadFile = filedialog.askopenfile(mode="rb", **file_opt)
    print(AlreadyTrainedModel_ReadFile.name)
    # if user dosen't upload file then don't print error messgae
    if AlreadyTrainedModel_ReadFile is None:
        AlreadyTrainedModel_name = "Not Uploaded"

    if AlreadyTrainedModel_ReadFile is not None:
        AlreadyTrainedModel_name = AlreadyTrainedModel_ReadFile.name

    return AlreadyTrainedModel_name


def runPreProcess(graph_file_name,gui):
    return PreProcess.runPreProcess(graph_file_name,gui)

#==============================================================================
def OpenTestDataFileWindowTwo():
    ''' function to select, upload and read test data file '''
    data_file_name = ""
    read_file =  ""


    # upload csv and excel (xlsx) dependent data file
    file_opt = options = {}
    options['filetypes'] = [("CSV files", "*.csv")]


    # open dialog box - opne and read selected file
    test_data_file_dialogbox  = tk.filedialog.askopenfile(mode = "rb", **file_opt)


    if test_data_file_dialogbox is None:
        data_file_name = "empty"
        read_file = "empty"

    if test_data_file_dialogbox is not None:

        data_file_name = test_data_file_dialogbox.name
        try:
            # Use index_col = [0] to add column to use as the row labels of
            # the DataFrame and then transpose
            read_file = pd.read_csv(
                    test_data_file_dialogbox, index_col = [0]).transpose()
        except:
            read_file = "empty"

    return data_file_name, read_file
