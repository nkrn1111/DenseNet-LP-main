# ==============================================================================
#
# IMPORTANT NOTE: Don't add a coding line here! It's not necessary.
#
# ==============================================================================
# Dependencies
# ==============================================================================
# ========
# tkinter
# ========
import multiprocessing as mp
import os
from subprocess import Popen, PIPE
import tkinter as tk
from tkinter import ttk, messagebox, IntVar, DoubleVar, StringVar, BooleanVar, filedialog

from PIL import ImageTk, Image
import webbrowser

# ========
# Other
# ========
import pandas as pd
import datetime
from decimal import Decimal
import sys
import numpy as np
import csv
from os import path

# ======
# Fonts
# ======
from Gui import DenseNetController

if sys.platform == "linux" or sys.platform == "win64" or sys.platform == "win32":
    font = "FreeSans"
    # Font 24
    FONT_24 = (font, 20)
    FONT_24_bold = (font, 20, "bold")
    FONT_24_underline = (font, 20, "underline")

    # Font 20
    FONT_20 = (font, 16)
    FONT_20_bold = (font, 16, "bold")
    FONT_20_underline = (font, 16, "underline")

    # Font 16
    FONT_16 = (font, 13)
    FONT_16_bold = (font, 12, "bold")
    FONT_16_underline = (font, 13, "underline")

    # Font 14
    FONT_14 = (font, 12)
    FONT_14_bold = (font, 11, "bold")
    FONT_14_underline = (font, 12, "underline")

    # Font 13
    FONT_13 = (font, 11)
    FONT_13_bold = (font, 10, "bold")
    FONT_13_underline = (font, 11, "underline")

    # Font 12
    FONT_12 = (font, 10)
    FONT_12_bold = (font, 9, "bold")
    FONT_12_underline = (font, 10, "underline")

else:
    font = "Arial"
    # Font 24
    FONT_24 = (font, 24)
    FONT_24_bold = (font, 24, "bold")
    FONT_24_underline = (font, 24, "underline")

    # Font 20
    FONT_20 = (font, 20)
    FONT_20_bold = (font, 20, "bold")
    FONT_20_underline = (font, 20, "underline")

    # Font 16
    FONT_16 = (font, 16)
    FONT_16_bold = (font, 16, "bold")
    FONT_16_underline = (font, 16, "underline")

    # Font 14
    FONT_14 = (font, 14)
    FONT_14_bold = (font, 14, "bold")
    FONT_14_underline = (font, 14, "underline")

    # Font 13
    FONT_13 = (font, 13)
    FONT_13_bold = (font, 13, "bold")
    FONT_13_underline = (font, 13, "underline")

    # Font 12
    FONT_12 = (font, 12)
    FONT_12_bold = (font, 12, "bold")
    FONT_12_underline = (font, 12, "underline")

########################
### Sklearn and nltk ###
########################
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.svm import LinearSVC, NuSVC, SVC

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.semi_supervised import LabelPropagation

from sklearn.neural_network import MLPClassifier

# train_test_split used when user select 1 (Train Sample Size (%))
# cross_val_score and cross_val_predict used when user select 2 (K-fold Cross-Validation)
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.utils import shuffle
from sklearn import metrics
import joblib


# ==============================================================================
# Class ClassificaIO:
#
# ClassificaIO is the main class that is necessory to add new windows to.
# Define ClassificaIO class and pass tk.TK as a parameter.
# ==============================================================================
class TopLevel():

    ###########################
    # define function show_frame with two arguments self and controlar
    def show_frame(self, cont):
        # cont is the key used to look for value in self.frames dictionary
        frame = self.frames[cont]
        # tkraise will rais the start window to the front
        frame.tkraise()
        ###########################

    def __init__(self, top=None):


        top.wm_title("Noam")  # add name to software

        # container_frame is tkinter frame used to add bunch of stuff
        # pack used to pack things in container_frame
        # fill used to fill the space
        # expand used to expand beyond space set
        # configure to minimum size for row and column to 0 with priority
        # weight 1
        container_frame = tk.Frame()
        container_frame.pack(side="top", fill="both", expand=True)
        container_frame.grid_rowconfigure(0, weight=1)
        container_frame.grid_columnconfigure(0, weight=1)

        # creat file object in the menu=
        menu = tk.Menu()
        top.config(menu=menu)
        self.stdout = mp.Queue()
        self.stderr = mp.Queue()
        self.proc = None  # process
        # self.frames used to add all windows (pages) in the software. Loop
        # through windows and add the frames to the dictionary self.frames. Use
        # nsew (north south east west) to stretch everything to window size
        self.frames = {}
        for window in (StartWindow,
                       WindowOne_Use_My_Own_Training_Data,
                       WindowTwo_Already_Trained_My_Model):
            each_frame = window(container_frame, self)

            self.frames[window] = each_frame
            each_frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartWindow)  # always show start window first
        ###########################


# ==============================================================================
############## Functions Used in All of Following Classes Starts ##############


# ==============================================================================
# Function NavigationButtons: allow user to navigate ClassificaIO.
#
# one input:
# command to move to StartWindow (home)
# ==============================================================================
def NavigationButtons(self, home):
    # define function to call user manual
    def user_manual():
        webbrowser.open_new("https://goo.gl/Y9J9tD")

    # quit program
    def clint_exit():
        # output message and ask user before quiting
        quiting = messagebox.askquestion(title="Quit",
                                         message="Are you sure you want to " \
                                                 "exit the program?")
        # if user select True which return yes then destroy meaning quit
        if quiting == "yes":
            sys.exit(0)

    # tkinter frame used to add home, backward and forward buttons
    mainframe = tk.Frame(self, bg="white")

    # home button
    ttk.Button(mainframe, text="HOME", command=home).pack(side="left", padx=10)
    # help button
    ttk.Button(mainframe, text="HELP", command=user_manual).pack(side="left", padx=10)
    # exit button
    ttk.Button(mainframe, text="EXIT PYTHON", command=clint_exit).pack(side="left", padx=10)

    mainframe.pack(padx=42, anchor="w")


# ==============================================================================


# ==============================================================================
# Function ErrorMessage: notify user with error message when error has occurred.
#
# Two inputs:
# arg1_message: main error message
# arg2_detail: ditails
# ==============================================================================
def ErrorMessage(arg1_title, arg2_message):
    # return error message
    return messagebox.showwarning(title=arg1_title,
                                  message=arg2_message,
                                  detail="\nFor instructions on using this " \
                                         "program, please refer to the \"user manual\" " \
                                         "by clicking the \"Help\" button in the " \
                                         "menu bar in ClassificaIO")


# ==============================================================================


# pre-populated list of all allowed upload files.
current_data_upload = ["#Use My Own Training Data Uploaded Files",
                       "Dependent Data: Not Uploaded",
                       "Target Data: Not Uploaded",
                       "Features Data: Not Uploaded",
                       "Test Data: Not Uploaded",

                       "------------------------------------",
                       "#Already Trained My Model Uploaded Files",
                       "Model: Not Uploaded",
                       "Test Data: Not Uploaded",

                       "------------------------------------",
                       "#Upload History"]


# ==============================================================================
# Function UserCurrentDataFileUpload: trace any modification occure to dependent,
# target, Features and testing data file upload, as well as tranied model. It
# also provide user with Log History for any file upload.
#
# Six inputs:
# arg0_StandardizeDataFile_func:   e.g. StandardizeDependentDataFile

# arg1_current_data_upload:        current_data_upload

# data_file_upload_index:          index value used to index current_data_upload
#                                  for corresponding namespace

# arg2_upload_type:                e.g. "Dependent Data: "

# arg3_file_name:                  e.g. dependent_data_file_name

# arg4_file_path:                  e.g. "#Dependent Data Path:"

# arg5_current_data_upload_result: current_data_upload_result
# ==============================================================================
def CurrentDataFileUploadTracker(arg0_StandardizeDataFile_func,
                                 arg1_current_data_upload,
                                 data_file_upload_index,
                                 arg2_upload_type,
                                 arg3_file_name,
                                 arg4_file_path,
                                 arg5_current_data_upload_result):
    arg1_current_data_upload[data_file_upload_index] = (arg2_upload_type.title()
                                                        + (arg3_file_name.split("/")[-1]))
    if arg3_file_name == "Not Uploaded":
        pass
    else:
        arg1_current_data_upload.append(datetime.datetime.now())
        arg1_current_data_upload.append(arg4_file_path)
        arg1_current_data_upload.append(arg3_file_name)
        arg1_current_data_upload.append("")
    arg5_current_data_upload_result.set(arg1_current_data_upload)


############### Functions Used in All of Fallowing Classes Ends ###############
# ==============================================================================


# ==============================================================================
# Class StartWindow:
#
# StartWindow is the first window that the user will see once ClassificaIO run.
# It contains 3 buttons that user use to get to the other windows.
# ==============================================================================
class StartWindow(tk.Frame):

    # Defin init (initialize) function where self is the first argument
    def __init__(self, parent, controller):

        # Parent is the parent class (main class) which is ClassificaIO
        tk.Frame.__init__(self, parent)

        ###########################
        # Add Scrollbar to start window
        def OnFrameConfigureStartWindow(canvas):
            '''Reset the scroll region to encompass the inner frame'''
            canvas.configure(scrollregion=canvas.bbox("all"))

        # create main canvas for start window
        StartWindowCanvas = tk.Canvas(self, highlightthickness=0, bg="white")

        # create main frame and add to convas
        StartWindowCanvasFrame = tk.Frame(StartWindowCanvas, bg="white")

        # add vertical scrollbar to canvas
        vsb = tk.Scrollbar(StartWindowCanvas,
                           orient="vertical",
                           command=StartWindowCanvas.yview,
                           width=13)
        StartWindowCanvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        # add horizontal scrollbar to canvas
        hsb = tk.Scrollbar(StartWindowCanvas,
                           orient="horizontal",
                           command=StartWindowCanvas.xview,
                           width=13)
        StartWindowCanvas.configure(xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")

        StartWindowCanvas.pack(fill="both", anchor="c", expand=True)

        StartWindowCanvas.create_window((1, 1),
                                        window=StartWindowCanvasFrame,
                                        anchor="c")

        # bind WindowOneCanvasFrame to canvas
        StartWindowCanvasFrame.bind("<Configure>",
                                    lambda event,
                                           canvas=StartWindowCanvas: OnFrameConfigureStartWindow(
                                        StartWindowCanvas))
        ###########################

        ###########################

        # Add navigation buttons
        NavigationButtons(StartWindowCanvas,
                          lambda: controller.show_frame(StartWindow))

        tk.Label(StartWindowCanvasFrame, bg="white").pack(padx=2)
        ###########################

        ###########################
        # Add lab logo and ClassificaIO name
        StartWindowMainFrame = tk.Frame(StartWindowCanvasFrame, bg="white")
        tk.Label(StartWindowMainFrame, bg="white").pack(pady=10)
        logo_and_ClassificaIO_name_frame = tk.Frame(StartWindowMainFrame, bg="white")

        # lab logo
        here = path.abspath(path.dirname(__file__))
        logoPath = path.join(here, "LinkPridiction.png")
        upload_logo = Image.open(logoPath)
        resized_logo = upload_logo.resize((150, 150), Image.ANTIALIAS)
        photoimage_logo = ImageTk.PhotoImage(resized_logo)
        labeled_logo = tk.Label(logo_and_ClassificaIO_name_frame,
                                image=photoimage_logo, bg="white")
        labeled_logo.image = photoimage_logo
        labeled_logo.pack(side="left", padx=10, anchor="c")

        # ClassificaIO name
        ClassificaIO_label = tk.Label(logo_and_ClassificaIO_name_frame,
                                      text="ClassificaIO", font=FONT_24_bold, bg="white")
        ClassificaIO_label.pack(side="right", padx=10, anchor="c")

        logo_and_ClassificaIO_name_frame.pack(anchor="c")

        tk.Label(StartWindowMainFrame, bg="white").pack(pady=10)

        # ClassificaIO title
        tk.Label(StartWindowMainFrame,
                 text="Machine Learning for Classification",
                 font=FONT_20_bold, bg="white").pack(anchor="c")

        tk.Label(StartWindowMainFrame, bg="white").pack(pady=20)

        # Button 1 if users using their own training data
        ttk.Button(StartWindowMainFrame,
                   width=35,
                   text="use my own training data".title(),
                   command=lambda: controller.show_frame(
                       WindowOne_Use_My_Own_Training_Data),
                   default='active').pack(anchor="c")

        tk.Label(StartWindowMainFrame, bg="white").pack(pady=1)

        # Button 2 if users already trained and exported the model
        ttk.Button(StartWindowMainFrame,
                   width=35,
                   text="already trained my model".title(),
                   command=lambda: controller.show_frame(
                       WindowTwo_Already_Trained_My_Model),
                   default='active').pack(anchor="c")

        if sys.platform == "linux":
            tk.Label(StartWindowMainFrame, bg="white").pack(pady=183)
            StartWindowMainFrame.pack(anchor="c", padx=537)

        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(StartWindowMainFrame, bg="white").pack(pady=192)
            StartWindowMainFrame.pack(anchor="c", padx=540)

        else:
            tk.Label(StartWindowMainFrame, bg="white").pack(pady=170)
            StartWindowMainFrame.pack(anchor="c", padx=546)

            ###########################


# ==============================================================================
class WindowOne_Use_My_Own_Training_Data(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)

        ###########################
        ###########################
        # Add Scrollbar to window one
        def OnFrameConfigureWindowOne(canvas):
            '''Reset the scroll region to encompass the inner frame'''
            canvas.configure(scrollregion=canvas.bbox("all"))

        # create main canvas for window two
        WindowOneCanvas = tk.Canvas(self, highlightthickness=0, bg="white")

        # create main frame and add to convas
        WindowOneCanvasFrame = tk.Frame(WindowOneCanvas, bg="white")

        # add vertical scrollbar to canvas
        vsb = tk.Scrollbar(WindowOneCanvas,
                           orient="vertical",
                           command=WindowOneCanvas.yview,
                           width=13)
        WindowOneCanvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        # add horizontal scrollbar to canvas
        hsb = tk.Scrollbar(WindowOneCanvas,
                           orient="horizontal",
                           command=WindowOneCanvas.xview,
                           width=13)
        WindowOneCanvas.configure(xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")

        WindowOneCanvas.pack(fill="both", anchor="c", expand=True)

        WindowOneCanvas.create_window((1, 1),
                                      window=WindowOneCanvasFrame,
                                      anchor="c")

        # bind WindowOneCanvasFrame to canvas
        WindowOneCanvasFrame.bind("<Configure>",
                                  lambda event,
                                         canvas=WindowOneCanvas: OnFrameConfigureWindowOne(
                                      WindowOneCanvas))
        ###########################
        ###########################

        ###########################
        ###########################
        # Add navigation buttons
        NavigationButtons(WindowOneCanvas,
                          lambda: controller.show_frame(StartWindow))

        tk.Label(WindowOneCanvasFrame, bg="white").pack(padx=2)
        ###########################
        ###########################

        ###########################
        ###########################
        # current_data_upload_result is to trace any modification occure
        # regarding user upload files.
        global current_data_upload
        global current_data_upload_result
        current_data_upload_result = StringVar()
        current_data_upload_result.set(current_data_upload)
        ###########################
        ###########################

        global training_data_upload_widgets
        training_data_upload_widgets = []  # contain widgets to be deleted

        # Allow user to upload traning data files
        def training_data_upload_function():

            global training_data_upload_selection_result

            training_data_upload_selection_result = training_data_upload_selection.get()

            # destroy widgets
            global training_data_upload_widgets
            for widget in training_data_upload_widgets[:]:
                widget.destroy()
                training_data_upload_widgets.remove(widget)

                ###########################
            # Dependent, Target and Features data files
            if training_data_upload_selection_result == "1":
                training_data_upload_selection_frame = ttk.Frame(TrainingDataUploadFrame, padding=(10, 10, 10, 10),
                                                                 relief="flat", borderwidth=1)

                ttk.Button(training_data_upload_selection_frame, text="Dependent", width=10,
                           command=lambda: CurrentDataFileUploadTracker(StandardizeDependentDataFile(),
                                                                        current_data_upload, 1, "Dependent Data: ",
                                                                        dependent_data_file_name,
                                                                        "#Dependent Data Path:",
                                                                        current_data_upload_result)).pack()

                ttk.Button(training_data_upload_selection_frame, text="Target", width=10,
                           command=lambda: CurrentDataFileUploadTracker(StandardizeTargetDataFile(),
                                                                        current_data_upload, 2, "Target Data: ",
                                                                        target_data_file_name, "#Target Data Path:",
                                                                        current_data_upload_result)).pack()

                ttk.Button(training_data_upload_selection_frame, text="Features", width=10,
                           command=lambda: CurrentDataFileUploadTracker(StandardizeFeaturesDataFile(),
                                                                        current_data_upload, 3, "Features Data: ",
                                                                        features_data_file_name, "#Features Data Path:",
                                                                        current_data_upload_result)).pack()

                training_data_upload_selection_frame.pack(anchor="c")

            # Dependent and Target data files
            if training_data_upload_selection_result == "2":
                training_data_upload_selection_frame = ttk.Frame(TrainingDataUploadFrame, padding=(10, 10, 10, 10),
                                                                 relief="flat", borderwidth=1)

                ttk.Button(training_data_upload_selection_frame, text="Dependent", width=10,
                           command=lambda: CurrentDataFileUploadTracker(StandardizeDependentDataFile(),
                                                                        current_data_upload, 1, "Dependent Data: ",
                                                                        dependent_data_file_name,
                                                                        "#Dependent Data Path:",
                                                                        current_data_upload_result)).pack()

                ttk.Button(training_data_upload_selection_frame, text="Target", width=10,
                           command=lambda: CurrentDataFileUploadTracker(StandardizeTargetDataFile(),
                                                                        current_data_upload, 2, "Target Data: ",
                                                                        target_data_file_name, "#Target Data Path:",
                                                                        current_data_upload_result)).pack()

                ttk.Label(training_data_upload_selection_frame).pack(pady=2)

                training_data_upload_selection_frame.pack(anchor="c")
                ###########################

            # destroy later
            training_data_upload_widgets = (training_data_upload_widgets[:] + [training_data_upload_selection_frame])
            # pack them afterwards
            for widget in training_data_upload_widgets:
                widget.pack()

                ###########################
                ###########################
                ###########################
                ###########################

        # list of all 12 classifiers
        ListOfAllClassifiers = ("I. Linear_model",
                                '1: LogisticRegression',
                                '2: PassiveAggressiveClassifier',
                                '3: Perceptron',
                                '4: RidgeClassifier',
                                '5: Stochastic Gradient Descent (SGDClassifier)',

                                "II. Discriminant_analysis",
                                '6: LinearDiscriminantAnalysis',
                                '7: QuadraticDiscriminantAnalysis',

                                'III. Support vector machines (SVMs)',
                                '8: LinearSVC',
                                '9: NuSVC',
                                '10: SVC',

                                'IV. Neighbors',
                                '11: KNeighborsClassifier',
                                '12: NearestCentroid',
                                '13: RadiusNeighborsClassifier',

                                'V. Gaussian_process',
                                '14: GaussianProcessClassifier',

                                'VI. Naive_bayes',
                                '15: BernoulliNB',
                                '16: GaussianNB',
                                '17: MultinomialNB',

                                'VII. Trees',
                                '18: DecisionTreeClassifier',
                                '19: ExtraTreeClassifier',

                                'VIII. Ensemble',
                                '20: AdaBoostClassifier',
                                '21: BaggingClassifier',
                                '22: ExtraTreesClassifier',
                                '23: RandomForestClassifier',

                                'IX. Semi_supervised',
                                '24: LabelPropagation',

                                'X. Neural_network',
                                '25: MLPClassifier')

        AllClassifiers = StringVar(value=ListOfAllClassifiers)

        # SelectedClassifier used to set classifier once is selected
        SelectedClassifier = StringVar()

        # this list will contain widgets to be deleted

        global widgets
        widgets = []

        # This function will contain 14 other functions:
        # each function will be to call a separate classifier

        def ClassifiersSelectionFunc(*args):

            press_selection = AllClassifiersListbox.curselection()
            if len(press_selection) == 1:
                press_selection_idx = int(press_selection[0])
                AllClassifiersListbox.see(press_selection_idx)
                classifier_name = ListOfAllClassifiers[press_selection_idx]

                # Set selected classifier to SelectedClassifier
                SelectedClassifier.set(classifier_name)
                #########################################################
                #########################################################
                #########################################################

                #########################################################
                #########################################################
                #########################################################
                # Call function EmptyFrame if user select
                # #Generalized Linear Models from classifiers list.
                if classifier_name == "I. Linear_model":
                    EmptyFrame(WindowOneCanvasFrame, classifier_name)

                # Call function LogisticRegression_MainFunc if user select
                # LogisticRegression from classifiers list.
                if classifier_name == '1: LogisticRegression':

                    def LogisticRegression_MainFunc():

                        global widgets
                        for widget in widgets[:]:
                            widget.destroy()
                            widgets.remove(widget)

                        predicted_value_result = IntVar()
                        predicted_value_result.set("")

                        model_evaluation_and_statistics_result = IntVar()
                        model_evaluation_and_statistics_result.set("")

                        testing_result = IntVar()
                        testing_result.set("")

                        ###########################################

                        def LogisticRegression_SubFunc():
                            global CommonFeatures_for_DataTraining
                            global CallClassifier
                            global confusion_matrix_accuracy_error_result
                            global Classifier_Pramaters
                            global predicted_value
                            global testing_value

                            LR_TrainTestSplit_vs_KFold_Result = LR_TrainTestSplit_vs_KFold.get()
                            LR_TrainTestSplit_Scale_Result = LR_TrainTestSplit_Scale.get() / 100
                            LR_KFold_Result = LR_KFold.get()

                            # get random_state value
                            if LR_random_state_int_or_none.get() == "None":
                                LR_random_state_Result = None
                            else:
                                LR_random_state_Result = LR_random_state.get()

                            LR_penalty_Result = LR_penalty.get()  # get penalty value
                            LR_multi_class_result = LR_multi_class.get()  # get multi_class value
                            LR_solver_result = LR_solver.get()  # get solver value
                            LR_max_iter_result = LR_max_iter.get()  # get max_iter value
                            LR_tol_result = LR_tol.get()  # get tol value
                            LR_intercept_scaling_result = LR_intercept_scaling.get()  # get intercept_scaling value
                            LR_verbose_result = LR_verbose.get()  # get verbose value
                            LR_n_jobs_result = LR_n_jobs.get()  # get n_jobs value
                            LR_C_result = LR_C.get()  # get C value
                            LR_fit_intercept_result = LR_fit_intercept.get()  # get fit_intercept value
                            LR_dual_result = LR_dual.get()  # get dual value
                            LR_warm_start_result = LR_warm_start.get()  # get warm_start value

                            CallClassifier = LogisticRegression(
                                penalty=LR_penalty_Result,
                                dual=LR_dual_result,
                                tol=LR_tol_result,
                                C=LR_C_result,
                                fit_intercept=LR_fit_intercept_result,
                                intercept_scaling=LR_intercept_scaling_result,
                                class_weight=None,
                                random_state=LR_random_state_Result,
                                solver=LR_solver_result,
                                max_iter=LR_max_iter_result,
                                multi_class=LR_multi_class_result,
                                verbose=LR_verbose_result,
                                warm_start=LR_warm_start_result,
                                n_jobs=LR_n_jobs_result)

                            #######################################
                            ############ Training Data ############
                            #######################################



                            ########################################
                            ############# Testing Data #############
                            ########################################

                            try:

                                if training_data_upload_selection_result == "1" or training_data_upload_selection_result == "2":
                                    all_gene_probs = list(map(str, OpenTestDataFile_output_ReadFile_new.columns.values))

                                    # if user select 1, user have to upload all
                                    # "Dependent, Target and Features" data files
                                    if training_data_upload_selection_result == "1":
                                        CommonFeatures_for_DataTesting = list(
                                            set(all_gene_probs).intersection(OpenFeaturesDataFile_output_ReadFile_new))

                                    # if user select 2, user have to upload both
                                    # "Dependent, Target" data files
                                    if training_data_upload_selection_result == "2":
                                        CommonFeatures_for_DataTesting = list(
                                            set(all_gene_probs).intersection(OpenTestDataFile_output_ReadFile_oroginal))

                                    testing_X = OpenTestDataFile_output_ReadFile_new[CommonFeatures_for_DataTesting]
                                    testing_y = CallClassifier.predict(testing_X)

                                    testing_value = []
                                    for i, j in zip(list(np.array(testing_X.index)), testing_y):
                                        testing_value_result = [OpenTestDataFile_output_ReadFile_new["id"][i], "一", j]
                                        testing_value.append(testing_value_result)

                                    total_testing_num = ("Total objects tested: " + str(len(testing_value)))
                                    testing_value.insert(0, total_testing_num)
                                    testing_result.set(testing_value)

                            except NameError:
                                pass

                            ########################################
                            ############# Export Model #############
                            ########################################

                        def ExportModel():
                            DataFile = tk.filedialog.asksaveasfilename()

                            if DataFile is None:
                                return

                            if DataFile:
                                ExportModelResult = [CommonFeatures_for_DataTraining,
                                                     CallClassifier,
                                                     confusion_matrix_accuracy_error_result,
                                                     Classifier_Pramaters]

                                joblib.dump(ExportModelResult, DataFile + '.pkl')

                        ##################################################
                        ############# Export Training Result #############
                        ##################################################

                        def ExportTrainingResult():
                            DataFile = tk.filedialog.asksaveasfile(mode="w", defaultextension=".csv")

                            if DataFile is None:
                                return

                            if DataFile:
                                writer = csv.writer(DataFile)
                                writer.writerows(
                                    [["##Analysis Done Using ClassificaIO on " + str(datetime.date.today())]])
                                writer.writerows([[""]])

                                if training_data_upload_selection_result == "1":
                                    writer.writerows([["##dependent data file:".title()]])
                                    writer.writerows([["##" + dependent_data_file_name]])
                                    writer.writerows([["##target data file:".title()]])
                                    writer.writerows([["##" + target_data_file_name]])
                                    writer.writerows([["##features data file:".title()]])
                                    writer.writerows([["##" + features_data_file_name]])

                                if training_data_upload_selection_result == "2":
                                    writer.writerows([["##dependent data file:".title()]])
                                    writer.writerows([["##" + dependent_data_file_name]])
                                    writer.writerows([["##target data file:".title()]])
                                    writer.writerows([["##" + target_data_file_name]])

                                writer.writerows([[""]])

                                for pramater in Classifier_Pramaters:
                                    writer.writerows([["##" + pramater]])

                                writer.writerows([[""]])

                                writer.writerows([["##confusion matrix, model accuracy & error:".title()]])
                                for item in confusion_matrix_accuracy_error_result:
                                    writer.writerow(["##" + item])

                                writer.writerows([[""]])

                                writer.writerow(["##" + predicted_value[0]])
                                for item2 in predicted_value[1:]:
                                    writer.writerow([item2[0], item2[2], item2[4]])

                        ##################################################
                        ############# Export Training Result #############
                        ##################################################

                        def ExportTestingResult():

                            DataFile = tk.filedialog.asksaveasfile(mode="w", defaultextension=".csv")

                            if DataFile is None:
                                return

                            if DataFile:
                                writer_testing = csv.writer(DataFile)
                                writer_testing.writerows(
                                    [["##Analysis Done Using ClassificaIO on " + str(datetime.date.today())]])
                                writer_testing.writerows([[""]])

                                if training_data_upload_selection_result == "1":
                                    writer_testing.writerows([["##Testing data file:".title()]])
                                    writer_testing.writerows([["##" + test_data_file_name]])
                                    writer_testing.writerows([["##dependent data file:".title()]])
                                    writer_testing.writerows([["##" + dependent_data_file_name]])
                                    writer_testing.writerows([["##target data file:".title()]])
                                    writer_testing.writerows([["##" + target_data_file_name]])
                                    writer_testing.writerows([["##features data file:".title()]])
                                    writer_testing.writerows([["##" + features_data_file_name]])

                                if training_data_upload_selection_result == "2":
                                    writer_testing.writerows([["##Testing data file:".title()]])
                                    writer_testing.writerows([["##" + test_data_file_name]])
                                    writer_testing.writerows([["##dependent data file:".title()]])
                                    writer_testing.writerows([["##" + dependent_data_file_name]])
                                    writer_testing.writerows([["##target data file:".title()]])
                                    writer_testing.writerows([["##" + target_data_file_name]])

                                writer_testing.writerows([[""]])

                                for pramater in Classifier_Pramaters:
                                    writer_testing.writerows([["##" + pramater]])

                                writer_testing.writerows([[""]])

                                writer_testing.writerows([["##confusion matrix, model accuracy & error:".title()]])
                                for item in confusion_matrix_accuracy_error_result:
                                    writer_testing.writerow(["##" + item])

                                writer_testing.writerows([[""]])

                                writer_testing.writerow(["##" + testing_value[0]])
                                for item4 in testing_value[1:]:
                                    writer_testing.writerow([item4[0], item4[2]])


# ==============================================================================
##### Functions Used in Class: WindowOne_Use_My_Own_Training_Data Starts ######


# ==============================================================================
# Function EmptyFrame: will destroy any widget in frame once a classifier title
# selected.

# inmput =  WindowOneCanvasFrame
# ==============================================================================
def EmptyFrame(MainConves, ClassifierName):
    global widgets
    for widget in widgets[:]:
        widget.destroy()
        widgets.remove(widget)

    if ClassifierName == "I. Linear_model":

        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='1: LogisticRegression', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='2: PassiveAggressiveClassifier', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='3: Perceptron', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='4: RidgeClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='5: Stochastic Gradient Descent (SGDClassifier)', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=134)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=137)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=129)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == "II. Discriminant_analysis":
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='6: LinearDiscriminantAnalysis', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='7: QuadraticDiscriminantAnalysis', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=170)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=174)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=168)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == "III. Support vector machines (SVMs)":
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='8: LinearSVC', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='9: NuSVC', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='10: SVC', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=158)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=162)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=155)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == "IV. Neighbors":
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='11: KNeighborsClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='12: NearestCentroid', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='13: RadiusNeighborsClassifier', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=158)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=162)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=155)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'V. Gaussian_process':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='14: GaussianProcessClassifier', fg='SteelBlue3', font=FONT_16,
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=182)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=187)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=181)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'VI. Naive_bayes':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='15: BernoulliNB', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='16: GaussianNB', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='17: MultinomialNB', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=158)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=162)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=155)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'VII. Trees':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='18: DecisionTreeClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='19: ExtraTreeClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=170)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=174)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=168)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'VIII. Ensemble':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='20: AdaBoostClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='21: BaggingClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='22: ExtraTreesClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='23: RandomForestClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=146)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=149)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=142)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'IX. Semi_supervised':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='24: LabelPropagation', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=182)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=187)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=181)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'X. Neural_network':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='25: MLPClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=182)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=187)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=181)
        EmptyFrame_Frame.pack(anchor="c")

    # destroy these later
    widgets = widgets[:] + [EmptyFrame_Frame]

    for widget in widgets:
        widget.pack()  # pack them afterwards


# ==============================================================================
# Function OpenDependentDataFile: used to upload dependent data file, as well as
# read and transpose dependent data file to use by Sklearn.

# ==============================================================================
def OpenDependentDataFile():
    ''' function to select, upload and read dependent data file '''
    data_file_name = ""
    read_file = ""

    # upload csv and excel (xlsx) dependent data file
    file_opt = options = {}
    options['filetypes'] = [("CSV files", "*.csv")]

    # open dialog box - opne and read selected file
    dependent_data_file_dialogbox = tk.filedialog.askopenfile(mode="rb", **file_opt)

    if dependent_data_file_dialogbox is None:
        data_file_name = "empty"
        read_file = "empty"

    if dependent_data_file_dialogbox is not None:

        data_file_name = dependent_data_file_dialogbox.name
        try:
            # Use index_col = [0] to add column to use as the row labels of
            # the DataFrame and then transpose
            read_file = pd.read_csv(
                dependent_data_file_dialogbox, index_col=[0]).transpose()
        except:
            read_file = "empty"

    return data_file_name, read_file


# ==============================================================================
# Function StandardizeDependentDataFile: will check if dependent data file was
# uploaded or not and execute accordingly. If file is uploaded correctely then
# it will add "id" header to column index 0.

# "id" header will be used to merge both dependent and target datasets.
# ==============================================================================
def StandardizeDependentDataFile():
    ''' function to standardize dependent data file '''
    # globalize dependent data file name to show in user "current data upload"
    # window which show user Log History for any file upload.
    global dependent_data_file_name
    global OpenDependentDataFile_output_ReadFile_new
    global OpenDependentDataFile_output_ReadFile_oroginal

    # call function where files been uploaded
    OpenDependentDataFile_output = OpenDependentDataFile()

    # index 0 = file name
    OpenDependentDataFile_output_FileName = OpenDependentDataFile_output[0]
    # index 1 = read file
    OpenDependentDataFile_output_ReadFile = OpenDependentDataFile_output[1]

    # if both file name and read file are empty (no file uploaded)
    if (OpenDependentDataFile_output_ReadFile == "empty"
            and OpenDependentDataFile_output_FileName == "empty"):
        # reset dependent file name to  not uploaded
        dependent_data_file_name = "Not Uploaded"

    # if both file name and read file are not empty (file has been uploaded)
    if (OpenDependentDataFile_output_ReadFile != "empty"
            and OpenDependentDataFile_output_FileName != "empty"):
        # keep dependent file name the same
        dependent_data_file_name = OpenDependentDataFile_output_FileName

        OpenDependentDataFile_output_ReadFile_new = OpenDependentDataFile_output_ReadFile
        OpenDependentDataFile_output_ReadFile_new.reset_index(inplace=True)
        accessions_headr = list(OpenDependentDataFile_output_ReadFile_new.columns)[0]
        # change first column headr (CEL files IDs) to id
        OpenDependentDataFile_output_ReadFile_new = (
            OpenDependentDataFile_output_ReadFile_new.rename(columns=
                                                             {accessions_headr: 'id'}))
        OpenDependentDataFile_output_ReadFile_new["id"] = list(
            map(str, OpenDependentDataFile_output_ReadFile_new["id"]))
        OpenDependentDataFile_output_ReadFile_oroginal = list(
            map(str, OpenDependentDataFile_output_ReadFile.columns.values))

    # if file name not empty and read file is empty (no file uploaded)
    if (OpenDependentDataFile_output_ReadFile == "empty" and
            OpenDependentDataFile_output_FileName != "empty"):
        # reset dependent file name to  not uploaded
        dependent_data_file_name = "Not Uploaded"

        file_name_error = OpenDependentDataFile_output_FileName.split("/")[-1]
        ErrorMessage("Dependent Data File In Use",
                     "An error has occurred during parsing \"" + file_name_error \
                     + "\", the error may have been caused by dataset format!")


# ==============================================================================
# Function OpenTargetDataFile: used to upload and read dependent data file to
# use by Sklearn.

# ==============================================================================
def OpenTargetDataFile():
    ''' function to select, upload and read target data file '''
    data_file_name = ""
    read_file = ""

    # upload csv and excel (xlsx) dependent data file
    file_opt = options = {}
    options['filetypes'] = [("CSV files", "*.csv")]

    # open dialog box - opne and read selected file
    target_data_file_dialogbox = tk.filedialog.askopenfile(mode="rb", **file_opt)

    if target_data_file_dialogbox is None:
        data_file_name = "empty"
        read_file = "empty"

    if target_data_file_dialogbox is not None:

        data_file_name = target_data_file_dialogbox.name
        try:
            # open and read csv file
            read_file = pd.read_csv(
                target_data_file_dialogbox)
        except:
            read_file = "empty"

    return data_file_name, read_file


# ==============================================================================
# Function StandardizeTargetDataFile: will add "id" header to column index 0
# and "target" header to column index 1.

# "id" header will be used to merge both dependent and target datasets.
# ==============================================================================
def StandardizeTargetDataFile():
    ''' function to standardize target data file '''
    # globalize target data file name to show in user "current data upload"
    # window which show user Log History for any file upload.
    global target_data_file_name
    global OpenTargetDataFile_output_ReadFile_new

    # call function where files been uploaded
    OpenTargetDataFile_output = OpenTargetDataFile()

    # index 0 = file name
    OpenTargetDataFile_output_FileName = OpenTargetDataFile_output[0]
    # index 1 = read file
    OpenTargetDataFile_output_ReadFile = OpenTargetDataFile_output[1]

    # if both file name and read file are empty (no file uploaded)
    if (OpenTargetDataFile_output_ReadFile == "empty"
            and OpenTargetDataFile_output_FileName == "empty"):
        # reset target file name to  not uploaded
        target_data_file_name = "Not Uploaded"

    # if both file name and read file are not empty (file has been uploaded)
    if (OpenTargetDataFile_output_ReadFile != "empty"
            and OpenTargetDataFile_output_FileName != "empty"):
        # keep target file name the same
        target_data_file_name = OpenTargetDataFile_output_FileName

        # change first column headr (CEL files IDs) to ID
        accessions_headr_id = list(OpenTargetDataFile_output_ReadFile.columns)[0]
        accessions_headr_target = list(OpenTargetDataFile_output_ReadFile.columns)[1]

        # add "id" header to column index 0.
        OpenTargetDataFile_output_ReadFile = (
            OpenTargetDataFile_output_ReadFile.rename(columns=
                                                      {accessions_headr_id: 'id'}))

        # add "target" header to column index 1.
        OpenTargetDataFile_output_ReadFile_new = (
            OpenTargetDataFile_output_ReadFile.rename(columns=
                                                      {accessions_headr_target: 'target'}))
        OpenTargetDataFile_output_ReadFile_new["id"] = list(map(str, OpenTargetDataFile_output_ReadFile_new["id"]))

    # if file name not empty and read file is empty (no file uploaded)
    if (OpenTargetDataFile_output_ReadFile == "empty" and
            OpenTargetDataFile_output_FileName != "empty"):
        # reset target file name to  not uploaded
        target_data_file_name = "Not Uploaded"

        file_name_error = OpenTargetDataFile_output_FileName.split("/")[-1]
        ErrorMessage("Target Data File In Use",
                     "An error has occurred during parsing \"" + file_name_error \
                     + "\", the error may have been caused by dataset format!")


# ==============================================================================
# Function OpenFeaturesDataFile: used to upload features data file, as well as
# read and transpose features data file to use by Sklearn.

# ==============================================================================
def OpenFeaturesDataFile():
    ''' function to select, upload and read features data file '''
    data_file_name = ""
    read_file = ""

    # upload csv and excel (xlsx) dependent data file
    file_opt = options = {}
    options['filetypes'] = [("CSV files", "*.csv")]

    # open dialog box - opne and read selected file
    features_data_file_dialogbox = tk.filedialog.askopenfile(mode="rb", **file_opt)

    if features_data_file_dialogbox == None:
        data_file_name = "empty"
        read_file = "empty"

    if features_data_file_dialogbox != None:

        data_file_name = features_data_file_dialogbox.name
        try:
            # Use index_col = [0] to add column to use as the row labels of
            # the DataFrame and then transpose
            read_file = pd.read_csv(
                features_data_file_dialogbox, index_col=[0]).transpose()
        except:
            read_file = "empty"

    return data_file_name, read_file


# ==============================================================================
# Function StandardizeFeaturesDataFile: will extract columns headers incase user
# which to extract and intersect a subset of headers

# ==============================================================================
def StandardizeFeaturesDataFile():
    ''' function to standardize features data file '''
    # globalize target data file name to show in user "current data upload"
    # window which show user Log History for any file upload.
    global features_data_file_name
    global OpenFeaturesDataFile_output_ReadFile_new

    # call function where files been uploaded
    OpenFeaturesDataFile_output = OpenFeaturesDataFile()

    # index 0 = file name
    OpenFeaturesDataFile_output_FileName = OpenFeaturesDataFile_output[0]
    # index 1 = read file
    OpenFeaturesDataFile_output_ReadFile = OpenFeaturesDataFile_output[1]

    # if both file name and read file are empty (no file uploaded)
    if (OpenFeaturesDataFile_output_ReadFile == "empty"
            and OpenFeaturesDataFile_output_FileName == "empty"):
        # reset target file name to  not uploaded
        features_data_file_name = "Not Uploaded"

    # if both file name and read file are not empty (file has been uploaded)
    if (OpenFeaturesDataFile_output_ReadFile != "empty"
            and OpenFeaturesDataFile_output_FileName != "empty"):
        # keep target file name the same
        features_data_file_name = OpenFeaturesDataFile_output_FileName
        OpenFeaturesDataFile_output_ReadFile_new = list(map(str,
                                                            OpenFeaturesDataFile_output_ReadFile.columns.values))

    # if file name not empty and read file is empty (no file uploaded)
    if (OpenFeaturesDataFile_output_ReadFile == "empty" and
            OpenFeaturesDataFile_output_FileName != "empty"):
        # reset target file name to  not uploaded
        features_data_file_name = "Not Uploaded"

        file_name_error = OpenFeaturesDataFile_output_FileName.split("/")[-1]
        ErrorMessage("Features Data File In Use",
                     "An error has occurred during parsing \"" + file_name_error \
                     + "\", the error may have been caused by dataset format!")


# ==============================================================================
# Function OpenTestDataFile: used to upload test data file, as well as
# read and transpose test data file to use by Sklearn.

# ==============================================================================
def OpenTestDataFile():
    ''' function to select, upload and read test data file '''
    data_file_name = ""
    read_file = ""

    # upload csv and excel (xlsx) dependent data file
    file_opt = options = {}
    options['filetypes'] = [("CSV files", "*.csv")]

    # open dialog box - opne and read selected file
    test_data_file_dialogbox = tk.filedialog.askopenfile(mode="rb", **file_opt)

    if test_data_file_dialogbox == None:
        data_file_name = "empty"
        read_file = "empty"

    if test_data_file_dialogbox != None:

        data_file_name = test_data_file_dialogbox.name
        try:
            # Use index_col = [0] to add column to use as the row labels of
            # the DataFrame and then transpose
            read_file = pd.read_csv(
                test_data_file_dialogbox, index_col=[0]).transpose()
        except:
            read_file = "empty"

    return data_file_name, read_file


# ==============================================================================
# Function StandardizeTestDataFile: will add "id" header to column index 0

# "id" header will be used to merge test, dependent and target datasets.
# ==============================================================================
def StandardizeTestDataFile():
    ''' function to standardize test data file '''
    # globalize test data file name to show in user "current data upload"
    # window which show user Log History for any file upload.
    global test_data_file_name
    global OpenTestDataFile_output_ReadFile_new
    global OpenTestDataFile_output_ReadFile_oroginal

    # call function where files been uploaded
    OpenTestDataFile_output = OpenTestDataFile()

    # index 0 = file name
    OpenTestDataFile_output_FileName = OpenTestDataFile_output[0]
    # index 1 = read file
    OpenTestDataFile_output_ReadFile = OpenTestDataFile_output[1]

    # if both file name and read file are empty (no file uploaded)
    if (OpenTestDataFile_output_ReadFile == "empty"
            and OpenTestDataFile_output_FileName == "empty"):
        # reset test file name to "not uploaded"
        test_data_file_name = "Not Uploaded"

    # if both file name and read file are not empty (file has been uploaded)
    if (OpenTestDataFile_output_ReadFile != "empty"
            and OpenTestDataFile_output_FileName != "empty"):
        # keep test file name the same
        test_data_file_name = OpenTestDataFile_output_FileName

        OpenTestDataFile_output_ReadFile_new = OpenTestDataFile_output_ReadFile
        OpenTestDataFile_output_ReadFile_new.reset_index(inplace=True)
        accessions_headr = list(OpenTestDataFile_output_ReadFile_new.columns)[0]
        # change first column headr (CEL files IDs) to id
        OpenTestDataFile_output_ReadFile_new = (
            OpenTestDataFile_output_ReadFile_new.rename(columns=
                                                        {accessions_headr: 'id'}))
        OpenTestDataFile_output_ReadFile_oroginal = OpenTestDataFile_output_ReadFile

    # if file name not empty and read file is empty (no file uploaded)
    if (OpenTestDataFile_output_ReadFile == "empty" and
            OpenTestDataFile_output_FileName != "empty"):
        # reset test file name to "not uploaded"
        test_data_file_name = "Not Uploaded"

        file_name_error = OpenTestDataFile_output_FileName.split("/")[-1]
        ErrorMessage("Test Data File In Use",
                     "An error has occurred during parsing \"" + file_name_error \
                     + "\", the error may have been caused by dataset format!")


def EmptyFrame(MainConves, ClassifierName):
    global widgets
    for widget in widgets[:]:
        widget.destroy()
        widgets.remove(widget)

    if ClassifierName == "I. Linear_model":

        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='1: LogisticRegression', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='2: PassiveAggressiveClassifier', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='3: Perceptron', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='4: RidgeClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='5: Stochastic Gradient Descent (SGDClassifier)', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=134)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=137)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=129)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == "II. Discriminant_analysis":
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='6: LinearDiscriminantAnalysis', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='7: QuadraticDiscriminantAnalysis', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=170)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=174)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=168)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == "III. Support vector machines (SVMs)":
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='8: LinearSVC', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='9: NuSVC', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='10: SVC', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=158)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=162)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=155)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == "IV. Neighbors":
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='11: KNeighborsClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='12: NearestCentroid', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='13: RadiusNeighborsClassifier', font=FONT_16, fg='SteelBlue3',
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=158)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=162)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=155)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'V. Gaussian_process':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='14: GaussianProcessClassifier', fg='SteelBlue3', font=FONT_16,
                 bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=182)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=187)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=181)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'VI. Naive_bayes':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='15: BernoulliNB', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='16: GaussianNB', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        tk.Label(EmptyFrame_Frame, text='17: MultinomialNB', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=158)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=162)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=155)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'VII. Trees':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='18: DecisionTreeClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='19: ExtraTreeClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=170)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=174)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=168)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'VIII. Ensemble':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='20: AdaBoostClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='21: BaggingClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='22: ExtraTreesClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        tk.Label(EmptyFrame_Frame, text='23: RandomForestClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=146)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=149)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=142)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'IX. Semi_supervised':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='24: LabelPropagation', font=FONT_16, fg='SteelBlue3', bg="white").pack(
            anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=182)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=187)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=181)
        EmptyFrame_Frame.pack(anchor="c")

    if ClassifierName == 'X. Neural_network':
        EmptyFrame_Frame = tk.Frame(MainConves, bg="white")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=45)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=50)
        tk.Label(EmptyFrame_Frame, text=ClassifierName, font=FONT_16_underline, fg='SteelBlue3', bg="white").pack()
        tk.Label(EmptyFrame_Frame, text='25: MLPClassifier', font=FONT_16, fg='SteelBlue3', bg="white").pack(anchor="w")
        if sys.platform == "linux":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=182)
        elif sys.platform == "win64" or sys.platform == "win32":
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=187)
        else:
            tk.Label(EmptyFrame_Frame, bg="white").pack(anchor="w", pady=181)
        EmptyFrame_Frame.pack(anchor="c")

    # destroy these later
    widgets = widgets[:] + [EmptyFrame_Frame]

    for widget in widgets:
        widget.pack()  # pack them afterwards


# ==============================================================================
# Class WindowTwo_Already_Trained_My_Model:
#
# WindowTwo_Already_Trained_My_Model
# ==============================================================================
class WindowTwo_Already_Trained_My_Model(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)

        ###########################
        # Add Scrollbar to window Two
        def OnFrameConfigureWindowTwo(canvas):
            '''Reset the scroll region to encompass the inner frame'''
            canvas.configure(scrollregion=canvas.bbox("all"))

        # create main canvas for window two
        WindowTwoCanvas = tk.Canvas(self, highlightthickness=0, bg="white")

        # create main frame and add to convas
        WindowTwoCanvasFrame = tk.Frame(WindowTwoCanvas, bg="white")

        # add vertical scrollbar to canvas
        vsb = tk.Scrollbar(WindowTwoCanvas,
                           orient="vertical",
                           command=WindowTwoCanvas.yview,
                           width=13)
        WindowTwoCanvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        # add horizontal scrollbar to canvas
        hsb = tk.Scrollbar(WindowTwoCanvas,
                           orient="horizontal",
                           command=WindowTwoCanvas.xview,
                           width=13)
        WindowTwoCanvas.configure(xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")

        WindowTwoCanvas.pack(fill="both", anchor="c", expand=True)

        WindowTwoCanvas.create_window((1, 1),
                                      window=WindowTwoCanvasFrame,
                                      anchor="c")

        # bind WindowTwoCanvasFrame to canvas
        WindowTwoCanvasFrame.bind("<Configure>",
                                  lambda event,
                                         canvas=WindowTwoCanvas: OnFrameConfigureWindowTwo(WindowTwoCanvas))
        ###########################

        ###########################
        # Add navigation buttons
        NavigationButtons(WindowTwoCanvas,
                          lambda: controller.show_frame(StartWindow))

        tk.Label(WindowTwoCanvasFrame, bg="white").pack(padx=2)
        ###########################

        #########################################################
        #########################################################
        #########################################################

        # current_data_upload_result is to trace any modification occure
        # regarding user upload files.
        current_data_upload_result.set(current_data_upload)

        # ClassifierParameters_WindowTwo is to set trained any modification occur
        # regarding user upload files.
        ClassifierParameters_WindowTwo = StringVar()
        ClassifierParameters_WindowTwo.set("")

        ConfusionMatrixModelAccuracyError_WindowTwo = StringVar()
        ConfusionMatrixModelAccuracyError_WindowTwo.set("")


        testing_result_WindowTwo = IntVar()
        testing_result_WindowTwo.set("")

        #########################################################
        #########################################################
        #########################################################
        def ExecuteAlreadyTrainedModel():

           # try:
                global testing_value_WindowTwo
                testing_value_WindowTwo = []
                ########################################
                ############ Trained Model #############
                ########################################






                # output confusion matrix, model accuracy & error located at index 2
                #run PreProcess stage
                temp=list
                temp1=[]
                temp=DenseNetController.runPreProcess(os.path.basename(AlreadyTrainedModel_name),gui)
                print ("---------------------------------------------")
                for i in temp:
                    print (i)
                #ConfusionMatrixModelAccuracyError_WindowTwo.set(temp)

                with Popen('dir', shell=True, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
                    for line in p.stdout:
                        temp1.append(line)

                ConfusionMatrixModelAccuracyError_WindowTwo.set(temp1)
                ########################################
                ############# Testing Data #############
                ########################################






            #
            #
            # except NameError:
            #     ErrorMessage("Data Upload Error",
            #                  "We Can't Sync Your Data: Please " +
            #                  "upload all required files " +
            #                  "\"Trained Model and Testing Data\"!")
            #
            # except IndexError:
            #     ErrorMessage("Data Upload Error",
            #                  "We Can't Sync Your Data: Please " +
            #                  "upload all required files: " +
            #                  "\"Trained Model and Testing Data\"!")
            #
            # except ValueError:
            #     ErrorMessage("Dataframe Shape Error",
            #                  "An error has occurred during parsing " +
            #                  "\"Trained Model and Testing Data\" data " +
            #                  "files, the error may have been caused by " +
            #                  "dataframe shape or a wrong file been uploaded!")

        ##################################################
        ############## Export Testing Result #############
        ##################################################

        def ExportTestingResultWindowTwo():
            print("")

        #########################################################
        #########################################################
        #########################################################

        WindowTwoUserInputFrame = ttk.Frame(WindowTwoCanvasFrame, relief="raised", borderwidth=1)

        ###############################################################

        UploadTrainingModelFrame = ttk.Frame(WindowTwoUserInputFrame, padding=(29, 20, 29, 5))

        ttk.Label(UploadTrainingModelFrame, text="UPLOAD TRAINING MODEL FILE".upper(), font=FONT_14_bold).pack(
            anchor="c")
        ttk.Button(UploadTrainingModelFrame, text="Model File",
                   command=lambda: CurrentDataFileUploadTracker(UploadAlreadyTrainedModelWindowTwo(),
                                                                current_data_upload, 7, "Model: ",
                                                                AlreadyTrainedModel_name, "#Model Path:",
                                                                current_data_upload_result), width=12).pack(anchor="c",
                                                                                                            pady=10)
        UploadTrainingModelFrame.pack_configure(side="left", anchor="n")
        ###############################################################

        # First divider between frames
        FristDividerFrame = ttk.Frame(WindowTwoUserInputFrame, padding=(0, 161, 0, 0), relief="raised", borderwidth=1)
        ttk.Label(FristDividerFrame).pack()
        FristDividerFrame.pack(side="left", anchor="c")

        ###############################################################
        if sys.platform == "linux" or sys.platform == "win64" or sys.platform == "win32":
            testing_data_frame = ttk.Frame(WindowTwoUserInputFrame, padding=(29, 20, 29, 5))
        else:
            testing_data_frame = ttk.Frame(WindowTwoUserInputFrame, padding=(33, 20, 33, 5))
        ttk.Label(testing_data_frame, text="UPLOAD TESTING DATA FILE", font=FONT_14_bold).pack(anchor="c")
        ttk.Button(testing_data_frame, text="Testing Data",
                   command=lambda: CurrentDataFileUploadTracker(StandardizeTestDataFileWindowTwo(), current_data_upload,
                                                                8, "Test Data: ", test_data_file_nameWindowTwo,
                                                                "#Test Data Path:", current_data_upload_result),
                   width=12).pack(anchor="c", pady=10)

        testing_data_frame.pack(side="left", anchor="n")
        ###############################################################

        # Second divider between frames
        SecondDividerFrame = ttk.Frame(WindowTwoUserInputFrame, padding=(0, 161, 0, 0), relief="raised", borderwidth=1)
        ttk.Label(SecondDividerFrame).pack()
        SecondDividerFrame.pack(side="left", anchor="c")

        ###############################################################

        CurrentDataSelectionFrame = ttk.Frame(WindowTwoUserInputFrame, padding=(14, 20, 10, 39))

        ttk.Label(CurrentDataSelectionFrame, text="CURRENT DATA UPLOAD", font=FONT_14_bold).pack(anchor="c")

        # Create data selection list box


        current_selection_data_listbox = tk.Listbox(CurrentDataSelectionFrame, relief="raise", borderwidth=1,
        listvariable=current_data_upload_result, exportselection=False,width=38, height=6)


        # add vertical scrollbar to list
        current_selection_data_scroll_bar_v = tk.Scrollbar(CurrentDataSelectionFrame, orient="vertical",
                                                           command=current_selection_data_listbox.yview, width=10)
        current_selection_data_listbox.config(yscrollcommand=current_selection_data_scroll_bar_v.set)
        current_selection_data_scroll_bar_v.pack(side="right", fill="y")

        # add horizontal scrollbar to list
        current_selection_data_scroll_bar_h = tk.Scrollbar(CurrentDataSelectionFrame, orient="horizontal",
                                                           command=current_selection_data_listbox.xview, width=10)
        current_selection_data_listbox.config(xscrollcommand=current_selection_data_scroll_bar_h.set)
        current_selection_data_scroll_bar_h.pack(side="bottom", fill="x")

        current_selection_data_listbox.pack(expand=True, fill="both")
        CurrentDataSelectionFrame.pack(side="left", anchor="n")
        ###############################################################


        WindowTwoUserInputFrame.pack(anchor="c", padx=292, pady=15)

        ClassifierEvaluation_MainFrame = tk.Frame(WindowTwoCanvasFrame, bg="white")



        ttk.Button(ClassifierEvaluation_MainFrame, width=8, default='active', text="Submit",command=ExecuteAlreadyTrainedModel).pack(anchor="c")
        tk.Label(ClassifierEvaluation_MainFrame, textvariable=ClassifierParameters_WindowTwo, wraplength=825)


        ###########################
        ###########################

        output_MainFrame = ttk.Frame(ClassifierEvaluation_MainFrame, relief="raised", borderwidth=1)

        ###########################
        ###########################

        # confusion matrix, model accuracy & error

        model_evaluation_statistics_frame = ttk.Frame(output_MainFrame, padding=(10, 10, 10, 10))
        model_evaluation_statistics_subframe = ttk.Frame(model_evaluation_statistics_frame)
        ttk.Label(model_evaluation_statistics_subframe, text="confusion matrix, model accuracy & error".upper(),
                  font=FONT_13_bold).pack(anchor="c")

        if sys.platform == "linux":
            model_evaluation_statistics_listbox = tk.Listbox(model_evaluation_statistics_subframe, relief="raise",
                                                             exportselection=False, selectmode="multiple",
                                                             font=["FreeMono", 12, "bold"],
                                                             listvariable=ConfusionMatrixModelAccuracyError_WindowTwo,
                                                             width=34, height=10)
        elif sys.platform == "win64" or sys.platform == "win32":
            model_evaluation_statistics_listbox = tk.Listbox(model_evaluation_statistics_subframe, relief="raise",
                                                             exportselection=False, selectmode="multiple",
                                                             font=["Courier", 9],
                                                             listvariable=ConfusionMatrixModelAccuracyError_WindowTwo,
                                                             width=54, height=10)
        else:
            model_evaluation_statistics_listbox = tk.Listbox(model_evaluation_statistics_subframe, relief="raise",
                                                             exportselection=False, selectmode="multiple",
                                                             font=["Courier", 15, "bold"],
                                                             listvariable=ConfusionMatrixModelAccuracyError_WindowTwo,
                                                             width=34, height=10)

            # add vertical scrollbar to list
        model_evaluation_statistics_scrollbar_v = tk.Scrollbar(model_evaluation_statistics_subframe, orient="vertical",
                                                               command=model_evaluation_statistics_listbox.yview,
                                                               width=10)
        model_evaluation_statistics_listbox.config(yscrollcommand=model_evaluation_statistics_scrollbar_v.set)
        model_evaluation_statistics_scrollbar_v.pack(side="right", fill="y")

        # add vertical scrollbar to list
        model_evaluation_statistics_scrollbar_h = tk.Scrollbar(model_evaluation_statistics_subframe,
                                                               orient="horizontal",
                                                               command=model_evaluation_statistics_listbox.xview,
                                                               width=10)
        model_evaluation_statistics_listbox.config(xscrollcommand=model_evaluation_statistics_scrollbar_h.set)
        model_evaluation_statistics_scrollbar_h.pack(side="bottom", fill="x")

        model_evaluation_statistics_listbox.pack(expand=True, fill="both")
        model_evaluation_statistics_subframe.pack(anchor="n")

        # output confusion matrix, model accuracy & error button
        model_evaluation_statistics_frame.pack(side="left", anchor="n")

        ###########################
        ###########################

        subframe1 = ttk.Frame(output_MainFrame, padding=(0, 180, 0, 0), relief="raised", borderwidth=1)
        ttk.Label(subframe1).pack()
        subframe1.pack(side="left", anchor="c")

        ###########################
        ###########################

        # testing results
        testing_results_frame = ttk.Frame(output_MainFrame, padding=(14, 10, 10, 10))
        testing_results_subframe = ttk.Frame(testing_results_frame)
        ttk.Label(testing_results_subframe, text="testing result: id 一 prediction".upper(), font=FONT_13_bold).pack(
            anchor="c")
        if sys.platform == "linux":
            testing_results_listbox = tk.Listbox(testing_results_subframe, relief="raise", exportselection=False,
                                                 selectmode="multiple", listvariable=testing_result_WindowTwo, width=40,
                                                 height=10)
        elif sys.platform == "win64" or sys.platform == "win32":
            testing_results_listbox = tk.Listbox(testing_results_subframe, relief="raise", exportselection=False,
                                                 selectmode="multiple", listvariable=testing_result_WindowTwo, width=53,
                                                 height=10)
        else:
            testing_results_listbox = tk.Listbox(testing_results_subframe, relief="raise", exportselection=False,
                                                 selectmode="multiple", listvariable=testing_result_WindowTwo, width=36,
                                                 height=10)

        # add vertical scrollbar to list
        testing_results_scrollbar_v = tk.Scrollbar(testing_results_subframe, orient="vertical",
                                                   command=testing_results_listbox.yview, width=10)
        testing_results_listbox.config(yscrollcommand=testing_results_scrollbar_v.set)
        testing_results_scrollbar_v.pack(side="right", fill="y")

        # add vertical scrollbar to list
        testing_results_scrollbar_h = tk.Scrollbar(testing_results_subframe, orient="horizontal",
                                                   command=testing_results_listbox.xview, width=10)
        testing_results_listbox.config(xscrollcommand=testing_results_scrollbar_h.set)
        testing_results_scrollbar_h.pack(side="bottom", fill="x")

        testing_results_listbox.pack(expand=True, fill="both")
        testing_results_subframe.pack(anchor="n")

        # output testing results button
        ttk.Button(testing_results_frame, text="Export Testing", command=ExportTestingResultWindowTwo).pack(
            side="bottom", anchor="e")
        testing_results_frame.pack(side="left", anchor="n")

        ###########################
        ###########################

        output_MainFrame.pack()

        ###########################
        ###########################

        if sys.platform == "linux":
            ClassifierEvaluation_MainFrame.pack(anchor="c", pady=45)

        elif sys.platform == "win64" or sys.platform == "win32":
            ClassifierEvaluation_MainFrame.pack(anchor="c", pady=90)

        else:
            ClassifierEvaluation_MainFrame.pack(anchor="c", pady=70)


#==============================================================================
# Function UploadAlreadyTrainedModel: used to upload PKL based file.
# PKL file include:
# index [0]: Features for Data Training
# index [1]: Classifier
# index [2]: Classifier confusion matrix, accuracy and error

#==============================================================================
def UploadAlreadyTrainedModelWindowTwo ():
    # globalize both pkl data file and its name, the file name will show in user
    # "current data upload" window which show user Log History for any file upload.


    global AlreadyTrainedModel_name
    AlreadyTrainedModel_name=""
    AlreadyTrainedModel_name=DenseNetController.UploadAlreadyTrainedModelWindowTwo()

# ==============================================================================
# Function clearClassifica:
#
# reset initialized global variables to original values
# ==============================================================================
def clearClassifica():
    copyLocals = ['current_data_upload_result',
                  'widgets', 'dependent_data_file_name',
                  'OpenDependentDataFile_output_ReadFile_new',
                  'OpenDependentDataFile_output_ReadFile_oroginal',
                  'target_data_file_name',
                  'OpenTargetDataFile_output_ReadFile_new',
                  'features_data_file_name',
                  'OpenFeaturesDataFile_output_ReadFile_new',
                  'test_data_file_name',
                  'OpenTestDataFile_output_ReadFile_new',
                  'OpenTestDataFile_output_ReadFile_oroginal',
                  'current_data_upload',
                  'training_data_upload_widgets',
                  'training_data_upload_selection_result',
                  'CommonFeatures_for_DataTraining',
                  'CallClassifier',
                  'confusion_matrix_accuracy_error_result',
                  'Classifier_Pramaters',
                  'predicted_value',
                  'testing_value',
                  'AlreadyTrainedModel_joblib',
                  'AlreadyTrainedModel_name',
                  'test_data_file_nameWindowTwo',
                  'OpenTestDataFile_output_ReadFile_newWindowTwo',
                  'testing_value_WindowTwo']

    for var in copyLocals:
        if var in globals().keys():
            del (globals()[var])
    global widgets, current_data_upload, current_data_upload_result, training_data_upload_widgets

    widgets = []
    current_data_upload = ['#Use My Own Training Data Uploaded Files',
                           'Dependent Data: Not Uploaded',
                           'Target Data: Not Uploaded',
                           'Features Data: Not Uploaded',
                           'Test Data: Not Uploaded',
                           '------------------------------------',
                           '#Already Trained My Model Uploaded Files',
                           'Model: Not Uploaded',
                           'Test Data: Not Uploaded',
                           '------------------------------------',
                           '#Upload History']
    current_data_upload_result = ''
    training_data_upload_widgets = []
def OpenTestDataFileWindowTwo():
    data_file_name=""
    read_file=""
    return DenseNetController.OpenTestDataFileWindowTwo()
#==============================================================================
# Function UserCurrentDataFileUpload: trace any modification occure to dependent,
# target, Features and testing data file upload, as well as tranied model. It
# also provide user with Log History for any file upload.
#
# Six inputs:
# arg0_StandardizeDataFile_func:   e.g. StandardizeDependentDataFile

# arg1_current_data_upload:        current_data_upload

# data_file_upload_index:          index value used to index current_data_upload
#                                  for corresponding namespace

# arg2_upload_type:                e.g. "Dependent Data: "

# arg3_file_name:                  e.g. dependent_data_file_name

# arg4_file_path:                  e.g. "#Dependent Data Path:"

# arg5_current_data_upload_result: current_data_upload_result
#==============================================================================
def CurrentDataFileUploadTracker(arg0_StandardizeDataFile_func,
                                 arg1_current_data_upload,
                                 data_file_upload_index,
                                 arg2_upload_type,
                                 arg3_file_name,
                                 arg4_file_path,
                                 arg5_current_data_upload_result):

    arg1_current_data_upload[data_file_upload_index]= (arg2_upload_type.title()
                                              +(arg3_file_name.split("/")[-1]))
    if arg3_file_name == "Not Uploaded":
        pass
    else:
        arg1_current_data_upload.append(datetime.datetime.now())
        arg1_current_data_upload.append(arg4_file_path)
        arg1_current_data_upload.append(arg3_file_name)
        arg1_current_data_upload.append("")
    arg5_current_data_upload_result.set(arg1_current_data_upload)

#==============================================================================
# Function StandardizeTestDataFileWindowTwo: will add "id" header to column index 0

# "id" header
#==============================================================================
def StandardizeTestDataFileWindowTwo():
    ''' function to standardize test data file '''
    # globalize test data file name to show in user "current data upload"
    # window which show user Log History for any file upload.

    global test_data_file_nameWindowTwo
    global OpenTestDataFile_output_ReadFile_newWindowTwo

    # call function where files been uploaded
    OpenTestDataFile_output = OpenTestDataFileWindowTwo()

    # index 0 = file name
    OpenTestDataFile_output_FileName = OpenTestDataFile_output[0]
    # index 1 = read file
    OpenTestDataFile_output_ReadFile = OpenTestDataFile_output[1]

    # if both file name and read file are empty (no file uploaded)
    if (OpenTestDataFile_output_ReadFile == "empty"
        and OpenTestDataFile_output_FileName == "empty"):
        # reset test file name to "not uploaded"
        test_data_file_nameWindowTwo = "Not Uploaded"


    # if both file name and read file are not empty (file has been uploaded)
    if (OpenTestDataFile_output_ReadFile != "empty"
        and OpenTestDataFile_output_FileName != "empty"):
        # keep test file name the same
        test_data_file_nameWindowTwo = OpenTestDataFile_output_FileName

        OpenTestDataFile_output_ReadFile_newWindowTwo = OpenTestDataFile_output_ReadFile
        OpenTestDataFile_output_ReadFile_newWindowTwo.reset_index(inplace=True)
        accessions_headr = list(OpenTestDataFile_output_ReadFile_newWindowTwo.columns)[0]
        # change first column headr (CEL files IDs) to id
        OpenTestDataFile_output_ReadFile_newWindowTwo = (
                OpenTestDataFile_output_ReadFile_newWindowTwo.rename(columns =
                                                {accessions_headr:'id'}))


    # if file name not empty and read file is empty (no file uploaded)
    if (OpenTestDataFile_output_ReadFile == "empty" and
        OpenTestDataFile_output_FileName != "empty"):
        # reset test file name to "not uploaded"
        test_data_file_nameWindowTwo = "Not Uploaded"

        file_name_error= OpenTestDataFile_output_FileName.split("/")[-1]
        ErrorMessage("Test Data File In Use",
                 "An error has occurred during parsing \""+file_name_error\
                 +"\", the error may have been caused by dataset format!")




#==============================================================================
# Function OpenTestDataFile: used to upload test data file, as well as
# read and transpose test data file to use by Sklearn.

#==============================================================================
def OpenTestDataFile():
 data_file_name, read_file = OpenTestDataFile()
# ==============================================================================
# Function center:
#
# Center tkinter on user display
# ==============================================================================
def center(toplevel):
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight() - 60
    size = (1440, 820)
    x = w / 2 - size[0] / 2
    y = h / 2 - size[1] / 2
    # set dimensions of ClassificaIO to 1440x820
    # ClassificaIO will appear at x and y
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))


# ==============================================================================
# Function to run ClassificaIO script and center tkinter window
# ==============================================================================
def gui():
    global root,top
    root=tk.Tk()
    top=TopLevel(root)
    top.style = ttk.Style()
    if sys.platform == "linux" or sys.platform == "win64" or sys.platform == "win32":
        top.style.theme_use("clam")
    else:
        top.style.theme_use("aqua")
    center(root)
    #init the Gui params
    DenseNetController.init(root, top)
    root.mainloop()
    # clearClassifica()
class StdoutRedirector(object):
    def __init__(self, queue):
        self.output = queue

    def write(self, string):
        self.output.put(string)

    def flush(self):
        pass


if __name__ == '__main__':
    gui()
