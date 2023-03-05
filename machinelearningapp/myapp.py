# -*- coding: utf-8 -*-
import os, sys, io
from flask import Flask, render_template, render_template_string, url_for, redirect
from flask import jsonify, session, abort, request, flash, send_file, send_from_directory
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from flask import Markup
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import joblib
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import math, time
from docx import Document
from flask_simplelogin import SimpleLogin
import docx
from sklearn.metrics import confusion_matrix
from urllib.request import urlopen
from flask_wtf import FlaskForm
from wtforms import DateField
#from docx2pdf import convert
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from os import listdir
from os.path import isfile, join
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from sqlalchemy import create_engine
import time
import datetime
import psycopg2
import json, re, ast
from sklearn.preprocessing import MinMaxScaler
from docx.shared import Pt
import datetime
#from mpl_toolkits.mplot3d import Axes3D
import pytz, random, string
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
##from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, login_required
from flask_security import Security, login_required, SQLAlchemySessionUserDatastore, current_user
#from database import db_session, init_db
#from models import User, Role
from sklearn.cluster import KMeans
#from flask_user import current_user, login_required, roles_required, UserManager, UserMixin, user_registered
from PIL import Image
from resizeimage import resizeimage
#from flask_dropzone import Dropzone
from io import BytesIO
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVC
from matplotlib import pyplot
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import solve_triangular
from sympy import *
import datetime
import re, string
import warnings
import glob, pickle
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import ClassifierChain
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import unicodedata
from sklearn import metrics



activationlist = ["identity", "logistic", "tanh", "relu"]
solverlist = ["adam", "lbfgs", "sgd"]
learningratelist = ["constant", "invscaling", "adaptive"]
listofscalersandencoders = ["MinMaxScaler", "StandardScaler", "LabelEncoder", "OneHotEncoder", "KBinsDiscretizer", "LabelBinarizer"]
linearlist = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Chained Regression", "Random Forest Regressor", "Support Vector Regressor"]
#dump(sc, 'std_scaler.bin', compress=True)
#sc=load('std_scaler.bin')

path_rm = "G:\\Dropbox\\work\\home\\ai\\static\\models\\"
UPLOAD_FOLDER = "G:\\Dropbox\\work\\home\\ai\\static\\uploads"
DOWNLOAD_FOLDER = "G:\\Dropbox\\work\\home\\ai\\static\\downloads"
MODEL_FOLDER = "G:\\Dropbox\\work\\home\\ai\\static\\models"

ALLOWED_EXTENSIONS = set(['xlsx', 'csv', 'xls'])

def insertdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Singapore"))
    hr = str(currentDT.time().hour)
    mi = str(currentDT.time().minute)
    sc = str(currentDT.time().second)
    ms = str(currentDT.time().microsecond)
    timeString = hr+'_'+mi+'_'+sc+'_'+ms
    insertdt = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    DATE_UNDERSCORE = currentDT.strftime("%Y_%m_%d")
    return insertdt

def get_datetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Singapore"))
    hr = str(currentDT.time().hour)
    mi = str(currentDT.time().minute)
    sc = str(currentDT.time().second)
    ms = str(currentDT.time().microsecond)
    timeString = hr+'_'+mi+'_'+sc+'_'+ms
    DATE = currentDT.strftime("%Y_%m_%d_")
    DATE_UNDERSCORE = currentDT.strftime("%Y_%m_%d")
    markTime = hr+'_'+mi+'_'+sc
    dateandtime = DATE + markTime
    return dateandtime

def LS_intersect(p0,a0,p1,a1):
    """
    :param p0 : Nx2 (x,y) position coordinates
    :param p1 : Nx2 (x,y) position coordinates
    :param a0 : angles in degrees for each point in p0
    :param a1 : angles in degrees for each point in p1    
    :return: least squares intersection point of N lines from eq. 13 in 
             http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf
    """    

    ang = np.concatenate( (a0,a1) ) # create list of angles
    # create direction vectors with magnitude = 1
    n = []
    for a in ang:
        n.append([np.cos(np.radians(a)), np.sin(np.radians(a))])
    pos = np.concatenate((p0[:,0:2],p1[:,0:2])) # create list of points
    n = np.array(n)

    # generate the array of all projectors 
    nnT = np.array([np.outer(nn,nn) for nn in n ]) 
    ImnnT = np.eye(len(pos[0]))-nnT # orthocomplement projectors to n

    # now generate R matrix and q vector
    R = np.sum(ImnnT,axis=0)
    q = np.sum(np.array([np.dot(m,x) for m,x in zip(ImnnT,pos)]),axis=0)

    # and solve the least squares problem for the intersection point p 
    return np.linalg.lstsq(R,q,rcond=None)[0]

def intersect(P0,P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function 
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    """
    # generate all line direction vectors 
    n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors 
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    # see fig. 1 

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q,rcond=None)[0]

    return p

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

gammalist = ['scale','auto']
kernellist = ['rbf', 'linear', 'poly', 'sigmoid']


#local db-----
engine = create_engine('postgresql+psycopg2://postgres:xxx@localhost:5432/mydb') # SELECT * FROM TABLE (DATAFRAME)
db = psycopg2.connect(database="mydb", user='postgres', password='xxx', host='localhost', port= '5432') # INSERTING DATA ROW, UPDATING DATA ROW
cur = db.cursor()



# #initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['SECURITY_RECOVERABLE'] = True
app.config['MAX_CONTENT_LENGTH'] = 200 * 1000 * 1000
app.config['SECRET_KEY'] = "OCML3BRawWEUeaxcuKHLpw"
app.config['SECURITY_PASSWORD_SALT'] = app.config['SECRET_KEY'] #os.urandom(12) #os.environ.get("SECURITY_PASSWORD_SALT", '146585145368132386173505678016728509634')
#app.config['SIMPLELOGIN_USERNAME'] = 'homerml'
#app.config['SIMPLELOGIN_PASSWORD'] = 'aiporTal1001'
#app.config['DEBUG'] = True

clustersizelist = ['3','4','5','6','7','8']

@app.route("/")
def home():
        return render_template("index.html")

@app.route("/ingest")
def ingest():
        return render_template("ingest.html")

def is_float(value):
    return isinstance(value, float)

def is_int(value):
    if type(value)==str:
        return value.isnumeric()

##def is_fl(value):
##    if type(value)==str:
##        if value.isnumeric()==False and value.isalnum():
##            return True

##def is_fl(value):
##    if type(value)==str:
##        if all(x.isalnum() or x.isspace() for x in value):
##            return True

def is_date(value):
    if type(value)==str:
        try:
            dfdt = pd.to_datetime(value)
            return True
        except:
            return False

#def is_fl(value):
    #if type(value)==str:
        #if (bool(re.match('^[0-9].*$', value))==True): #value.isalnum()==False or value.isnumeric() and is_date(value)==False:
            #return True

def is_fl(value):
    if type(value)==str:
        try:
            dfdt = pd.to_datetime(value)
            return False
        except:
            if value.isnumeric()==False and value.replace('.','').isdigit():
                return True

 
def contains_button(strvalue):
    if type(strvalue)==str:
        if 'button' in strvalue:
            return True

@app.template_filter('nl2br')
def nl2br(s):
    return s.replace("\n", "<br>")

#def multiply(x, y):
#    return str(x * y)

##func_dict = {
##    "is_float": is_float,
##    "is_int": is_int,
##}
##
##def render(template):
##    env = Environment(loader=FileSystemLoader("templates"))
##    jinja_template = env.get_template(template)
##    jinja_template.globals.update(func_dict)
##    template_string = jinja_template.render()
##    return template_string

@app.route("/ingestprocessor", methods =['GET', 'POST'])
def ingestprocessor():
    
    if request.method == 'GET':

        query_str = "SELECT mark_xy FROM xymarker"
        df = pd.read_sql_query(query_str, con=engine)
        listmarkers = df['mark_xy'].tolist()
        print("List markers: ", listmarkers)

        return render_template("ingest.html", tablenamelist=listmarkers)
          
    if request.method == 'POST':

        if request.form.get("action_select_tb") == "Select":
            tabletable = request.form['tbname']
            querytable = """SELECT * FROM {}""".format(tabletable)
            df = pd.read_sql_query(querytable, con=engine)
            df_origin = df
            df = df.drop(['Edit', 'Delete'], axis=1)
            colnames = df.columns
            btnflag = False
            selectlist = request.form.get('plaintablelist')
            mylist = ast.literal_eval(selectlist)

            #tablename = request.form['sheetname']
            #print("Calling table: ", tablename)
            #querytable = """SELECT * FROM {}""".format(tablename)
            #df = pd.read_sql_query(querytable, con=engine)
            #colnames = df.columns
            #dfcol = df.drop(['Edit', 'Delete'], axis=1)
            colnames_origin = df_origin.columns

            #selecte = request.form.get('tnames')
            #myinput = ast.literal_eval(selecte)
            #print("myinput", myinput)

            result_tuples = df_origin.to_records() #df_col

            headerPt = []

            for column_name in df_origin.columns: 

                columnchoiceform = """

                                  <input type="hidden" name="column_name" value='{0}'>
                                
                                  <div class="mt-3">
                                     <button type="submit" class="btn btn-primary btn-sm" name="getcol" value='{1}'>{2}</button>
                                  </div>
                              
                             """.format(column_name, column_name, column_name)

                headerPt.append(columnchoiceform)
            
            return render_template("ingest.html", result_tuples = result_tuples, is_float = is_float, is_int = is_int, is_fl = is_fl, is_date = is_date, contains_button = contains_button, df_table = [df_origin.to_html(classes='data table-striped', escape=False)],
                                                            colheaders=df_origin.columns, colnames=colnames, tablelist=mylist, sheetn=tabletable, headerPt=headerPt, btnflag=btnflag, tablen=tabletable, tablenamelist=mylist) # sheetnamelist=listsheets)
            
            #return render_template("ingest.html", tablenamelist=mylist, colnames=colnames, btnflag=btnflag, tablen=tabletable)
            
        
        if request.form.get("action_upload") == "Upload":
            file = request.files['ingestion_uploadfile']
            nametable = request.form['nametable']
            if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    #uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    #time.sleep(1)
                    filen = filename
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], get_datetime()+filename))
                    filename = os.path.join(app.config['UPLOAD_FOLDER'], get_datetime()+filename)
                    df = pd.DataFrame()
                    #df_init = pd.read_excel(filename)
                    #df_init = df_init.loc[:, ~df_init.columns.str.contains('^Unnamed')] #Remove unnamed index
                    #print("Read: ", df)
                    tsql = ""
                    query_str = "SELECT mark_xy FROM xymarker"
                    dfm = pd.read_sql_query(query_str, con=engine)
                    listmarkers = dfm['mark_xy'].tolist()
                        
                    if ".csv" in filename:

                        tablelist = []
                           
                        df = pd.read_csv(filename)

                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        
                        newcols = []
                        for i in df.columns:
                            trimmedcol = re.sub(r'[\W_]+', ' ', i)
                            newcols.append(trimmedcol)
                        df.columns = newcols
                        
                        columnnames = []
                        for colnn in list(df.columns):
                            corrected = unicodedata.normalize('NFKC', colnn)
                            columnnames.append(corrected)

                        df.columns = columnnames 
                
                        #-------------------------------------------
                        df = df.replace(r'^\s*$', np.nan, regex=True)

                        #df = df.replace(['-', '?'], [np.nan, np.nan], regex=True)
                        df = df.replace({'-': np.nan, '?': np.nan})
                        
                        df = df.dropna(how='any', axis=0)

                        df = df.dropna(how='all', axis=1)
                        
                        print("Final dataframe size: ", df.shape)

                        colnames = df.columns
                        totalrows = len(df)
                        
                        editTest = []
                        deleteTest = []
                        
                        for rowctr in df.index:
                               
                                editform = """
                                                          
                                              <input type="hidden" name="skidrowedit" value='{}'>
    
                                              <div class="mt-3">
                                                <button type="submit" class="btn btn-success" name="getaction" value="Edit">Edit</button>
                                              </div>
                                        
                                      """.format(rowctr)
                                deleteform = """

                                              <input type="hidden" name="skidrowdelete" value='{}'>
                                            
                                              <div class="mt-3">
                                                 <button type="submit" class="btn btn-primary" name="getaction" value="Delete">Delete</button>
                                              </div>
                                          
                                         """.format(rowctr)
                          
                                editTest.append(editform)
                                deleteTest.append(deleteform)
                        
                        tsql = nametable + filen.replace(".csv","")

                        listsheets = []
                    
                    if ".xlsx" in filename or ".xls" in filename:
                    
                        xl = pd.ExcelFile(filename)
                        listsheets = xl.sheet_names  # see all sheet names
                        print("Sheetnames: ", listsheets)
                        #xl.parse(sheet_name)  # read a specific sheet to DataFrame
                        tablelist = []
                        
                        for tabsheet in listsheets:

                                df = pd.read_excel(filename, sheet_name=tabsheet)

                                df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #Remove unnamed index
                                #remove brackets-----
                                newcols = []
                                for i in df.columns:
                                    trimmedcol = re.sub(r'[\W_]+', ' ', i)
                                    #trimmedcol = unicodedata.normalize('NFKC', trimmedcol)
                                    newcols.append(trimmedcol)
                                df.columns = newcols
                                #---------------------
                                columnnames = []
                                for colnn in list(df.columns):
                                    corrected = unicodedata.normalize('NFKC', colnn)
                                    columnnames.append(corrected)

                                df.columns = columnnames 
                                #print(df)
                                #print("read: ", df)
                                #numOfVariables = len(df.columns)
                                #df.dropna(how='all', axis=0, thresh=numOfVariables/2, inplace=True)
                                #print("drop thres: ", df)
                                
                                #df = pd.read_excel(filename)

                                #-------------------------------------------
                                df = df.replace(r'^\s*$', np.nan, regex=True)
                                #print("Replace: ", df)

                                #df = df.replace(r'-', np.nan, regex=True)
                                df = df.replace({'-': np.nan, '?': np.nan})
                                #print("Replace: ", df)
                                
                                #df = df.drop_duplicates()
                                #print("Drop dup: ", df)
                                
                                #empty_cols = [col for col in df.columns if df[col].isnull().all()]
                                #df = df.drop(empty_cols, axis=1, inplace=True)
                                #print("Drop NULL: ", df)
                                ##-->df = df.dropna(how='any', axis=0)
                                df = df.dropna(how='all', axis=1)
                                
                                #print("Drop NA: ", df)
                                print("Final dataframe size: ", df.shape)
                                
                                #df = df.drop_duplicates()
                                #df = df.replace("\n", "", regex=True)
                                #--------------------------------------------

                                colnames = df.columns
                                totalrows = len(df)
                                
                                #print("drop col: ", df)
                                #listTest = []

                                editTest = []
                                deleteTest = []
                ##                linkup = """<button type="submit" name="ingest_editbutton" value="ingest_edit" class="btn btn-success">Edit</button><br><button type="submit" name="ingest_deleterow" value="ingest_deleterow" class="btn btn-danger">Delete</button>"""
                ##                editform = """
                ##                                <form action="/edit_ingest" method="post" enctype="multipart/form-data">
                ##         
                ##                                  <input type="hidden" name="sid" value='{}'>
                ##                                  
                ##                                  <div class="mt-3">
                ##                                    <button type="submit" class="btn btn-success">Edit</button>
                ##                                  </div>
                ##                                </form>
                ##                              """
                ##                deleteform = """               
                ##                                   <form action="/delete_ingest" method="post" enctype="multipart/form-data">
                ##                                 
                ##                                      <input type="hidden" name="sid" value='{}'>
                ##                                  
                ##                                      <div class="mt-3">
                ##                                         <button type="submit" class="btn btn-primary">Delete</button>
                ##                                      </div>
                ##                                   </form>
                ##                                 """
                                ## <input type="hidden" id="seltable" name="seltable" value='{{sheetn}}'>
                                ## <input type="hidden" id="tnames" name="tnames" value='{{tablelist}}'>
                                
                                #tablename = request.form['seltable']
                                #print("Selected table pre upload: ", tablename)
                                
                                ## 
                                ## <input type="submit" class="btn btn-success" name="action_get" value="Edit">
                                for rowctr in df.index:
                                        #rowctr = list(df.loc[row, :])
                                        #print("row: ", rowctr)
                                        #showrow = df.loc[rowctr,:]
                                        editform = """
                                                                  
                                                      <input type="hidden" name="skidrowedit" value='{}'>
            
                                                      <div class="mt-3">
                                                        <button type="submit" class="btn btn-success" name="getaction" value="Edit">Edit</button>
                                                      </div>
                                                
                                              """.format(rowctr)
                                        deleteform = """

                                                      <input type="hidden" name="skidrowdelete" value='{}'>
                                                    
                                                      <div class="mt-3">
                                                         <button type="submit" class="btn btn-primary" name="getaction" value="Delete">Delete</button>
                                                      </div>
                                                  
                                                 """.format(rowctr)
                                    #trow = """<input type="hidden" name="eachrow" value='{}'>""".format(str(rowctr))
                                    #linkup = """<button type="submit" name="ingest_editbutton" value="ingest_edit" class="btn btn-success">Edit</button><br><button type="submit" name="ingest_deleterow" value="ingest_deleterow" class="btn btn-danger">Delete</button>"""
                                    #for v in range(len(df)):
                                        editTest.append(editform)
                                        deleteTest.append(deleteform)
                                
                                tsql = nametable + tabsheet
                                    
                    df.insert(0, 'Edit', editTest)
                    df.insert(1, 'Delete', deleteTest)
                    df = df.replace("\n", "", regex=True)

                    #tsql = nametable + tabsheet
                    tsql = ''.join(e for e in tsql if e.isalnum()).lower()
                    tablelist.append(tsql)

                    df.to_sql(tsql, con=engine, if_exists='replace', index=False)

                    try:

                        insertstatement = """ INSERT INTO xymarker (dateandtime, mark_xy, username) VALUES ('%s', '%s', '%s') """ % (insertdatetime(), tsql, "tommy")
                        cur.execute(insertstatement)
                        db.commit()

                    except:

                        insertstatement = """ INSERT INTO xymarker (dateandtime, mark_xy, username) VALUES ('%s', '%s', '%s') ON CONFLICT (mark_xy) DO NOTHING """ % (insertdatetime(), tsql, "tommy")
                        cur.execute(insertstatement)
                        db.commit()

                    time.sleep(0.3)
      
                    querytable = "SELECT * FROM {}".format(tablelist[0])
                    df_firstrun = pd.read_sql_query(querytable, con=engine)
                    colnames = df_firstrun.columns
                    columnames = []
                    for col in colnames:
                    #print(col)
                        if col != "Edit" and col != "Delete":
                            columnames.append(col)
                 
                    btnflag = False
               
                    
                    flash("File saved.")
                    return render_template("ingest.html", tablenamelist=listmarkers, colnames=columnames, btnflag=btnflag,
                                           sheetnamelist=listsheets, tablelist=tablelist, sheetn=tablelist[0]) # df_table = [df.to_html(classes='data table-striped', escape=False)]
                   

        
        elif request.form.get("action_get") == "Confirm":
                
                selectedcolumnsX = request.form.getlist('picklistX')
                selectedcolumnsY = request.form.getlist('picklistY')
             
                            
                tabname = ""

                tabnam2 = request.form['seltable'] # seltable
                print("seltable select 3: ", tabnam2)
                
                tabnam = request.form['sheetname'] # seltable
                print("sheetname select 1: ", tabnam)
                if tabnam:
                    tabname = tabnam
                if tabnam2:
                    tabname = tabnam2
                
                tabnam = request.form['select_table']
                print("selecttable from db: ", tabnam)
                if tabnam:
                    tabname = tabnam

                myinput = []
                selecte = request.form.get('tnames')
                if selecte:
                    myinput = ast.literal_eval(selecte)
                    print("myinput: ", myinput)
                
                select = request.form.get('plaintablelist')
                if select:
                    myinput = ast.literal_eval(select)
                    print("myinput from db: ", myinput)

                #if not tabname:
                    #tabname = myinput[0]
                    
                querytable = "SELECT * FROM {}".format(tabname)
                df = pd.read_sql_query(querytable, con=engine)
                df_full = df
                columnames = df_full.columns
                
                dfX = df[selectedcolumnsX]
                dfY = df[selectedcolumnsY]

                dfX.to_sql(tabname+'inputx3', con=engine, if_exists='replace', index=False)
                dfY.to_sql(tabname+'labely3', con=engine, if_exists='replace', index=False)
                
                colnamesX = dfX.columns
                colnamesY = dfY.columns

                columnames = []
                for col in df_full.columns:
                    #print(col)
                    if col != "Edit" and col != "Delete":
                        columnames.append(col)

                detectbtnflag = True
                displayselected = False

                query_str = "SELECT mark_xy FROM xymarker"
                dfnm = pd.read_sql_query(query_str, con=engine)
                listmarkers = dfnm['mark_xy'].tolist()
                print("List markers: ", listmarkers)


               

                return render_template("ingest.html", df_tableX = [dfX.to_html(classes='data table-striped', escape=False)], df_tableY = [dfY.to_html(classes='data table-striped', escape=False)],
                                       Xcolnames=colnamesX, Ycolnames=colnamesY, colnames=columnames, detectbtnflag=detectbtnflag, selectedcolumnsX=selectedcolumnsX,
                                       displayselected=displayselected, selectedcolumnsY=selectedcolumnsY, tablelist=myinput, sheetn=tabname, tablenamelist=listmarkers)
            
        elif request.form.get('action_selectsheet') == "Select":

            tablename = request.form['sheetname']
            print("Calling table: ", tablename)
            querytable = """SELECT * FROM {}""".format(tablename)
            df = pd.read_sql_query(querytable, con=engine)
            #colnames = df.columns
            dfcol = df.drop(['Edit', 'Delete'], axis=1)
            colnames = dfcol.columns

            selecte = request.form.get('tnames')
            myinput = ast.literal_eval(selecte)
            print("myinput", myinput)

            result_tuples = df.to_records() #df_col

            headerPt = []

            for column_name in df.columns: 

                columnchoiceform = """

                                  <input type="hidden" name="column_name" value='{0}'>
                                
                                  <div class="mt-3">
                                     <button type="submit" class="btn btn-primary btn-sm" name="getcol" value='{1}'>{2}</button>
                                  </div>
                              
                             """.format(column_name, column_name, column_name)

                headerPt.append(columnchoiceform)
            
            return render_template("ingest.html", result_tuples = result_tuples, is_float = is_float, is_int = is_int, is_fl = is_fl, is_date = is_date, contains_button = contains_button, df_table = [df.to_html(classes='data table-striped', escape=False)],
                                                            colheaders=df.columns, colnames=colnames, tablelist=myinput, sheetn=tablename, headerPt=headerPt) # sheetnamelist=listsheets)

        elif request.form.get('getaction') == "Edit":
        
            onerow = request.form['skidrowedit']
            
            tablename = request.form['seltable']
            print("edit_ingest: seltable ", tablename)
            selecte = request.form.get('tnames')
            print("edit_ingest: tnames ", selecte)
            myinput = ast.literal_eval(selecte)

        
            querytable = """SELECT * FROM {}""".format(tablename)
   
            df = pd.read_sql_query(querytable, con=engine)
            dfcol = df.drop(['Edit', 'Delete'], axis=1)
            colnames = dfcol.columns
            #dft = df.index[int(onerow)].to_frame()
            dft = df.loc[int(onerow), :].to_frame().T
            dft = dft.drop(['Edit', 'Delete'], axis=1)

            colnn = list(dft.columns)
            print(colnn, type(colnn))
            val = dft.values.tolist()[0]
            print(val, type(val))
            val2 = dft.values[0].tolist()
            print("zerofront: ", val2, type(val2))

            nam = []
            lengthOfColumns = len(colnn)
            for i in range(lengthOfColumns):
                    strlabel = "col" + str(i)
                    nam.append(strlabel)

            editorform = []

            btnflag = True

            for c, v, n in zip(colnn, val, nam):
                htmlout = """
                                  <label>'{0}'</label>
                                  <input type="text" name='{2}' value='{1}'>
                                  <br>
                              
                         """.format(c, v, n)
                editorform.append(htmlout)

            print("colnn: ", colnn)
            
            print("nam: ", nam)
            #dft.insert(len(dft.columns), 'Correction', editorform)

            dfa = pd.DataFrame(editorform)
            
            dfa = dfa.replace("\n", "", regex=True)
            
            return render_template("ingest.html", df_table_selected = [dfa.to_html(classes='data table-striped', escape=False)], nam=nam, colnn=colnn, colnames=colnames, btnflag=btnflag,
                                                               selectedrow=onerow, tablelist=myinput, sheetn=tablename, colheaders=dfa.columns)

        elif request.form.get('getaction') == "Delete":

            onerow = request.form['skidrowdelete']

            tablename = request.form['seltable']
            print("edit_ingest: seltable ", tablename)
            selecte = request.form.get('tnames')
            print("edit_ingest: tnames ", selecte)
            myinput = ast.literal_eval(selecte)
            
            querytable = """SELECT * FROM {}""".format(tablename)
            
            df = pd.read_sql_query(querytable, con=engine)
            
            editTest = []
            deleteTest = []
            
            df.drop(df.index[int(onerow)], inplace=True) # sql delete
            df.reset_index(drop=True, inplace=True)

            for rowctr in df.index:
                            
                        editform = """
                                   
                                      <input type="hidden" name="skidrowedit" value='{}'>

                                      <div class="mt-3">
                                        <button type="submit" class="btn btn-success" name="getaction" value="Edit">Edit</button>
                                      </div>
                                
                              """.format(rowctr)
                        deleteform = """

                                      <input type="hidden" name="skidrowdelete" value='{}'>
                                    
                                      <div class="mt-3">
                                         <button type="submit" class="btn btn-primary" name="getaction" value="Delete">Delete</button>
                                      </div>
                                  
                                 """.format(rowctr)
                   
                        editTest.append(editform)
                        deleteTest.append(deleteform)
                            
        
            df['Edit'] = editTest
            
            df['Delete'] = deleteTest
            
            df = df.replace("\n", "", regex=True)
     
            df.to_sql(tablename, con=engine, if_exists='replace', index=False)

            time.sleep(0.5)
            
            dft = pd.read_sql_query(querytable, con=engine)

            colnames = []
            for col in dft.columns:
                if col != "Edit" and col != "Delete":
                    colnames.append(col)

            result_tuples = dft.to_records()

            btnflag = False
            
            return render_template("ingest.html", result_tuples = result_tuples, is_float = is_float, is_int = is_int, is_fl = is_fl, is_date = is_date, contains_button = contains_button,
                                                            df_table = [dft.to_html(classes='data table-striped', escape=False)],
                                                            colnames=colnames, btnflag=btnflag, selectedrow=onerow, tablelist=myinput, sheetn=tablename, colheaders=dft.columns)

        else:
            flash("Invalid. Please try again.")
            return redirect(url_for('ingest'))



@app.route("/edit_save", methods =['POST'])
def edit_save():
    if request.method == 'POST':
        onerow = request.form['selectedrow']
        tablename = request.form['seltable']
        selecte = request.form.get('tnames')
        myinput = ast.literal_eval(selecte)

        #name = request.form['name_arbitrary']
        #querytable = "SELECT * FROM tesdb"
        querytable = """SELECT * FROM {}""".format(tablename)
        df = pd.read_sql_query(querytable, con=engine)
        dfcol = df.drop(['Edit', 'Delete'], axis=1)
        colnames = dfcol.columns
        
        name = request.form.getlist('name_arbitrary')
        res1 = ast.literal_eval(name[0])
        print("editsave name: ", res1)
        coolio = request.form.getlist('col_arbitrary')
        res2 = ast.literal_eval(coolio[0])
        #coolio = request.form['col_arbitrary']
        print("editsave col: ", res2)
        lista = []
        cola = []
        for a in res1:
            ab = request.form[a]
            lista.append(ab)
        print("lista: ", lista)

        df.iloc[int(onerow), 2:] = lista
        df.to_sql(tablename, con=engine, if_exists='replace', index=False)

        btnflag = False
        
        flash("Data saved.")
        return render_template("ingest.html", df_table = [df.to_html(classes='data table-striped', escape=False)], colnames=colnames, btnflag=btnflag, tablelist=myinput, sheetn=tablename)


            

@app.route("/edit_ingest", methods =['POST'])
def edit_ingest():
    if request.method == 'POST':
            onerow = request.form['skidrowedit']
            tablename = request.form['seltable']
            print("edit_ingest: seltable ", tablename)
            selecte = request.form.get('tnames')
            print("edit_ingest: tnames ", selecte)
            myinput = ast.literal_eval(selecte)

    
            querytable = """SELECT * FROM {}""".format(tablename)
         
            df = pd.read_sql_query(querytable, con=engine)
            dfcol = df.drop(['Edit', 'Delete'], axis=1)
            colnames = dfcol.columns
            #dft = df.index[int(onerow)].to_frame()
            dft = df.loc[int(onerow), :].to_frame().T
            dft = dft.drop(['Edit', 'Delete'], axis=1)

            colnn = list(dft.columns)
            print(colnn, type(colnn))
            val = dft.values.tolist()[0]
            print(val, type(val))
            val2 = dft.values[0].tolist()
            print("zerofront: ", val2, type(val2))

            nam = []
            lengthOfColumns = len(colnn)
            for i in range(lengthOfColumns):
                    strlabel = "col" + str(i)
                    nam.append(strlabel)

            editorform = []

            btnflag = True

            for c, v, n in zip(colnn, val, nam):
                htmlout = """
                                  <label>'{0}'</label>
                                  <input type="text" name='{2}' value='{1}'>
                                  <br>
                              
                         """.format(c, v, n)
                editorform.append(htmlout)

            print("colnn: ", colnn)
            
            print("nam: ", nam)
            #dft.insert(len(dft.columns), 'Correction', editorform)

            dfa = pd.DataFrame(editorform)
            
            dfa = dfa.replace("\n", "", regex=True)
            
            return render_template("ingest.html", df_table_selected = [dfa.to_html(classes='data table-striped', escape=False)], nam=nam, colnn=colnn, colnames=colnames, btnflag=btnflag, selectedrow=onerow, tablelist=myinput, sheetn=tablename)

            
@app.route("/delete_ingest", methods =['POST'])
def delete_ingest():
    if request.method == 'POST':
            onerow = request.form['skidrowdelete']
        
            querytable = "SELECT * FROM tesdb"
            df = pd.read_sql_query(querytable, con=engine)
     
            editTest = []
            deleteTest = []
            
            df.drop(df.index[int(onerow)], inplace=True) # sql delete
            df.reset_index(drop=True, inplace=True)

            for rowctr in df.index:
               
                            editform = """
                                    <form action="/edit_ingest" method="post" enctype="multipart/form-data">
             
                                      <input type="hidden" name="skidrowedit" value='{}'>
                                      
                                      <div class="mt-3">
                                        <button type="submit" class="btn btn-success">Edit</button>
                                      </div>
                                    </form>
                                  """.format(rowctr)
                            deleteform = """               
                                       <form action="/delete_ingest" method="post" enctype="multipart/form-data">
                                     
                                          <input type="hidden" name="skidrowdelete" value='{}'>
                                          
                                          <div class="mt-3">
                                             <button type="submit" class="btn btn-primary">Delete</button>
                                          </div>
                                       </form>
                                     """.format(rowctr)
                  
                            editTest.append(editform)
                            deleteTest.append(deleteform)
                            
            #df.insert(0, 'Edit', editTest)
            df['Edit'] = editTest
            #df.insert(1, 'Delete', deleteTest)
            df['Delete'] = deleteTest
            df = df.replace("\n", "", regex=True)
            #df = df.drop_duplicates()
            #print(df)
            df.to_sql("tesdb", con=engine, if_exists='replace', index=False)

            time.sleep(0.5)

            #colnames = df.columns
            
            dft = pd.read_sql_query(querytable, con=engine)

            colnames = []
            for col in dft.columns:
                if col != "Edit" and col != "Delete":
                    colnames.append(col)

            btnflag = False
            
            return render_template("ingest.html", df_table = [dft.to_html(classes='data table-striped', escape=False)], colnames=colnames, btnflag=btnflag)



def isnan(value):
    try:
        return math.isnan(float(value))
    except:
        return False


    
def check_int(s):
    s = str(s)
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def detect_column_types(dfxx):

    msg_float = ""
    msg_none = ""
    msg_nan = ""
    msg_float2 = ""
    msg_dt = ""
    msg_int = ""
    msg_str = ""
    msg_str_alpha = ""
    msg_str_alnum = ""
    msg_str_int = ""
    msg_str_float = ""
    
    genlist = []

    for dfx in list(dfxx.columns):

        idfloat = []
        idisnan = []
        isfloater = []
        idisstr = []
        dtlist = []
        disnone = []
        idfloat = []
        isinteger = []
        alphalist = []
        alnumlist = []
        numericintlist = []
        numericfloatlist = []

        dfx1 = list(dfxx[dfx])
    
        for v in dfx1:
            
            cf = type(v)
            if cf == float:
                idfloat.append(v)

            cf = type(v)
            if isinstance(v, int):
                isinteger.append(v)
        

       
            cf = type(v)
            if cf == float:
                if math.isnan(v):
                    idisnan.append(math.nan)
                elif math.isnan(v)==False:
                    isfloater.append(v)
       

            #for v in list(dfx):
            cf = type(v)
            if cf == str:
                #idisstr.append(cf)
                try:
                    dfdt = pd.to_datetime(v)
                    dtlist.append("datetime")
                except:
                    idisstr.append(v)
                    #pass
                    if v.isalpha():
                        alphalist.append(v)

                    #if re.match("^[A-Za-z0-9]*$", v):
                    if v.isalnum() and not v.isnumeric() and not v.isalpha():
                        alnumlist.append(v)

                    if check_int(v):
                        numericintlist.append(v)
        
            cf = type(v)
            if not v:
                disnone.append('blanc')

        print(dfx,"--------------------------------------------")

        if len(idfloat) == len(dfx1):
            print("All are floats.")
            msg_float = "All are floats."
        elif len(idfloat) != len(dfx1):
            print(len(idfloat), " out of ", len(dfx1), " are floats.")
            msg_float = str(len(idfloat)) + " out of " + str(len(dfx1)) +" are floats."
        numFL = len(idfloat)
                
        if len(disnone) == len(dfx1):
            print("All are None.")
            msg_none = "All are None."
        elif len(disnone) != len(dfx1):
            print(len(disnone), " out of ", len(dfx1), " are None.")
            msg_none = str(len(disnone)) + " out of " + str(len(dfx1)) + " are None."
        numNONE = len(disnone)
            
        if len(idisnan) != len(dfx1):
            #print("Not all are floating numbers.")
            msg_nan = str(len(idisnan)) + " out of " + str(len(dfx1)) + " are NaN."
            print(len(idisnan), " out of ", len(dfx1), " are NaN.")
        elif len(idisnan) == len(dfx1):
            print("All are NaN.")
            msg_nan = "All are NaN."
        numNAN = len(idisnan)

        if len(isfloater) != len(dfx1):
            #print("Not all are floating numbers.")
            msg_float2 = str(len(isfloater)) + " out of " + str(len(dfx1)) + " are floats."
            print(len(isfloater), " out of ", len(dfx1), " are floats.")
        elif len(isfloater) == len(dfx1):
            print("All are floats.")
            msg_float2 = "All are floats."
        numFL2 = len(isfloater)

        if len(dtlist) == len(dfx1):
            print("All are DateTime.")
            msg_dt = "All are DateTime."
        elif len(dtlist) != len(dfx1):
            print(len(dtlist), " out of ", len(dfx1), " are DateTime.")
            msg_dt = str(len(dtlist)) + " out of " + str(len(dfx1)) + " are DateTime."
        numDATETIME = len(dtlist)

        if len(isinteger) == len(dfx1):
            print("All are Integers.")
            msg_int = "All are Integers."
        elif len(isinteger) != len(dfx1):
            print(len(isinteger), " out of ", len(dfx1), " are integers.")
            msg_int = str(len(isinteger)) + " out of " + str(len(dfx1)) + " are integers."
        numINT = len(isinteger)

        if len(idisstr) == len(dfx1):
            print("All are strings.")
            msg_str = "All are strings."
        elif len(idisstr) != len(dfx1):
            print(len(idisstr), " out of ", len(dfx1), " are strings.")
            msg_str = str(len(idisstr)) + " out of " + str(len(dfx1)) + " are strings."
        numSTR = len(idisstr)

        #-----------------------------------------
        if idisstr:
        
               
                print(len(alphalist), " out of ", len(idisstr), " are alphas in the string.")
                numALPHA = len(alphalist)
                msg_str_alpha = str(len(alphalist)) + " out of " + str(len(idisstr)) + " are alphas in the string."

                
                print(len(alnumlist), " out of ", len(idisstr), " are alphanums in the string.")
                numALNUM = len(alnumlist)
                msg_str_alnum = str(len(alnumlist)) + " out of " + str(len(idisstr)) + " are alphanums in the string."

                
                print(len(numericintlist), " out of ", len(idisstr), " are ints or floats in the string.")
                numNUMERIC_INT = len(numericintlist)
                msg_str_int = str(len(numericintlist)) + " out of " + str(len(idisstr)) + " are ints or floats in the string."



        idisnan = []
        isfloater = []
        idisstr = []
        dtlist = []
        disnone = []
        idfloat = []
        isinteger = []
        alphalist = []
        alnumlist = []
        numericintlist = []
        numericfloatlist = []



        continuous = [numFL, numNONE, numNAN, numFL2, numDATETIME, numINT, numSTR, numALPHA, numALNUM, numNUMERIC_INT]
        

  
        return continuous




@app.route("/dataprocessing", methods=['POST'])
def dataprocessing():
    if request.method == 'POST':

        if request.form.get("action_detect") == "Proceed to Next Stage":

                x_and_y = request.form['XYtablenames']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                selectedcolumnsX = request.form.get('xcolumnnames')
                selectedcolumnsY = request.form.get('ycolumnnames')

                print("X raw: ", selectedcolumnsX, type(selectedcolumnsX))
                print("Y raw: ", selectedcolumnsY, type(selectedcolumnsY))

                resX = ast.literal_eval(selectedcolumnsX)
                resY = ast.literal_eval(selectedcolumnsY)
                
                print("X extracted: ", resX, type(resX))
                print("X extracted: ", resY, type(resY))

              
                
                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)
            

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                #for c in selectedcolumnsX:
                #    print(c.index)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]


                return render_template("preprocessing.html", df_tableX = [df_x.to_html(classes='data table-striped', escape=False)], df_tableY = [df_y.to_html(classes='data table-striped', escape=False)],
                                                                             listoftables = [x_and_y], tablename=x_and_y, columnames_x=columnames_x, columnames_y=columnames_y)

                
    
        if request.form.get("action_detect") == "Process Data":
                 
                selectedcolumnsX = request.form.get('xcolumnnames')
                selectedcolumnsY = request.form.get('ycolumnnames')

                print("X raw: ", selectedcolumnsX, type(selectedcolumnsX))
                print("Y raw: ", selectedcolumnsY, type(selectedcolumnsY))

                resX = ast.literal_eval(selectedcolumnsX)
                resY = ast.literal_eval(selectedcolumnsY)
                
                print("X extracted: ", resX, type(resX))
                print("X extracted: ", resY, type(resY))

                tabname = request.form['seltable']

   
                
                querytable = "SELECT * FROM {}".format(tabname)
                df = pd.read_sql_query(querytable, con=engine)


                columnames = []
                for col in df.columns:
                    #print(col)
                    if col != "Edit" and col != "Delete":
                        columnames.append(col)

          
                
                dfxx = df[resX]
                dfxy = df[resY]
                
         

                msg_float = ""
                msg_none = ""
                msg_nan = ""
                msg_float2 = ""
                msg_dt = ""
                msg_int = ""
                msg_str = ""
                msg_str_alpha = ""
                msg_str_alnum = ""
                msg_str_int = ""
                msg_str_float = ""
                msg_punctuations = ""

                #for i in dfxx:

                for dfx in list(dfxx.columns):

                    idfloat = []
                    idisnan = []
                    isfloater = []
                    idisstr = []
                    dtlist = []
                    disnone = []
                    idfloat = []
                    isinteger = []
                    alphalist = []
                    alnumlist = []
                    numericintlist = []
                    numericfloatlist = []
                    punctuations = []

                    dfx1 = list(dfxx[dfx])
                
                    for v in dfx1:
                        print("v value type: ", type(v))
                        cf = type(v)
                        if cf == float:
                            idfloat.append(v)

                        cf = type(v)
                        if isinstance(v, int):
                            isinteger.append(v)
                    
     
                   
                        cf = type(v)
                        if cf == float:
                            if math.isnan(v):
                                idisnan.append(math.nan)
                            elif math.isnan(v)==False:
                                isfloater.append(v)
                   

                        #for v in list(dfx):
                        cf = type(v)
                        if cf == str:
                            #idisstr.append(cf)
                            try:
                                dfdt = pd.to_datetime(v)
                                dtlist.append("datetime")
                            except:
                                idisstr.append(v)
                                #pass
                                if v.isalpha():
                                    alphalist.append(v)

                                #if re.match("^[A-Za-z0-9]*$", v):
                                if v.isalnum() and not v.isnumeric() and not v.isalpha():
                                    alnumlist.append(v)

                                if not v.isalnum() and not v.isnumeric() and not v.isalpha():
                                    punctuations.append(v)
        
                                if check_int(v):
                                    numericintlist.append(v)

                                    
                    
                        cf = type(v)
                        if not v:
                            disnone.append('blanc')

                    print(dfx,"--------------------------------------------")

                    numALPHA = 0
                    numALNUM = 0
                    numNUMERIC_INT = 0

                    numPUNCT = 0

                    if len(idfloat) == len(dfx1):
                        print("All are floats.")
                        msg_float = "All are floats."
                    elif len(idfloat) != len(dfx1):
                        print(len(idfloat), " out of ", len(dfx1), " are floats.")
                        msg_float = str(len(idfloat)) + " out of " + str(len(dfx1)) +" are floats."
                    numFL = len(idfloat)
                            
                    if len(disnone) == len(dfx1):
                        print("All are None.")
                        msg_none = "All are None."
                    elif len(disnone) != len(dfx1):
                        print(len(disnone), " out of ", len(dfx1), " are None.")
                        msg_none = str(len(disnone)) + " out of " + str(len(dfx1)) + " are None."
                    numNONE = len(disnone)

                    if len(punctuations) == len(dfx1):
                        print("All are punctuations.")
                        msg_punctuations = "All are punctuations."
                    elif len(punctuations) != len(dfx1):
                        print(len(punctuations), " out of ", len(dfx1), " are punctuations.")
                        msg_punctuations = str(len(punctuations)) + " out of " + str(len(dfx1)) + " are punctuations."
                    numPUNCT = len(punctuations)
                        
                    if len(idisnan) != len(dfx1):
                        #print("Not all are floating numbers.")
                        msg_nan = str(len(idisnan)) + " out of " + str(len(dfx1)) + " are NaN."
                        print(len(idisnan), " out of ", len(dfx1), " are NaN.")
                    elif len(idisnan) == len(dfx1):
                        print("All are NaN.")
                        msg_nan = "All are NaN."
                    numNAN = len(idisnan)

                    if len(isfloater) != len(dfx1):
                        #print("Not all are floating numbers.")
                        msg_float2 = str(len(isfloater)) + " out of " + str(len(dfx1)) + " are floats."
                        print(len(isfloater), " out of ", len(dfx1), " are floats.")
                    elif len(isfloater) == len(dfx1):
                        print("All are floats.")
                        msg_float2 = "All are floats."
                    numFL2 = len(isfloater)

                    if len(dtlist) == len(dfx1):
                        print("All are DateTime.")
                        msg_dt = "All are DateTime."
                    elif len(dtlist) != len(dfx1):
                        print(len(dtlist), " out of ", len(dfx1), " are DateTime.")
                        msg_dt = str(len(dtlist)) + " out of " + str(len(dfx1)) + " are DateTime."
                    numDATETIME = len(dtlist)

                    if len(isinteger) == len(dfx1):
                        print("All are Integers.")
                        msg_int = "All are Integers."
                    elif len(isinteger) != len(dfx1):
                        print(len(isinteger), " out of ", len(dfx1), " are integers.")
                        msg_int = str(len(isinteger)) + " out of " + str(len(dfx1)) + " are integers."
                    numINT = len(isinteger)

                    if len(idisstr) == len(dfx1):
                        print("All are strings.")
                        msg_str = "All are strings."
                    elif len(idisstr) != len(dfx1):
                        print(len(idisstr), " out of ", len(dfx1), " are strings.")
                        msg_str = str(len(idisstr)) + " out of " + str(len(dfx1)) + " are strings."
                    numSTR = len(idisstr)

                    #-----------------------------------------
                    if idisstr:
                    
                            print(len(alphalist), " out of ", len(idisstr), " are alphas in the string.")
                            numALPHA = len(alphalist)
                            msg_str_alpha = str(len(alphalist)) + " out of " + str(len(idisstr)) + " are alphas in the string."

                            print(len(alnumlist), " out of ", len(idisstr), " are alphanums in the string.")
                            numALNUM = len(alnumlist)
                            msg_str_alnum = str(len(alnumlist)) + " out of " + str(len(idisstr)) + " are alphanums in the string."

                            print(len(numericintlist), " out of ", len(idisstr), " are ints or floats in the string.")
                            numNUMERIC_INT = len(numericintlist)
                            msg_str_int = str(len(numericintlist)) + " out of " + str(len(idisstr)) + " are ints or floats in the string."


                    idisnan = []
                    isfloater = []
                    idisstr = []
                    dtlist = []
                    disnone = []
                    idfloat = []
                    isinteger = []
                    alphalist = []
                    alnumlist = []
                    numericintlist = []
                    numericfloatlist = []

   

                    continuous = [dfx, numFL, numNONE, numNAN, numFL2, numDATETIME, numINT, numSTR, numALPHA, numALNUM, numNUMERIC_INT]

                    print(continuous)

                    ##-----------------datetime----------------
                    markindex = 0
                    if continuous[5] > int(len(dfxx)/2) and numNUMERIC_INT > 0 and continuous[5] > numNUMERIC_INT: 

                            for index, row in dfxx.iterrows():

                                    try:
                    
                                        dfdt = pd.to_datetime(row[dfx])
                                        markindex = index
                                        print(dfdt)
                                        
                                    except:
                                        
                                        dfxx.loc[index, dfx] = dfxx.loc[markindex, dfx]
                                        print(dfxx.loc[index, dfx])

                    #print(list(dfxx[dfx]))
                    
                    ##-----------------none str---------------------
                    markindex4 = 0
                    if continuous[7] != len(dfxx) and (numNONE != 0 or numNAN != 0) and numINT==0:

                        for index, row in dfxx.iterrows():

                            if row[dfx]:
                    
                                markindex4 = index
                                
                            elif not row[dfx]:
                                
                                dfxx.loc[index, dfx] = dfxx.loc[markindex4, dfx]
                                print(dfxx.loc[index, dfx])

                
                    ##-----------------float---------------------                            
                    markindex2 = 0
                    if continuous[1] > int(len(dfxx)/2) and continuous[4] != len(dfxx) and (numNONE != 0 or numNAN != 0):

                            for index, row in dfxx.iterrows():

                                    try:
                    
                                        dfdt = int(row[dfx])
                                        markindex2 = index
                                        print("try: ", row[dfx])
                                        
                                    except:

                                        t =list(dfxx[dfx])
                                        
                                        c = 0
                                        g = []
                                        for a in t:
                                            d = a >= c
                                            c = a
                                            g.append(d)

                                        kval = []
                                        for k in g:
                                            if k==True:
                                                kval.append(k)

                                        if len(g) == len(kval):

                                                # sequential
                                                if markindex2+1 < len(dfxx)-1:
                                                    if dfxx.loc[markindex2, dfx] > dfxx.loc[markindex2+1, dfx]:
                                                        dd = abs(dfxx.loc[markindex2, dfx] - dfxx.loc[markindex2+1, dfx])
                                                        dfxx.loc[index, dfx] = dfxx.loc[markindex2+1, dfx] - dd/2
                                                    elif dfxx.loc[markindex2, dfx] < dfxx.loc[markindex2+1, dfx]:
                                                        dd = abs(dfxx.loc[markindex2, dfx] - dfxx.loc[markindex2+1, dfx])
                                                        dfxx.loc[index, dfx] = dfxx.loc[markindex2+1, dfx] + dd/2

                                        else:

                                                # non-sequential
                                                dfxx.loc[index, dfx] = dfxx.loc[markindex2, dfx]
                                                print("except: ", dfxx.loc[index, dfx])

                    
                    markindex3 = 0
                    markdigit = 0
                    
                    if continuous[7] > int(len(dfxx)*0.75): # and (numNONE == 0 and numNAN == 0):
                            
                            
                            for index, row in dfxx.iterrows():

                                    digitT = 0
                                    letterT = 0
                                    digitE = 0
                                    letterE = 0

                                    if not row[dfx]:
                                      dfxx.loc[index, dfx] = dfxx.loc[index-1, dfx]
                                      

                                    try:

                                        for ch in str(int(float(row[dfx]))):
                                            if ch.isdigit():
                                                digitT=digitT+1
                                            elif ch.isalpha():
                                                letterT=letterT+1
                                              
                                            else:
                                              pass

                                        print("[try] digits vs letters", digitT, letterT)

                                        markdigit = digitT
                    
                                        dfdt = int(float(row[dfx]))
                                        markindex3 = index
                                        print("try: ", row[dfx])

                                     
       
                                    except:

                                        print("except: ", dfxx.loc[index, dfx])

                                        found = re.split('\W+', dfxx.loc[index, dfx])

                                        try:

                                            dfxx.loc[index, dfx] = str(float(found[0]))
                                        
                                            print("found[0]: ", found[0])

                                        except:

                                            pass

                                        pass
                                        
                            
                
                detectbtnflag = False
                processbtnflag = True

                colnames = []
                for col in df.columns:
                    if col != "Edit" and col != "Delete":
                        colnames.append(col)
                
                inputxtablename = tabname + "inputx"
                dfxx.to_sql(inputxtablename, con=engine, if_exists='replace', index=False)
                inputxtablename2 = tabname + "inputx2"
                dfxx.to_sql(inputxtablename2, con=engine, if_exists='replace', index=False)

                insertstatement = """ UPDATE xymarker SET dateandtime='%s',
                                                                              mark_x='%s',
                                                                              mark_x2='%s',
                                                                              username='%s'
                                                                              WHERE mark_xy='%s' """ % \
                                                                               (insertdatetime(), inputxtablename, inputxtablename2, "tommy", tabname)
                cur.execute(insertstatement)
                db.commit()

                for s in dfxx.columns:
                    print(dfxx[s].dtype)


                displayselected = False

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)

                toprocessbtnflag = True
                
              

                msg_float = ""
                msg_none = ""
                msg_nan = ""
                msg_float2 = ""
                msg_dt = ""
                msg_int = ""
                msg_str = ""
                msg_str_alpha = ""
                msg_str_alnum = ""
                msg_str_int = ""
                msg_str_float = ""
                msg_punctuations = ""

                #for i in dfxx:

                for dfx in list(dfxy.columns):

                    idfloat = []
                    idisnan = []
                    isfloater = []
                    idisstr = []
                    dtlist = []
                    disnone = []
                    idfloat = []
                    isinteger = []
                    alphalist = []
                    alnumlist = []
                    numericintlist = []
                    numericfloatlist = []
                    punctuations = []

                    dfx1 = list(dfxy[dfx])
                
                    for v in dfx1:
                        
                        cf = type(v)
                        if cf == float:
                            idfloat.append(v)

                        cf = type(v)
                        if isinstance(v, int):
                            isinteger.append(v)
                    
          
                   
                        cf = type(v)
                        if cf == float:
                            if math.isnan(v):
                                idisnan.append(math.nan)
                            elif math.isnan(v)==False:
                                isfloater.append(v)
                   

                        #for v in list(dfx):
                        cf = type(v)
                        if cf == str:
                            #idisstr.append(cf)
                            try:
                                dfdt = pd.to_datetime(v)
                                dtlist.append("datetime")
                            except:
                                idisstr.append(v)
                                #pass
                                if v.isalpha():
                                    alphalist.append(v)

                                #if re.match("^[A-Za-z0-9]*$", v):
                                if v.isalnum() and not v.isnumeric() and not v.isalpha():
                                    alnumlist.append(v)

                                if not v.isalnum() and not v.isnumeric() and not v.isalpha():
                                    punctuations.append(v)

                                if check_int(v):
                                    numericintlist.append(v)
                    
                        cf = type(v)
                        if not v:
                            disnone.append('blanc')

                    print(dfx,"--------------------------------------------")

                    numALPHA = 0
                    numALNUM = 0
                    numNUMERIC_INT = 0

                    numPUNCT = 0

                    if len(idfloat) == len(dfx1):
                        print("All are floats.")
                        msg_float = "All are floats."
                    elif len(idfloat) != len(dfx1):
                        print(len(idfloat), " out of ", len(dfx1), " are floats.")
                        msg_float = str(len(idfloat)) + " out of " + str(len(dfx1)) +" are floats."
                    numFL = len(idfloat)
                            
                    if len(disnone) == len(dfx1):
                        print("All are None.")
                        msg_none = "All are None."
                    elif len(disnone) != len(dfx1):
                        print(len(disnone), " out of ", len(dfx1), " are None.")
                        msg_none = str(len(disnone)) + " out of " + str(len(dfx1)) + " are None."
                    numNONE = len(disnone)

                    if len(punctuations) == len(dfx1):
                        print("All are punctuations.")
                        msg_punctuations = "All are punctuations."
                    elif len(punctuations) != len(dfx1):
                        print(len(punctuations), " out of ", len(dfx1), " are punctuations.")
                        msg_punctuations = str(len(punctuations)) + " out of " + str(len(dfx1)) + " are punctuations."
                    numPUNCT = len(punctuations)
                        
                    if len(idisnan) != len(dfx1):
                        #print("Not all are floating numbers.")
                        msg_nan = str(len(idisnan)) + " out of " + str(len(dfx1)) + " are NaN."
                        print(len(idisnan), " out of ", len(dfx1), " are NaN.")
                    elif len(idisnan) == len(dfx1):
                        print("All are NaN.")
                        msg_nan = "All are NaN."
                    numNAN = len(idisnan)

                    if len(isfloater) != len(dfx1):
                        #print("Not all are floating numbers.")
                        msg_float2 = str(len(isfloater)) + " out of " + str(len(dfx1)) + " are floats."
                        print(len(isfloater), " out of ", len(dfx1), " are floats.")
                    elif len(isfloater) == len(dfx1):
                        print("All are floats.")
                        msg_float2 = "All are floats."
                    numFL2 = len(isfloater)

                    if len(dtlist) == len(dfx1):
                        print("All are DateTime.")
                        msg_dt = "All are DateTime."
                    elif len(dtlist) != len(dfx1):
                        print(len(dtlist), " out of ", len(dfx1), " are DateTime.")
                        msg_dt = str(len(dtlist)) + " out of " + str(len(dfx1)) + " are DateTime."
                    numDATETIME = len(dtlist)

                    if len(isinteger) == len(dfx1):
                        print("All are Integers.")
                        msg_int = "All are Integers."
                    elif len(isinteger) != len(dfx1):
                        print(len(isinteger), " out of ", len(dfx1), " are integers.")
                        msg_int = str(len(isinteger)) + " out of " + str(len(dfx1)) + " are integers."
                    numINT = len(isinteger)

                    if len(idisstr) == len(dfx1):
                        print("All are strings.")
                        msg_str = "All are strings."
                    elif len(idisstr) != len(dfx1):
                        print(len(idisstr), " out of ", len(dfx1), " are strings.")
                        msg_str = str(len(idisstr)) + " out of " + str(len(dfx1)) + " are strings."
                    numSTR = len(idisstr)

                    #-----------------------------------------
                    if idisstr:
                    
                           
                            print(len(alphalist), " out of ", len(idisstr), " are alphas in the string.")
                            numALPHA = len(alphalist)
                            msg_str_alpha = str(len(alphalist)) + " out of " + str(len(idisstr)) + " are alphas in the string."

                          
                            print(len(alnumlist), " out of ", len(idisstr), " are alphanums in the string.")
                            numALNUM = len(alnumlist)
                            msg_str_alnum = str(len(alnumlist)) + " out of " + str(len(idisstr)) + " are alphanums in the string."

                   
                            print(len(numericintlist), " out of ", len(idisstr), " are ints or floats in the string.")
                            numNUMERIC_INT = len(numericintlist)
                            msg_str_int = str(len(numericintlist)) + " out of " + str(len(idisstr)) + " are ints or floats in the string."

           

                    idisnan = []
                    isfloater = []
                    idisstr = []
                    dtlist = []
                    disnone = []
                    idfloat = []
                    isinteger = []
                    alphalist = []
                    alnumlist = []
                    numericintlist = []
                    numericfloatlist = []

        

                    continuous = [dfx, numFL, numNONE, numNAN, numFL2, numDATETIME, numINT, numSTR, numALPHA, numALNUM, numNUMERIC_INT]

                    print(continuous)

                    ##-----------------datetime----------------
                    markindex = 0
                    if continuous[5] > int(len(dfxy)/2) and numNUMERIC_INT > 0 and continuous[5] > numNUMERIC_INT: 

                            for index, row in dfxy.iterrows():

                                    try:
                    
                                        dfdt = pd.to_datetime(row[dfx])
                                        markindex = index
                                        print(dfdt)
                                        
                                    except:
                                        
                                        dfxy.loc[index, dfx] = dfxy.loc[markindex, dfx]
                                        print(dfxy.loc[index, dfx])

                    #print(list(dfxy[dfx]))
                    
                    ##-----------------none str---------------------
                    markindex4 = 0
                    if continuous[7] != len(dfxy) and (numNONE != 0 or numNAN != 0) and numINT==0:

                        for index, row in dfxy.iterrows():

                            if row[dfx]:
                    
                                markindex4 = index
                                
                            elif not row[dfx]:
                                
                                dfxy.loc[index, dfx] = dfxy.loc[markindex4, dfx]
                                print(dfxy.loc[index, dfx])

                    ##-----------------float---------------------                            
                    markindex2 = 0
                    if continuous[1] > int(len(dfxy)/2) and continuous[4] != len(dfxy) and (numNONE != 0 or numNAN != 0):

                            for index, row in dfxy.iterrows():

                                    try:
                    
                                        dfdt = int(row[dfx])
                                        markindex2 = index
                                        print("try: ", row[dfx])
                                        
                                    except:

                                        t =list(dfxy[dfx])
                                        
                                        c = 0
                                        g = []
                                        for a in t:
                                            d = a >= c
                                            c = a
                                            g.append(d)

                                        kval = []
                                        for k in g:
                                            if k==True:
                                                kval.append(k)

                                        if len(g) == len(kval):

                                                # sequential
                                                if markindex2+1 < len(dfxy)-1:
                                                    if dfxy.loc[markindex2, dfx] > dfxy.loc[markindex2+1, dfx]:
                                                        dd = abs(dfxy.loc[markindex2, dfx] - dfxy.loc[markindex2+1, dfx])
                                                        dfxy.loc[index, dfx] = dfxy.loc[markindex2+1, dfx] - dd/2
                                                    elif dfxy.loc[markindex2, dfx] < dfxy.loc[markindex2+1, dfx]:
                                                        dd = abs(dfxy.loc[markindex2, dfx] - dfxy.loc[markindex2+1, dfx])
                                                        dfxy.loc[index, dfx] = dfxy.loc[markindex2+1, dfx] + dd/2

                                        else:

                                                # non-sequential
                                                dfxy.loc[index, dfx] = dfxy.loc[markindex2, dfx]
                                                print("except: ", dfxy.loc[index, dfx])

                                        
                                    
                    
                    markindex3 = 0
                    markdigit = 0
                    
                    if continuous[7] > int(len(dfxy)*0.75) and (numNONE == 0 and numNAN == 0):
                                        
                            
                            for index, row in dfxy.iterrows():

                                    digitT = 0
                                    letterT = 0
                                    digitE = 0
                                    letterE = 0

                                    try:

                                        for ch in str(int(float(row[dfx]))):
                                            
                                            if ch.isdigit():
                                              digitT=digitT+1
                                            elif ch.isalpha():
                                              letterT=letterT+1
                                              
                                            else:
                                              pass

                                        print("[try] digits vs letters", digitT, letterT)

                                        markdigit = digitT
                    
                                        dfdt = int(float(row[dfx]))
                                        markindex3 = index
                                        print("try: ", row[dfx])

                                    
       
                                    except:

                                        print("except: ", dfxy.loc[index, dfx])

                                        found = re.split('\W+', dfxy.loc[index, dfx])

                                        try:

                                            dfxy.loc[index, dfx] = str(float(found[0]))
                                        
                                            print("found[0]: ", found[0])

                                        except:

                                            pass

                                        pass

                    
                                   
                
                detectbtnflag = False
                processbtnflag = True

                colnames = []
                for col in df.columns:
                    if col != "Edit" and col != "Delete":
                        colnames.append(col)
                
                labelytablename = tabname + "labely"
                dfxy.to_sql(labelytablename, con=engine, if_exists='replace', index=False)
                labelytablename2 = tabname + "labely2"
                dfxy.to_sql(labelytablename2, con=engine, if_exists='replace', index=False)

                insertstatement = """ UPDATE xymarker SET dateandtime='%s',
                                                                              mark_y='%s',
                                                                              mark_y2='%s',
                                                                              username='%s'
                                                                              WHERE mark_xy='%s' """ % \
                                                                              (insertdatetime(), labelytablename, labelytablename2, "tommy", tabname)
                
                cur.execute(insertstatement)
                db.commit()

                designatedtable = [inputxtablename, labelytablename]

                for s in dfxy.columns:
                    print(dfxy[s].dtype)


                displayselected = False

                
                return render_template("ingest.html", df_tableX = [dfxx.to_html(classes='data table-striped', escape=False)], df_tableY = [dfxy.to_html(classes='data table-striped', escape=False)],
                                                                     detectbtnflag=detectbtnflag, colnames=colnames, displayselected=displayselected, resX=resX, resY=resY, tablelist=myinput, sheetn=tabname,
                                                                     toprocessbtnflag=toprocessbtnflag, designatedtable=designatedtable, selectedcolumnsX=selectedcolumnsX, selectedcolumnsY=selectedcolumnsY)
                
            

dfreplicateX = pd.DataFrame()
dfreplicateY = pd.DataFrame()


@app.route("/selectxy", methods=['POST'])
def selectxy():

        if request.method == 'POST':
            
            if request.form.get("action_get") == "Confirm":
                
                selectedcolumnsX = request.form.getlist('picklistX')
                selectedcolumnsY = request.form.getlist('picklistY')
           
                            
                tabname = ""
                
                tabnam = request.form['seltable']
                print("table name select 1: ", tabnam)
                if tabnam:
                    tabname = tabnam
                    
                tabnam = request.form['select_table']
                print("table name select 2: ", tabnam)
                if tabnam:
                    tabname = tabnam

                myinput = []
                selecte = request.form.get('tnames')
                if selecte:
                    myinput = ast.literal_eval(selecte)
                    print("myinput: ", myinput)
                
                select = request.form.get('plaintablelist')
                if select:
                    myinput = ast.literal_eval(select)
                    print("myinput: ", myinput)

                #if not tabname:
                    #tabname = myinput[0]
                    
                querytable = "SELECT * FROM {}".format(tabname)
                df = pd.read_sql_query(querytable, con=engine)
                df_full = df
                columnames = df_full.columns
                
                dfX = df[selectedcolumnsX]
                dfY = df[selectedcolumnsY]
                
                colnamesX = dfX.columns
                colnamesY = dfY.columns

                columnames = []
                for col in df_full.columns:
                    #print(col)
                    if col != "Edit" and col != "Delete":
                        columnames.append(col)

                detectbtnflag = True
                displayselected = False


        

                return render_template("ingest.html", df_tableX = [dfX.to_html(classes='data table-striped', escape=False)], df_tableY = [dfY.to_html(classes='data table-striped', escape=False)],
                                       Xcolnames=colnamesX, Ycolnames=colnamesY, colnames=columnames, detectbtnflag=detectbtnflag, selectedcolumnsX=selectedcolumnsX,
                                       displayselected=displayselected, selectedcolumnsY=selectedcolumnsY, tablelist=myinput, sheetn=tabname)
                #return render_template("ingest.html", df_table = [df.to_html(classes='data table-striped', escape=False)], colnames=colnames)
        else:
            return render_template("ingest.html")
        



@app.route("/machinelearning")
def machinelearning():
        return render_template("machinelearning.html")

@app.route("/outputvisualization")
def outputvisualization():
        return render_template("outputvisualization.html")

@app.route("/preprocessing", methods=['GET', 'POST'])
def preprocessing():

    
    if request.method == 'POST':

        #tablename = request.form['tablename']
        
        if request.form.get("showtable") == "Show Table":

                mod = request.form['transformmodel']

                testsize = request.form['testsize']

                markbinfile = request.form['binfilename']

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(', ')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], x_and_y_list[1])


                
                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)
             

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                #for c in selectedcolumnsX:
                #    print(c.index)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]



                return render_template("preprocessing.html", df_tableX = [df_x.to_html(classes='data table-striped', escape=False)], df_tableY = [df_y.to_html(classes='data table-striped', escape=False)],
                                                                             listoftables = [x_and_y], tablename=x_and_y, columnames_x=columnames_x, columnames_y=columnames_y, testsize=testsize, transformmodel=mod,
                                                                             xlabel="[Input X]", ylabel="[Target/Label Y]", labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test",
                                                                             markbinfile=markbinfile)
        
        elif request.form.get("processx") == "Process this Column":

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer()

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']
                
                testsize = request.form['testsize']

                markbinfile = request.form['binfilename']
                
                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(', ')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], x_and_y_list[1])

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
               

                dfxxnew = df_x[columnames_x]
                dfxynew = df_y[columnames_y]

                md = np.empty([2, 2])

                for k in dfxxnew.columns:
                    if selectedcolumnx == k:
                        from sklearn import preprocessing
                        if mod == "LabelEncoder":

                            le = preprocessing.LabelEncoder()
                            
                            lefit = le.fit(dfxxnew[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefit, kf, compress=True)
                            #pickle.dump(le, open(kf, 'wb'))
                            
                            md = lefit.transform(dfxxnew[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            dfxxnew[k] = singleColumn
                            print("dfxxnew: labelencoder snap ", dfxxnew)
                            dfxxnew.to_sql(x_and_y_list[0], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "OneHotEncoder":

                            ohe = preprocessing.OneHotEncoder()

                            mdx = ohe.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = ohe.transform(dfxxnew[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #dfxxnew[k] = singleColumn
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                                
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxxnew = pd.concat([dfxxnew, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxxnew = dfxxnew.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxxnew: onehotencoder snap ", dfxxnew)
                            dfxxnew.to_sql(x_and_y_list[0], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "LabelBinarizer":

                            lb = preprocessing.LabelBinarizer()

                            mdx = lb.fit(dfxxnew[k].to_numpy())
                            md = lb.transform(dfxxnew[k].to_numpy())
                            
                            #singleColumn = pd.DataFrame(md)
                            #dfxxnew[k] = singleColumn
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                                
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            dfxxnew = pd.concat([dfxxnew, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxxnew = dfxxnew.drop(columns=k, axis=1)
                            print("dfxxnew: labelbinarizer snap ", dfxxnew)
                            dfxxnew.to_sql(x_and_y_list[0], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "MinMaxScaler":

                            mm = preprocessing.MinMaxScaler()

                            mdx = mm.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = mm.transform(dfxxnew[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)

                            singleColumn = pd.DataFrame(md)
                            dfxxnew[k] = singleColumn
                            print("dfxxnew: minmax snap ", dfxxnew)
                            dfxxnew.to_sql(x_and_y_list[0], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "StandardScaler":

                            ss = preprocessing.StandardScaler()

                            mdx = ss.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = ss.transform(dfxxnew[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'StandardScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxxnew[k] = singleColumn
                            print("dfxxnew: standardscaler snap ", dfxxnew)
                            dfxxnew.to_sql(x_and_y_list[0], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "KBinsDiscretizer":

                            kbd = preprocessing.KBinsDiscretizer()
                            
                            mdx = kbd.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = kbd.transform(dfxxnew[k].to_numpy().reshape(-1, 1))
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                                
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxxnew = pd.concat([dfxxnew, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxxnew = dfxxnew.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxxnew: kbinsdiscretizer snap ", dfxxnew)
                            dfxxnew.to_sql(x_and_y_list[0], con=engine, if_exists='replace', index=False)


                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, df_tableX = [dfxxnew.to_html(classes='data table-striped', escape=False)], df_tableY = [dfxynew.to_html(classes='data table-striped', escape=False)],#xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                 labelstrXtrain="X", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny, labelstrYtrain="y",
                                                                                 markbinfile=markbinfile) # labelstrXtest="X Test", labelstrYtest="y Test")
        
        elif request.form.get("processy") == "Process this Column":

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer()

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']
                
                testsize = request.form['testsize']
                
                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(', ')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], x_and_y_list[1])

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])

                markbinfile = ""



                for k in dfxy.columns:
                    if selectedcolumny == k:
                        from sklearn import preprocessing
                        if mod == "LabelEncoder":

                            
                            le = preprocessing.LabelEncoder()

                            lefit = le.fit(dfxy[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            #print("alpha f: ", alphanumeric)
                            #print("LabelEncoder"+get_datetime()+alphanumeric)
                            
                            markbinfile= 'LabelEncoder'+get_datetime()+alphanumeric+'.bin'
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(lefit, kf, compress=True)
                            #pickle.dump(md, open(kf, 'wb'))

                            md = lefit.transform(dfxy[k].to_numpy())
                            singleColumn = pd.DataFrame(md)
                            dfxy[k] = singleColumn
                            print("dfxy: labelencoder snap ", dfxy)
                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "OneHotEncoder":

                            ohe = preprocessing.OneHotEncoder()
##                            md = ohe.fit_transform(dfxy[k].to_numpy().reshape(-1, 1))
##                            singleColumn = pd.DataFrame(md)
##                            dfxy[k] = singleColumn
##                            print("dfxy: onehotencoder snap ", dfxy)
##                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
                            mdx = ohe.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = ohe.transform(dfxy[k].to_numpy().reshape(-1, 1))
                            
                            #singleColumn = pd.DataFrame(md)
                            #dfxxnew[k] = singleColumn
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile= 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxy = pd.concat([dfxy, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxy = dfxy.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxy: onehotencoder snap ", dfxy)
                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "LabelBinarizer":

                            lb = preprocessing.LabelBinarizer()
##                            md = lb.fit_transform(dfxy[k].to_numpy())
##                            singleColumn = pd.DataFrame(md)
##                            dfxy[k] = singleColumn
##                            print("dfxy: labelbinarizer snap ", dfxy)
##                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
                            mdx = lb.fit(dfxy[k].to_numpy())
                            md = lb.transform(dfxy[k].to_numpy())
                            #singleColumn = pd.DataFrame(md)
                            #dfxxnew[k] = singleColumn
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile= 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            dfxy = pd.concat([dfxy, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxy = dfxy.drop(columns=k, axis=1)
                            print("dfxy: labelbinarizer snap ", dfxy)
                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "MinMaxScaler":

                            mm = preprocessing.MinMaxScaler()

                            mdx = mm.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = mm.transform(dfxy[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile= 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                                    
                            singleColumn = pd.DataFrame(md)
                            dfxy[k] = singleColumn
                            print("dfxy: minmax snap ", dfxy)
                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "StandardScaler":

                            ss = preprocessing.StandardScaler()

                            mdx = ss.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = ss.transform(dfxy[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile= 'StandardScaler'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxy[k] = singleColumn
                            print("dfxy: standardscaler snap ", dfxy)
                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
                            
                        elif mod == "KBinsDiscretizer":

                            kbd = preprocessing.KBinsDiscretizer()

                            mdx = kbd.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = kbd.transform(dfxy[k].to_numpy().reshape(-1, 1))
##                            singleColumn = pd.DataFrame(md)
##                            dfxy[k] = singleColumn
##                            print("dfxy: kbinsdiscretizer snap ", dfxy)
##                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile= 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxy = pd.concat([dfxy, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxy = dfxy.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxy: kbinsdiscretizer snap ", dfxy)
                            dfxy.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)

                #labelwithcol = "[Target/Label Y] Col: " + selectedcolumny
                
##            return render_template("preprocessing.html", listoftables=listoftables, tablename=tablename, table1 = [df.to_html(classes = 'data')])
                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, ylabel="[Target/Label Y]", xlabel="[INPUT X]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, df_tableX = [dfxx.to_html(classes='data table-striped', escape=False)], df_tableY = [dfxy.to_html(classes='data table-striped', escape=False)],#xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                 labelstrXtrain="X", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny, labelstrYtrain="y",
                                                                                 markbinfile=markbinfile) #, labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

        elif request.form.get("resetx") == "RESET":

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer()

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']

                markbinfile = request.form['binfilename']
                
                testsize = request.form['testsize']
                
                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(', ')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], x_and_y_list[1])

              

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0]+'2')
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1]+'2')
                df_y = pd.read_sql_query(querytabley, con=engine)

               
                df_x.to_sql(x_and_y_list[0], con=engine, if_exists='replace', index=False)
                df_y.to_sql(x_and_y_list[1], con=engine, if_exists='replace', index=False)
            
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[INPUT X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, df_tableX = [dfxx.to_html(classes='data table-striped', escape=False)], df_tableY = [dfxy.to_html(classes='data table-striped', escape=False)], #xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                 labelstrXtrain="X", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny, markbinfile=markbinfile) #, labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

                
        
        elif request.form.get("splitdata") == "Split Data":

                testsize = request.form['testsize']
                #tablename = request.form['tablename']
                mod = request.form['transformmodel']

                markbinfile = request.form['binfilename']

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(', ')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], x_and_y_list[1])


                
                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)
                #df_full = df
                #columnames = df.columns
                
                #dfX = df[selectedcolumnsX]
                #dfY = df[selectedcolumnsY]
                
                #colnamesX = dfX.columns
                #colnamesY = dfY.columns

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                #for c in selectedcolumnsX:
                #    print(c.index)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                
               
                X = dfxx
                print("X shape: ", X, X.shape)
                
                #y = dfxy[int(len(dfxy)/2)+1:len(dfxy)]
                y = dfxy
                print("y shape: ", y, y.shape)
##                while X.shape[0] != y.shape[0]:
##                    s = int(len(dfxx)/2)
##                    s = s - 1
##                    X = dfxx[0:s]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=6)
                print("Split shapes: ", X_train.shape,X_test.shape,y_train.shape,y_test.shape)
                print("Type X, y:", type(X_test), type(y_test))
                xtraindf = pd.DataFrame(X_train)
                print("xtraindf type: ", type(xtraindf))
                ytraindf = pd.DataFrame(y_train)
                print("ytraindf type: ", type(ytraindf))
                xtestdf = pd.DataFrame(X_test)
                ytestdf = pd.DataFrame(y_test)

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                xtraindf.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                ytraindf.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                xtestdf.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                ytestdf.to_sql(ytestname, con=engine, if_exists='replace', index=False)



                if not markbinfile:
                    markbinfile = "NT"

                insertstatement = """ UPDATE xymarker SET dateandtime='%s',
                                                                              mark_x_train='%s',
                                                                              mark_x_test='%s',
                                                                              mark_y_train='%s',
                                                                              mark_y_test='%s',
                                                                              username='%s',
                                                                              transform_y='%s'
                                                                              WHERE mark_xy='%s' """ % \
                                                                               (insertdatetime(), xtrainname, xtestname, ytrainname, ytestname, "tommy", markbinfile, x_and_y_list[0].replace('inputx', '')) #tabname rm substing
                cur.execute(insertstatement)
                db.commit()

                xtrainname2 = x_and_y_list[0] + "xtr2"
                ytrainname2 = x_and_y_list[1] + "ytr2"
                xtestname2 = x_and_y_list[0] + "xte2"
                ytestname2 = x_and_y_list[1] + "yte2"

                xtraindf.to_sql(xtrainname2, con=engine, if_exists='replace', index=False)
                ytraindf.to_sql(ytrainname2, con=engine, if_exists='replace', index=False)
                xtestdf.to_sql(xtestname2, con=engine, if_exists='replace', index=False)
                ytestdf.to_sql(ytestname2, con=engine, if_exists='replace', index=False)
                
                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, columnames_x=columnames_x, columnames_y=columnames_y, transformmodel=mod,
                                                                      xtrain = [xtraindf.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                      ytrain = [ytraindf.to_html(classes = 'data')], labelstrYtrain="y Train", markbinfile=markbinfile,
                                                                      xtest = [xtestdf.to_html(classes = 'data')], labelstrXtest="X Test",
                                                                      ytest = [ytestdf.to_html(classes = 'data')], labelstrYtest="y Test")


        elif request.form.get("resetsplit") == "RESET":

                #from sklearn.preprocessing import KBinsDiscretizer

                #kbd = KBinsDiscretizer()

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']
                
                testsize = request.form['testsize']
                
                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(', ')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], x_and_y_list[1])

        

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0]+'2')
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1]+'2')
                df_y = pd.read_sql_query(querytabley, con=engine)
           
                xtrainname = x_and_y_list[0] + "xtr2"
                ytrainname = x_and_y_list[1] + "ytr2"
                xtestname = x_and_y_list[0] + "xte2"
                ytestname = x_and_y_list[1] + "yte2"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                xtrainname0 = x_and_y_list[0] + "xtr"
                ytrainname0 = x_and_y_list[1] + "ytr"
                xtestname0 = x_and_y_list[0] + "xte"
                ytestname0 = x_and_y_list[1] + "yte"

                df_xtr.to_sql(xtrainname0, con=engine, if_exists='replace', index=False)
                df_ytr.to_sql(ytrainname0, con=engine, if_exists='replace', index=False)
                df_xte.to_sql(xtestname0, con=engine, if_exists='replace', index=False)
                df_yte.to_sql(ytestname0, con=engine, if_exists='replace', index=False)
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[INPUT X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #df_tableX = [dfxx.to_html(classes='data table-striped', escape=False)], df_tableY = [dfxy.to_html(classes='data table-striped', escape=False)], #xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train",
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test",
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test") #, labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

        
        elif request.form.get("processxtrain") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumnx_xtrain = request.form['selectedcolumnx_xtrain']
                
                #queryalltables = """SELECT * FROM {}""".format(tablename)
                #df = pd.read_sql_query(queryalltables, con=postgres_db, index_col=None)

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_xtr.columns:
                    if selectedcolumnx_xtrain == k:
                        if modsp == "LabelEncoder":
                            lefit = le.fit(df_xtr[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefit, kf, compress=True)
                            #pickle.dump(md, open(kf, 'wb'))
                            md = lefit.transform(df_xtr[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_xtr[k] = singleColumn
                            print("df_xtr: labelencoder snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #df_xtr[k] = singleColumn
                            #print("df_xtr: onehotencoder snap ", df_xtr)
                            #df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xtr = pd.concat([df_xtr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xtr = df_xtr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xtr: onehotencoder snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.fit_transform(df_xtr[k].to_numpy())
                            #singleColumn = pd.DataFrame(md)
                            #df_xtr[k] = singleColumn
                            #print("df_xtr: labelbinarizer snap ", df_xtr)
                            #df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_xtr = pd.concat([df_xtr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xtr = df_xtr.drop(columns=k, axis=1)
                            print("df_xtr: labelbinarizer snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_xtr[k] = singleColumn
                            print("df_xtr: minmax snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'StandardScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_xtr[k] = singleColumn
                            print("df_xtr: standardscaler snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))
                            dump(md, 'KBinsDiscretizer.bin', compress=True)
                            #singleColumn = pd.DataFrame(md)
                            #df_xtr[k] = singleColumn
                            #print("df_xtr: kbinsdiscretizer snap ", df_xtr)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xtr = pd.concat([df_xtr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xtr = df_xtr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xtr: kbinsdiscretizer snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train",
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test",
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test")
                                                                                 

        elif request.form.get("processxtest") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumnx_xtest = request.form['selectedcolumnx_xtest']
                
                #queryalltables = """SELECT * FROM {}""".format(tablename)
                #df = pd.read_sql_query(queryalltables, con=postgres_db, index_col=None)

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_xte.columns:
                    if selectedcolumnx_xtest == k:
                        if modsp == "LabelEncoder":
                            lefit = le.fit(df_xte[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefit, kf, compress=True)
                            #pickle.dump(md, open(kf, 'wb'))
                            md = lefit.transform(df_xte[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_xte[k] = singleColumn
                            print("df_xte: labelencoder snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #df_xte[k] = singleColumn
                            #print("df_xte: onehotencoder snap ", df_xte)
                            #df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xte = pd.concat([df_xte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xte = df_xte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xte: onehotencoder snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.transform(df_xte[k].to_numpy())
                            #singleColumn = pd.DataFrame(md)
                            #df_xte[k] = singleColumn
                            #print("df_xte: labelbinarizer snap ", df_xte)
                            #df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_xte = pd.concat([df_xte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xte = df_xte.drop(columns=k, axis=1)
                            print("df_xte: labelbinarizer snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_xte[k] = singleColumn
                            print("df_xte: minmax snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'StandardScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_xte[k] = singleColumn
                            print("df_xte: standardscaler snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #df_xte[k] = singleColumn
                            #print("df_xte: kbinsdiscretizer snap ", df_xte)
                            #df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xte = pd.concat([df_xte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xte = df_xte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xte: kbinsdiscretizer snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train",
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test",
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test")

        elif request.form.get("processytrain") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumny_ytrain = request.form['selectedcolumny_ytrain']
                
                #queryalltables = """SELECT * FROM {}""".format(tablename)
                #df = pd.read_sql_query(queryalltables, con=postgres_db, index_col=None)

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_ytr.columns:
                    if selectedcolumny_ytrain == k:
                        if modsp == "LabelEncoder":
                            lefit = le.fit(df_ytr[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefit, kf, compress=True)
                            #pickle.dump(lefit, open(kf, 'wb'))

                            md = lefit.transform(df_ytr[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_ytr[k] = singleColumn
                            print("df_ytr: labelencoder snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #df_ytr[k] = singleColumn
                            #print("df_ytr: onehotencoder snap ", df_ytr)
                            #df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_ytr = pd.concat([df_ytr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_ytr = df_ytr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_ytr: onehotencoder snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.fit_transform(df_ytr[k].to_numpy())
                            #singleColumn = pd.DataFrame(md)
                            #df_ytr[k] = singleColumn
                            #print("df_ytr: labelbinarizer snap ", df_ytr)
                            #df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_ytr = pd.concat([df_ytr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_ytr = df_ytr.drop(columns=k, axis=1)
                            print("df_ytr: labelbinarizer snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_ytr[k] = singleColumn
                            print("df_ytr: minmax snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'StandardScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_ytr[k] = singleColumn
                            print("df_ytr: standardscaler snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #df_ytr[k] = singleColumn
                            #print("df_ytr: kbinsdiscretizer snap ", df_ytr)
                            #df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_ytr = pd.concat([df_ytr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_ytr = df_ytr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_ytr: kbinsdiscretizer snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train",
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test",
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test")

        elif request.form.get("processytest") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumny_ytest = request.form['selectedcolumny_ytest']
                
                #queryalltables = """SELECT * FROM {}""".format(tablename)
                #df = pd.read_sql_query(queryalltables, con=postgres_db, index_col=None)

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_yte.columns:
                    if selectedcolumny_ytest == k:
                        if modsp == "LabelEncoder":
                            lefit = le.fit(df_yte[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefit, kf, compress=True)
                            #pickle.dump(lefit, open(kf, 'wb'))

                            md = lefit.transform(df_yte[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_yte[k] = singleColumn
                            print("df_yte: labelencoder snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.transform(df_yte[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #df_yte[k] = singleColumn
                            #print("df_yte: onehotencoder snap ", df_yte)
                            #df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_yte = pd.concat([df_yte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_yte = df_yte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_yte: onehotencoder snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.transform(df_yte[k].to_numpy())
                            
                            #singleColumn = pd.DataFrame(md)
                            #df_yte[k] = singleColumn
                            #print("df_yte: labelbinarizer snap ", df_yte)
                            #df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_yte = pd.concat([df_yte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_yte = df_yte.drop(columns=k, axis=1)
                            print("df_yte: labelbinarizer snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.transform(df_yte[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_yte[k] = singleColumn
                            print("df_yte: minmax snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.transform(df_yte[k].to_numpy().reshape(-1, 1))

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'StandardScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            df_yte[k] = singleColumn
                            print("df_yte: standardscaler snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.transform(df_yte[k].to_numpy().reshape(-1, 1))
                            #singleColumn = pd.DataFrame(md)
                            #df_yte[k] = singleColumn
                            #print("df_yte: kbinsdiscretizer snap ", df_yte)
                            #df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin')
                            dump(md, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_yte = pd.concat([df_yte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_yte = df_yte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_yte: kbinsdiscretizer snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train",
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test",
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test")

                
            
        elif request.form.get("fitdata") == "Fit Data":
                
                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                mod = request.form['transformmodel']
                #queryalltables = """SELECT * FROM {}""".format(tablename)
                #df = pd.read_sql_query(queryalltables, con=postgres_db, index_col=None)

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                # ----------------------------------------------------------
        
                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]
             
                X = dfxx[0:int(len(dfxx)/2)]
                print(X, X.shape)
                y = dfxy[int(len(dfxy)/2)+1:len(dfxy)]
                print(y, y.shape)
##                while X.shape[0] != y.shape[0]:
##                    s = int(len(dfxx)/2)
##                    s = s - 1
##                    X = dfxx[0:s]
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=6)

                # ----------------------------------------------------------

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                xtraindf = pd.DataFrame(X_train)
                ytraindf = pd.DataFrame(y_train)
                xtestdf = pd.DataFrame(X_test)
                ytestdf = pd.DataFrame(y_test)

                xtraindf.columns = columnames_x
                ytraindf.columns = columnames_y
                xtestdf.columns = columnames_x
                ytestdf.columns = columnames_y
                
                if mod=="MinMaxScaler":
                    mm.fit(X_train)
                    
                    model_min = mm.data_min_
                    model_max = mm.data_max_
                    model_range = mm.data_range_
                    model_feat = mm.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_min, model_max, model_range, model_feat]
                    attrnames = ['Min', 'Max', 'Range', 'Features']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    mm.fit(X_train)
                    mm_xtrain = mm.transform(X_train)
                    print("mm_xtrain: ", mm_xtrain)
                    mm_xtest = mm.transform(X_test)
                    print("mm_xtest: ", mm_xtest)

                    #----------------------------

                    mm.fit(y_train)
                
                    model_min = mm.data_min_
                    model_max = mm.data_max_
                    model_range = mm.data_range_
                    model_feat = mm.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_min, model_max, model_range, model_feat]
                    attrnames = ['Min', 'Max', 'Range', 'Features']

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)

                    # ----------------------------
                    
                    xtraindf = pd.DataFrame(X_train)
                    ytraindf = pd.DataFrame(y_train)
                    xtestdf = pd.DataFrame(X_test)
                    ytestdf = pd.DataFrame(y_test)
                                    
                    mm.fit(y_train)
                    mm_ytrain = mm.transform(y_train)
                    print("mm_ytrain: ", mm_ytrain)
                    mm_ytest = mm.transform(y_test)
                    print("mm_ytest: ", mm_xtest)

                    return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

                                                                  

                if mod=="StandardScaler":
                
                    ss.fit(X_train)
                    
                    model_mean = ss.mean_
                    model_var = ss.var_
                    model_feat = ss.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_mean, model_var, model_feat]
                    attrnames = ['Mean', 'Variance', 'Features']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    ss.fit(X_train)
                    ss_xtrain = ss.transform(X_train)
                    print("ss_xtrain: ", ss_xtrain)
                    ss_xtest = ss.transform(X_test)
                    print("ss_xtest: ", ss_xtest)

                    #----------------------------

                    ss.fit(y_train)
                
                    model_mean = ss.mean_
                    model_var = ss.var_
                    model_feat = ss.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_mean, model_var, model_feat]
                    attrnames = ['Mean', 'Variance', 'Features']

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)

                    
                    
                    ss.fit(y_train)
                    ss_ytrain = ss.transform(y_train)
                    print("ss_ytrain: ", ss_ytrain)
                    ss_ytest = ss.transform(y_test)
                    print("ss_ytest: ", ss_xtest)

                
                    return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=ss_xtrain, xtest=ss_xtest, ytrain=ss_ytrain, ytest=ss_ytest,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

                if mod=="OneHotEncoder":

                    ohe.fit(X_train)
                    
                    model_cat = ohe.categories_
                   
                    modelattr = [model_cat]
                    attrnames = ['Categories']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    #ohe.fit(X_train)
                    ohe_xtrain = ohe.transform(X_train)
                    print("ohe_xtrain: ", ohe_xtrain)

                    ohe.fit(X_test)
                    ohe_xtest = ohe.transform(X_test)
                    print("ohe_xtest: ", ohe_xtest)

                    #----------------------------

                    ohe.fit(y_train)
                
                    model_cat = ohe.categories_
                   
                    modelattr = [model_cat]
                    attrnames = ['Categories']

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)

                    #ohe.fit(y_train)
                    ohe_ytrain = ohe.transform(y_train)
                    print("ohe_ytrain: ", ohe_ytrain)

                    ohe.fit(y_test)
                    ohe_ytest = ohe.transform(y_test)
                    print("ohe_ytest: ", ohe_xtest)

                
                    return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=ohe_xtrain, xtest=ohe_xtest, ytrain=ohe_ytrain, ytest=ohe_ytest,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

                if mod=="LabelEncoder":

                    testsize = request.form['testsize']
                    #tablename = request.form['tablename']

                    x_and_y = request.form['tablename']
                    print("PTNS: ", x_and_y, type(x_and_y))
                    #x_and_y_list = x_and_y.strip('][').split(', ')
                    x_and_y_list = ast.literal_eval(x_and_y)
                    print("to list: ", x_and_y_list[0], x_and_y_list[1])


                    
                    querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                    df_x = pd.read_sql_query(querytablex, con=engine)

                    querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                    df_y = pd.read_sql_query(querytabley, con=engine)
                   

                    columnames_x = []
                    for xcol in df_x.columns:
                        #print(col)
                        if xcol != "Edit" and xcol != "Delete":
                            columnames_x.append(xcol)

                    columnames_y = []
                    for ycol in df_y.columns:
                        #print(col)
                        if ycol != "Edit" and ycol != "Delete":
                            columnames_y.append(ycol)

                    #for c in selectedcolumnsX:
                    #    print(c.index)
                    
                    dfxx = df_x[columnames_x]
                    dfxy = df_y[columnames_y]

                 
                    
                    X = dfxx[0:int(len(dfxx)/2)]
                    print(X, X.shape)
                    y = dfxy[int(len(dfxy)/2)+1:len(dfxy)]
                    print(y, y.shape)
                    while X.shape[0] != y.shape[0]:
                        s = int(len(dfxx)/2)
                        s = s - 1
                        X = dfxx[0:s]
                        
                    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=4)

                    selectedcolumn = request.form['selectedcolumn']
                    scol = dfxx[selectedcolumn].to_numpy()

                    #dfxy

                    transdat = le.fit_transform(scol)
                    

                    return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=transdat,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test", selectedcolumn=selectedcolumn)

                

                 
                if mod=="KBinsDiscretizer":
                
                    from sklearn.preprocessing import KBinsDiscretizer

                    kbd = KBinsDiscretizer()

                    # ---------------------------------------------

                    kbd.fit(X_train)
                
                    model_nbins = kbd.n_bins_
                    model_nfeatures_in = kbd.n_features_in_
                    modelattr = [model_nbins, model_nfeatures_in]
                    attrnames = ['Bins per Feature', 'Features']

                    encoderandstrategyflag = True

                    encode_method = ['onehot', 'onehot-dense', 'ordinal']
                    strategy_used = ['quantile', 'uniform', 'kmeans']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    
                    kbd_xtrain = kbd.transform(X_train)
                    print("kbd_xtrain: ", kbd_xtrain)
                    kbd_xtest = kbd.transform(X_test)
                    print("kbd_xtest: ", kbd_xtest)

                    # ----------------------------------------------

                    kbd.fit(y_train)
                
                    model_nbins = kbd.n_bins_
                    model_nfeatures_in = kbd.n_features_in_
                    modelattr = [model_nbins, model_nfeatures_in]
                    attrnames = ['Bins per Feature', 'Features']

                    encoderandstrategyflag = True

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)
                    
                    
                    kbd_ytrain = kbd.transform(y_train)
                    print("kbd_ytrain: ", kbd_ytrain)
                    kbd_ytest = kbd.transform(y_test)
                    print("kbd_ytest: ", kbd_xtest)

                    return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                 columnames_x=columnames_x, columnames_y=columnames_y, modelattrx=combix, modelattry=combiy,
                                                                                 #labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test",
                    #return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize,
                                                                                  xtrain = kbd_xtrain, labelstrXtrain="X Train",
                                                                                  ytrain = kbd_ytrain, labelstrYtrain="y Train",
                                                                                  xtest = kbd_xtest, labelstrXtest="X Test", encode_methods=encode_method, strategies_used=strategy_used,
                                                                                  ytest = kbd_ytest, labelstrYtest="y Test", encoderandstrategyflag=encoderandstrategyflag)

          
                    
        elif request.form.get("get_encodeandstrategy") == "Select Encoder and Strategy":
                
                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                mod = request.form['transformmodel']
                #queryalltables = """SELECT * FROM {}""".format(tablename)
                #df = pd.read_sql_query(queryalltables, con=postgres_db, index_col=None)

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                # ----------------------------------------------------------
        
                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]
             
                X = dfxx[0:int(len(dfxx)/2)]
                print(X, X.shape)
                y = dfxy[int(len(dfxy)/2)+1:len(dfxy)]
                print(y, y.shape)
##                while X.shape[0] != y.shape[0]:
##                    s = int(len(dfxx)/2)
##                    s = s - 1
##                    X = dfxx[0:s]
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=4)

                # ----------------------------------------------------------

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                # ------------------------------------------------

                enc = request.form['encode_method']
                strat = request.form['strategy_used']

                print("anc & strat: ", enc, strat)

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer(encode=enc, strategy=strat)

                # ---------------------------------------------------------

                encoderandstrategyflag = False

                kbd.fit(X_train)
                    
                model_nbins = kbd.n_bins_
                model_nfeatures_in = kbd.n_features_in_
                modelattr = [model_nbins, model_nfeatures_in]
                attrnames = ['Bins per Feature', 'Features']

                encode_method = ['onehot', 'onehot-dense', 'ordinal']
                strategy_used = ['quantile', 'uniform', 'kmeans']

                combix = []

                for a, b in zip(modelattr, attrnames):
                        strr = b + ": " + str(a)
                        combix.append(strr)

                kbd_xtrain = kbd.transform(X_train)
                print("kbd_xtrain: ", kbd_xtrain)
                kbd_xtest = kbd.transform(X_test)
                print("kbd_xtest: ", kbd_xtest)

                # ----------------------------------------------

                kbd.fit(y_train)
            
                model_nbins = kbd.n_bins_
                model_nfeatures_in = kbd.n_features_in_
                modelattr = [model_nbins, model_nfeatures_in]
                attrnames = ['Bins per Feature', 'Features']

                combiy = []

                for a, b in zip(modelattr, attrnames):
                        strr = b + ": " + str(a)
                        combiy.append(strr)

                kbd_ytrain = kbd.transform(y_train)
                print("kbd_ytrain: ", kbd_ytrain)
                kbd_ytest = kbd.transform(y_test)
                print("kbd_ytest: ", kbd_xtest)

                return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                             columnames_x=columnames_x, columnames_y=columnames_y, modelattrx=combix, modelattry=combiy,
                                                                             #labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test",
                #return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize,
                                                                              xtrain = kbd_xtrain, labelstrXtrain="X Train",
                                                                              ytrain = kbd_ytrain, labelstrYtrain="y Train",
                                                                              xtest = kbd_xtest, labelstrXtest="X Test", encode_methods=encode_method, strategies_used=strategy_used,
                                                                              ytest = kbd_ytest, labelstrYtest="y Test", encoderandstrategyflag=encoderandstrategyflag)

            

    flash("Response error.")
    return render_template("preprocessing.html") #, listoftables=listoftables)


def getListOfFiles(dirName):
     
    # create a list of file and sub directories 
    # names in the given directory 
 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                 
    return allFiles  

modelist = ['Normal Plots', 'Kmeans Clusters', 'Linear Regression', 'Logistic Regression', 'Ridge Regression',
                 'Lasso Regression', 'ElasticNet Regression', 'Chained Regression', 'Neural Network (MLP Classifier)', 'Neural Network (MLP Regressor)', 'Random Forest Classifier',
                 'Random Forest Regressor', 'Support Vector Classifier', 'Support Vector Regressor', 'Naive Bayes']

@app.route("/algorithmsandmodeling", methods=['GET', 'POST'])
def algorithmsandmodeling():

    if request.method == 'GET':

    
        querytable = "SELECT mark_x_train, mark_x_test, mark_y_train, mark_y_test FROM xymarker WHERE mark_x_test IS NOT NULL"
        df = pd.read_sql_query(querytable, con=engine)
        print("launch: ", df)
        df = df.dropna(how='all')
        listxy = df.values.tolist()

  

        return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, clustersizelist=clustersizelist, linearlist=linearlist)

    if request.method == 'POST':

        querytable = "SELECT mark_x_train, mark_x_test, mark_y_train, mark_y_test FROM xymarker"
        df = pd.read_sql_query(querytable, con=engine)
        df = df.dropna(how='all')
        listxy = df.values.tolist()
        tablename = ""
        try:
            tablename = request.form['tablenamex']
        except:
            pass
               
        if request.form.get("apply") == "Submit": 
                
                mod = request.form['modelname']

                if mod == "Normal Plots":

                

                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df_xtr = pd.read_sql_query(querytable, con=engine)

                    clean_tname_xte = x_and_y_list[2].replace('xte', '')
                    querytable = "SELECT * FROM {}".format(clean_tname_xte)
                    df_xte = pd.read_sql_query(querytable, con=engine)

                    inx = pd.concat([df_xtr, df_xte], ignore_index=True)

                    columnames1 = []
                    for xcol in inx.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)

                    print("df_xtr: ", df_xtr)
                    print("df_xte: ", df_xte)
                    print("inx: ", inx)

               

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1, tablenamex=x_and_y_list)

                if mod == "Kmeans Clusters":

                   

                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    
                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)


                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1,
                                                                                                 tablenamex=x_and_y_list, clustersizelist=clustersizelist)
                
                if mod == "Linear Regression":

                    

                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                 
                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)
                            
                    tnameXINPUT = x_and_y_list[0]##.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2]##.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1] #.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3] #.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    regr = LinearRegression()
                    #regr = Lasso(alpha=0.1)
                    
                    packlistX = []
                    chartid = []
                    newlistall = []
                    linearreq = []
                    newpredlistall = []
                    convertedlist = []
                    converted = ""
                    labelxy = []
                    yaxislabel = ""
                    accy = []
                    y02_x = []
                    y02_y = []
                    for i in X.columns:
                        newlist = []
                        newpredlist = []
                        convertedpredlist = []
                        for ii in Y.columns:
                            labelxy.append([i, ii])
                            yaxislabel = ii
                            #packlistY.append(Y.iloc[:, i].tolist())
                            for a, b in zip(X[i].tolist(), Y[ii].tolist()):
                                regr.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
                                predicted = regr.predict(XTEST[i].to_numpy().reshape(-1, 1))
                                predlist = [item for sublist in predicted for item in sublist]
                                #print("predlist: ", predlist)
                                #print("X.iloc[:, i].tolist(): ", X.iloc[:, i].tolist())
                                colpack = [X[i].tolist(), Y[ii].tolist()]
                                preddat = [XTEST[i].tolist(), predlist]
                                print("predicted: ", predicted, predicted.shape)
                                
                                accuracy = regr.score(XTEST[i].to_numpy().reshape(-1, 1), YTEST[ii].to_numpy().reshape(-1, 1))
                                print("Accuracy: ", accuracy)
                                accy.append(accuracy)
                               
                                packlistX.append(colpack)
                                
                                newlist.append({'x': a, 'y': b})

##                                for h, w in zip(XTEST[i].tolist(), predictedstr.tolist()):
##                                    w0 = "\'" + w + "\'"
##                                    convertedpredlist.append({'x': h, 'y': w0})
##                                converted = str(convertedpredlist).replace('\'', '')
                                #print("converted: ", converted)

                                A = np.vstack([X[i].tolist(), np.ones(len(X[i].tolist()))]).T
                                m2, b2 = np.linalg.lstsq(A, Y[ii].to_numpy(), rcond=None)[0]
                                y02_y = m2*X[i] + b2
                                y02_x = X[i].tolist()
                                y02_y = y02_y.tolist()

                            for h, w in zip(y02_x, y02_y):
                                newpredlist.append({'x': h, 'y': w})
                            linearreq = str(newpredlist).replace('\'', '')
                            
                        newpredlistall.append(linearreq) #linearreq
                        
                        newlistall.append(newlist)

  
                    tot = abs(sum(accy)/len(accy))
                    print("TotAVG: ", tot)
                        
                    datset = []
                    for num in range(len(X.columns)):
                        linearreq_eq = str(newlistall[num]).replace('\'', '')
                        datset.append(linearreq_eq)


                    
                    newstr = []
                    for xe in list(X.columns):
                        s = xe.translate({ord(c): None for c in string.whitespace})
                        newstr.append(s)


                    listall = [list(X.columns), list(Y.columns)]

                    print("labelxy: ", labelxy)
                    print("list(X.columns): ", list(X.columns))
                    
                    flatlist = [element for sub_list in listall for element in sub_list]  

                    #return xflag, yflag, xi, yi, slicer_eq, linearreq_eq, rawdata, tiX, tiY
                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

         
            
                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1,
                                                                                                 tablenamex=x_and_y_list, block=datset, lister=list(X.columns), ycolumn=list(Y.columns), xitem = yaxislabel,
                                                                                                 colorlist=colorlist, rgbalist=rgbalist, blockline=newpredlistall, flatlist=flatlist, labelxy = labelxy, accuracy = tot, yflag=True)
                
                if mod == "Logistic Regression":

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
                    
                    
                    
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass
                  
                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)
                            
                    tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Ywhole = pd.read_sql_query(querytable, con=engine)#.astype(float)

                    tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine)#.astype(float)

                    tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine)#.astype(float)

                    #print("YTEST raw frame: ", YTEST)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                   
                    logrog = LogisticRegression(solver='newton-cg', random_state=6, max_iter=500000)
           
                    print("X info")
                    X.info()
                    logrog.fit(X.to_numpy(), Y.to_numpy())
                    ctest = logrog.predict(XTEST.to_numpy())

                    print("Show ctest: ", ctest, ctest.shape)

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

                    accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                    print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    blk = []
                    blk2 = []
                    if list(transf_bin['transform_y'])[0] != "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': dd2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': dd2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': d2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2 in zip(xtest_column, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': d2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
            
                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
           

                    slicerflag = True
                    pointflag = True
                 

                    listall = [list(X.columns), list(Y.columns)]
                    
                    flatlist = [element for sub_list in listall for element in sub_list]  

                    #return xflag, yflag, xi, yi, slicer_eq, linearreq_eq, rawdata, tiX, tiY
                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                   

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=[],
                                                                                                 xyblockcurve2=[], pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, point=[], classlist = [], xyblockcurve3=blk2,
                                                                                                 xyblockcurve4=blk, Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')], accuracy=accu,
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])
                  
                

                if mod == "Ridge Regression":
                                    
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

                   
                    tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)
                            
                    tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)


                    ridge = Ridge(random_state=6)
                    ridge.fit(ORIX.to_numpy(), ORIY.to_numpy())
                    ctest = ridge.predict(XTEST.to_numpy())
                    
                    ##----------------------------------------------------

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    

                    accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                    accu2 = ridge.score(XTEST.to_numpy(), YTEST.to_numpy())
                    accu = [np.absolute(accu1), np.absolute(accu2)]
                    # #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                    print("Accuracy:", accu)
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    
                    ##ytest_recovered_list = YTESTList
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)
                    blk = []
                    blk2 = []
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)
 
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                    #pred_df = pd.DataFrame()
    
                

                    slicerflag = True
                    pointflag = True
                   

                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest, accuracy=accu[1],
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])

                      
                if mod == "Lasso Regression":
                                    
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])

                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

                    tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    ORIY = pd.read_sql_query(querytable, con=engine).astype(float)

                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)
                            
                    tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)



                    lasso = Lasso(random_state=6)
                    lasso.fit(ORIX.to_numpy(), ORIY.to_numpy())
                    ctest = lasso.predict(XTEST.to_numpy())

                    ##----------------------------------------------------

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                   

                    accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                    accu2 = lasso.score(XTEST.to_numpy(), YTEST.to_numpy())
                    accu = [np.absolute(accu1), np.absolute(accu2)]
                    #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                    print("Accuracy:", accu)
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    
                    ##ytest_recovered_list = YTESTList
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)
                    blk = []
                    blk2 = []
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
               
                    slicerflag = True
                    pointflag = True
                   


                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    
                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest, accuracy=accu[1],
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])

                if mod == "ElasticNet Regression":
                                
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
                 
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

                  
                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)

                    tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                            
                    tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)


                    elastic = ElasticNet(random_state=6, max_iter=50000)
                    elastic.fit(ORIX.to_numpy(), ORIY.to_numpy())
                    ctest = elastic.predict(XTEST.to_numpy())
                    
                    ##----------------------------------------------------

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    
                    accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                    accu2 = elastic.score(XTEST.to_numpy(), YTEST.to_numpy())
                    accu = [np.absolute(accu1), np.absolute(accu2)]
                    #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                    print("Accuracy:", accu)
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    
                    ##ytest_recovered_list = YTESTList
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)
                    blk = []
                    blk2 = []
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                    

                    slicerflag = True
                    pointflag = True
                   


                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    
                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns), accuracy=accu[1],
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])
                
                if mod == "Chained Regression":

                    
                    baseEstimator = "Linear Regression"

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
                                
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)

                    tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                            
                    tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)


                    orderlist = range(len(list(Y.columns)))
                    #print("orderlist: ", orderlist)
                    ##logreg = LogisticRegression(solver='lbfgs',multi_class='multinomial')
                    lreg = LinearRegression()
                    

                    chained = RegressorChain(base_estimator=lreg, order=orderlist)
                    chained.fit(ORIX.to_numpy(), ORIY.to_numpy())
                    ctest = chained.predict(XTEST.to_numpy())

                    ##----------------------------------------------------

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

          
                    accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                    accu2 = chained.score(XTEST.to_numpy(), YTEST.to_numpy())
                    accu = [np.absolute(accu1), np.absolute(accu2)]
                    #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                    print("Accuracy:", accu)
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    ##ytest_recovered_list = YTESTList
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)
                    blk = []
                    blk2 = []
                    for i in list(XTEST.columns):
                        xy_list35 = []
                        xy_list36 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list35.append({'x': c2, 'y': d2})
                        
                        x_y35 = str(xy_list35).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list36.append({'x': c2, 'y': d2})
                                
                        x_y36 = str(xy_list36).replace('\'', '')

                        blk.append(x_y35)
                        blk2.append(x_y36)

                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                    
                   

                    slicerflag = True
                    pointflag = True
                   
       
                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)

               
                    
                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], accuracy=accu[1],
                                                                                                 totalplots=totalplots, plotnames=plotnames,
                                                                                                 Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])

                

                if mod == "Random Forest Classifier":

                 

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
                    
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    origininputx_tbname = x_and_y_list[0].replace('xtr', '') + '3'
                    querytable = "SELECT * FROM {}".format(origininputx_tbname)
                    origininputxdf = pd.read_sql_query(querytable, con=engine)

                    originlabely_tbname = x_and_y_list[2].replace('ytr', '') + '3'
                    querytable = "SELECT * FROM {}".format(originlabely_tbname)
                    originlabelydf = pd.read_sql_query(querytable, con=engine)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

              
                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)

                    tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                            
                    tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Ywhole = pd.read_sql_query(querytable, con=engine)

                    tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine)

                    tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)


                    maxdepth = 8
                    nestimator = 20
                    
                    rfc = RandomForestClassifier(max_depth=maxdepth, random_state=6, n_estimators=nestimator)

                    #logregTEST = RandomForestClassifier(max_depth=maxdepth, random_state=4, n_estimators=nestimator)
                    rfc.fit(X.to_numpy(), Y.to_numpy())
                    ctest = rfc.predict(XTEST.to_numpy())

                    print("Show ctest: ", ctest, ctest.shape)

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

                    accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                    print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)

                    blk = []
                    blk2 = []
                    if list(transf_bin['transform_y'])[0] != "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': dd2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': dd2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': d2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2 in zip(xtest_column, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': d2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                        
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                    
                    

                    slicerflag = True
                    pointflag = True
                   


                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)

                    #classlist = ['Not applicable']
                    #if list(transf_bin['transform_y'])[0] != "NT":
                        #classlist = model_inv.classes_

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], accuracy=accu,
                                                                                                 totalplots=totalplots, plotnames=plotnames, classlist=[],
                                                                                                 Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 max_depth = maxdepth, n_estimator = nestimator, OriginLabelYdf=[originlabelydf.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 OriginInputXdf=[origininputxdf.to_html(classes='table table-striped text-center', justify='center')])


                if mod == "Random Forest Regressor":

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])

                  
            
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass


                  
                    
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)

                    tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                            
                    tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)


                    maxdepth = 8
                    nestimator = 20
                    
                    rfr = RandomForestRegressor(max_depth=maxdepth, random_state=6, n_estimators=nestimator)
                    rfr.fit(ORIX.to_numpy(), ORIY.to_numpy())
                    ctest = rfr.predict(XTEST.to_numpy())

                    ##----------------------------------------------------

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

                   
                    
                    accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                    accu2 = rfr.score(XTEST.to_numpy(), YTEST.to_numpy())
                    accu = [np.absolute(accu1), np.absolute(accu2)]
                    #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                    print("Accuracy:", accu)
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)

                    #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    
                    ##ytest_recovered_list = YTESTList
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)
                    blk = []
                    blk2 = []
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                 
                    
                    slicerflag = True
                    pointflag = True
                   


                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)

                    

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], yy0 = [], xx0 = [],
                                                                                                 totalplots=totalplots, plotnames=plotnames, accuracy=accu[1],
                                                                                                 Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 max_depth = maxdepth, n_estimator = nestimator)


                if mod == "Naive Bayes":

                    #processdata = request.form['dataprocessname']
                    #model_inv = load(path_rm+processdata)

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
            
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    # get *inputx data
                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

                    # get *labely data
                    clean_tnamey = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(clean_tnamey)
                    df_y = pd.read_sql_query(querytable, con=engine)

                 
         
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)


                         
                    tnameXINPUT = x_and_y_list[0]
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Ywhole = pd.read_sql_query(querytable, con=engine)

                    tnameYLABEL = x_and_y_list[2]
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine)

                    tnameXTEST = x_and_y_list[1]
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)


                    
                
                    logreg = MultinomialNB(fit_prior=True)

                    logreg.fit(X.to_numpy(), Y.to_numpy())
                    ctest = logreg.predict(XTEST.to_numpy())

                    print("Show ctest: ", ctest, ctest.shape)

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

                    accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                    print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    blk = []
                    blk2 = []
                    if list(transf_bin['transform_y'])[0] != "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': dd2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': dd2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': d2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2 in zip(xtest_column, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': d2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                        
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                    
                    #logreg.fit(X.to_numpy(), Y.to_numpy())
                 
                    slicerflag = True
                    pointflag = True
                   
                 

                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)

          

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest, 
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[],
                                                                                                 totalplots=totalplots, plotnames=plotnames, accuracy=accu,
                                                                                                 Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])
                                                                                                 #classlist=classlist)

                if mod == "Neural Network (MLP Classifier)":

                    activation = "relu"
                    solver = "adam"
                    learningrate = "constant"
                    learningrateinit = 0.001
                    epochs = 50000
                    hiddenlayers = 100

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])

                    #processdata = request.form['dataprocessname']
                    #model_inv = load(path_rm+processdata)
            
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    # get *inputx data
                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

                    # get *labely data
                    clean_tnamey = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(clean_tnamey)
                    df_y = pd.read_sql_query(querytable, con=engine)

                 
         
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)
                            
                    tnameXINPUT = x_and_y_list[0]
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Ywhole = pd.read_sql_query(querytable, con=engine)

                    tnameYLABEL = x_and_y_list[2]
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine)

                    tnameXTEST = x_and_y_list[1]
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)

                         

                    mlpc = MLPClassifier(random_state=6, max_iter=epochs, hidden_layer_sizes=hiddenlayers,
                                                       activation=activation, solver=solver, learning_rate=learningrate, learning_rate_init=learningrateinit)

                
                    mlpc.fit(X.to_numpy(), Y.to_numpy())
                    ctest = mlpc.predict(XTEST.to_numpy())

                    print("Show ctest: ", ctest, ctest.shape)

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

                    accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                    print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    blk = []
                    blk2 = []
                    if list(transf_bin['transform_y'])[0] != "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': dd2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': dd2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': d2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2 in zip(xtest_column, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': d2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                        
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)
                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)

                    #logreg.fit(X.to_numpy(), Y.to_numpy())
                 
                    slicerflag = True
                    pointflag = True
                   
                   

                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)

                    #print("predtest: ", predtest)

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=[],
                                                                                                 xyblockcurve2=[], pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], accuracy=accu,
                                                                                                 totalplots=totalplots, plotnames=plotnames, xyblockcurve3=blk2, xyblockcurve4=blk,
                                                                                                 Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 activation=activation, solver=solver, learningrate=learningrate, epochs=epochs,
                                                                                                 hiddenlayers=hiddenlayers, learningrateinit=learningrateinit, activationlist=activationlist,
                                                                                                 solverlist=solverlist, learningratelist=learningratelist)

                if mod == "Neural Network (MLP Regressor)":

                    activation = "relu"
                    solver = "adam"
                    learningrate = "constant"
                    learningrateinit = 0.001
                    epochs = 50000
                    hiddenlayers = 100

                    #processdata = request.form['dataprocessname']
                    #model_inv = load(path_rm+processdata)
            
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    # get *inputx data
                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    # get *labely data
                    clean_tnamey = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(clean_tnamey)
                    df_y = pd.read_sql_query(querytable, con=engine)

                
         
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)
                            
                    tnameXINPUT = x_and_y_list[0]
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2]
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1]
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)

                    mlpr = MLPRegressor(random_state=6, max_iter=epochs, hidden_layer_sizes=hiddenlayers,
                                                       activation=activation, solver=solver, learning_rate=learningrate, learning_rate_init=learningrateinit)
                    
                    mlpr.fit(X.to_numpy(), Y.to_numpy())
                    ctest = mlpr.predict(XTEST.to_numpy())
                    
                    #ctest_recovered = model_inv.inverse_transform(ctest)
                    ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    
                    #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                    accu2 = mlpr.score(XTEST.to_numpy(), YTEST.to_numpy())
                    accu = [np.absolute(accu1), np.absolute(accu2)]
                    #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                    print("Accuracy:", accu)
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
    
                    
                    ytest_recovered_list = YTESTList
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)
                    blk = []
                    blk2 = []
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                 
                    slicerflag = True
                    pointflag = True
                   


                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)

                    

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], accuracy=accu[1],
                                                                                                 totalplots=totalplots, plotnames=plotnames,
                                                                                                 Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 activation=activation, solver=solver, learningrate=learningrate, epochs=epochs,
                                                                                                 hiddenlayers=hiddenlayers, learningrateinit=learningrateinit, activationlist=activationlist,
                                                                                                 solverlist=solverlist, learningratelist=learningratelist)


                
                if mod == "Support Vector Classifier":

                    #processdata = request.form['dataprocessname']
                    #model_inv = load(path_rm+processdata)

                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
            
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    origininputx_tbname = x_and_y_list[0].replace('xtr', '') + '3'
                    querytable = "SELECT * FROM {}".format(origininputx_tbname)
                    origininputxdf = pd.read_sql_query(querytable, con=engine)

                    originlabely_tbname = x_and_y_list[2].replace('ytr', '') + '3'
                    querytable = "SELECT * FROM {}".format(originlabely_tbname)
                    originlabelydf = pd.read_sql_query(querytable, con=engine)

                    # get *inputx data
                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass

                    # get *labely data
                    clean_tnamey = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(clean_tnamey)
                    df_y = pd.read_sql_query(querytable, con=engine)

              
         
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)
                            
                    tnameXINPUT = x_and_y_list[0]
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Ywhole = pd.read_sql_query(querytable, con=engine)

                    tnameYLABEL = x_and_y_list[2]
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine)

                    tnameXTEST = x_and_y_list[1]
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)



                    ker = "rbf"
                    gam = "scale"

                    svclass = SVC(kernel=ker, gamma=gam)

                    #logregTEST = SVC(kernel=ker, gamma=gam)
                    svclass.fit(X.to_numpy(), Y.to_numpy())
                    ctest = svclass.predict(XTEST.to_numpy())

                    print("Show ctest: ", ctest, ctest.shape)

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

                    accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                    print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    blk = []
                    blk2 = []
                    if list(transf_bin['transform_y'])[0] != "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': dd2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': dd2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        for i in list(XTEST.columns):
                            xy_list30 = []
                            xy_list31 = []
                            xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                            for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                        #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list30.append({'x': c2, 'y': d2})
                            
                            x_y30 = str(xy_list30).replace('\'', '')

                            for c2, d2 in zip(xtest_column, ytest_recovered_list):
                                #print("X: ", X[i].tolist())
                                #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                                xy_list31.append({'x': c2, 'y': d2})
                                    
                            x_y31 = str(xy_list31).replace('\'', '')

                            blk.append(x_y30)
                            blk2.append(x_y31)
                        
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)

                    #logreg.fit(X.to_numpy(), Y.to_numpy())
                 
                    slicerflag = True
                    pointflag = True
                   
                    

                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)

                    #print("predtest: ", predtest)

                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                                 column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                                 block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                                 blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                                 xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                                 rgbalist_ytest=rgbalist_ytest, pointer=[], accuracy=accu,
                                                                                                 totalplots=totalplots, plotnames=plotnames, gamma=gam, kernel=ker, gammalist=gammalist, kernellist=kernellist,
                                                                                                 Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')], classlist=[],
                                                                                                 XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 OriginInputXdf=[origininputxdf.to_html(classes='table table-striped text-center', justify='center')],
                                                                                                 OriginLabelYdf=[originlabelydf.to_html(classes='table table-striped text-center', justify='center')])

                if mod == "Support Vector Regressor":
                    
                    #processdata = request.form['dataprocessname']
                    #model_inv = load(path_rm+processdata)
                    pathx = app.config['MODEL_FOLDER']
                    listOfFiles = getListOfFiles(pathx)
                    sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                    model_inv=load(sorted_files[0])
            
                    querytable = "SELECT mark_xy FROM xymarker"
                    df = pd.read_sql_query(querytable, con=engine)
                    list_xy = df['mark_xy'].tolist()
                    #print("list_xy: ", list_xy)
                    
                    x_and_y_list = ast.literal_eval(tablename)
                    #print("x_and_y_list: ", x_and_y_list)

                    # get *inputx data
                    clean_tname = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(clean_tname)
                    df = pd.read_sql_query(querytable, con=engine)

                    # get *labely data
                    clean_tnamey = x_and_y_list[2].replace('ytr', '')
                    querytable = "SELECT * FROM {}".format(clean_tnamey)
                    df_y = pd.read_sql_query(querytable, con=engine)

                    querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                    transf_bin = pd.read_sql_query(querytable2, con=engine)
                    print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                    for filesingle in sorted_files:
                        if list(transf_bin['transform_y'])[0] in filesingle:
                            kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                            if list(transf_bin['transform_y'])[0] != "NT":
                                try:
                                    model_inv = load(kf)
                                    print("Scaler/Encoder model loaded: ", model_inv)
                                except:
                                    pass
                    # list all model files
                    #pathx = app.config['MODEL_FOLDER']
                    #listOfFiles = getListOfFiles(pathx)
                    #sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)

                    #cleaned = []
                    #for item in sorted_files:
                        #itemnew = item.replace(path_rm, "")
                        #cleaned.append(itemnew)
         
                    columnames1 = []
                    for xcol in df.columns:
                        if xcol != "Edit" and xcol != "Delete":
                            columnames1.append(xcol)

                    tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    #print("SVR tnameYLABEL: ", tnameYLABEL)
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                            
                    tnameXINPUT = x_and_y_list[0]
                    querytable = "SELECT * FROM {}".format(tnameXINPUT)
                    X = pd.read_sql_query(querytable, con=engine).astype(float)

                    #tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                    #querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    #Ywhole = pd.read_sql_query(querytable, con=engine)

                    tnameYLABEL = x_and_y_list[2]
                    querytable = "SELECT * FROM {}".format(tnameYLABEL)
                    Y = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameXTEST = x_and_y_list[1]
                    querytable = "SELECT * FROM {}".format(tnameXTEST)
                    XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    tnameYTEST = x_and_y_list[3]
                    querytable = "SELECT * FROM {}".format(tnameYTEST)
                    YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                    shapeOfX = X.shape
                    shapeOfY = Y.shape

                    print("shapeOfX: ", shapeOfX)
                    print("shapeOfY: ", shapeOfY)

                    print("shapeOfXnp: ", X.to_numpy().shape)
                    print("shapeOfYnp: ", Y.to_numpy().shape)

                    ker = "rbf"
                    gam = "auto"
                    #maxiter = 2000

                    svreg = SVR(kernel=ker, gamma=gam)
                    svreg.fit(ORIX.to_numpy(), ORIY.to_numpy())
                    ctest = svreg.predict(XTEST.to_numpy())

                    ##----------------------------------------------------

                    ctest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ctest_recovered = model_inv.inverse_transform(ctest)
                        except:
                            ctest_recovered = ctest
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ctest_recovered = ctest
                    
                    ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                    
                    print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                    print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                    print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                    print("prediction: ", ctest, " shape: ", ctest.shape)
                    XList = X.to_numpy().tolist()
                    Y_List = Y.to_numpy().tolist()
                    YList = [item for sublist in Y_List for item in sublist]
                    XTESTList = XTEST.to_numpy().tolist()
                    ctestList = ctest.tolist()
                    YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                    ytest_recovered = np.array([2,2])
                    if list(transf_bin['transform_y'])[0] != "NT":
                        try:
                            ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                        except:
                            ytest_recovered = np.array(YTESTList)
                    elif list(transf_bin['transform_y'])[0] == "NT":
                        ytest_recovered = np.array(YTESTList)
                        
                    ytest_recovered_list = ytest_recovered.tolist()
                    print("np.array(YTESTList): ", np.array(YTESTList))
                    print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)

                    ##----------------------------------------------------
                    

                    accu = []
                    try:
                        if any(np.isnan(ctest)) == False and any(np.isnan(YTEST.to_numpy())) == False:
                            accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                            accu2 = svreg.score(XTEST.to_numpy(), YTEST.to_numpy())
                            accu = [np.absolute(accu1), np.absolute(accu2)]
                        elif any(np.isnan(ctest)) == True and any(np.isnan(YTEST.to_numpy())) == False:
                            accu1 = "NA"
                            accu2 = svreg.score(XTEST.to_numpy(), YTEST.to_numpy())
                            accu = [accu1, np.absolute(accu2)]
                        elif any(np.isnan(ctest)) == False and any(np.isnan(YTEST.to_numpy())) == True:
                            accu1 = "NA"
                            accu2 = "NA"
                            accu = [accu1, accu2]
                        elif any(np.isnan(ctest)) == True and any(np.isnan(YTEST.to_numpy())) == True:
                            accu = ["NA"]
                    except:
                        pass
                    print("Accuracy:", accu)
                    print("ctest shape: ", ctest.shape)
                    print("ytest shape: ", YTEST.to_numpy().shape)
                    
                    #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    
                    #ytest_recovered_list = YTESTList
                    print("xtr to list: ", XList)
                    print("ytr to list: ", YList)
                    print("xte to list: ", XTESTList)
                    print("pred to list: ", ctestList)
                    print("yte to list: ", YTESTList)
                    print("prediction recovered label: ", ctest_recovered_list)
                    print("yte recovered label: ", ytest_recovered_list)
                    blk = []
                    blk2 = []
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    #print("blk: ", blk)
                    #print("blk2: ", blk2)

                    
                    pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                    pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                    pred_df = pd.concat([pred01, pred02], axis=1)
                    print("pred01: ", pred01)
                    print("pred02: ", pred02)
                    print("pred_df:L ", pred_df)
                 

                    #logreg.fit(X.to_numpy(), Y.to_numpy())
                 
                    slicerflag = True
                    pointflag = True
                   
                    

                    colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                    rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                    'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                    rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                    totalplots = len(X.columns) * len(Y.columns)
                    print("Total number of plots: ", totalplots)

                    plotnames = []
                    for nn in X.columns:
                        for mm in Y.columns:
                            stringName = [nn, mm]
                            plotnames.append(stringName)


                    return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                             xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                             rgbalist_ytest=rgbalist_ytest, pointer=[],
                                                                             totalplots=totalplots, plotnames=plotnames, gamma=gam, kernel=ker, gammalist=gammalist, kernellist=kernellist,
                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                             accuracy=accu[1])




        if request.form.get("support_vector_regressor") == "Select":

                mod = "Support Vector Regressor"

                ker = request.form['kernel']
                gam = request.form['gamma']

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])
                #processdata = request.form['dataprocessname']
                #model_inv=load(path_rm+processdata)
                #model_inv = pickle.load(open(path_rm+processdata, 'rb'))
                #print("model_inv: ", model_inv)
            
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                #print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                #print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

             
            
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                svreg = SVR(kernel=ker, gamma=gam)
                #cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=6)
                #cvs = cross_val_score(svreg, X.to_numpy(), Y.to_numpy(), cv=cv)
                #print("cvs: ", cvs)
                svreg.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = svreg.predict(XTEST.to_numpy())
                
                #ctest_recovered = model_inv.inverse_transform(ctest)

                ##----------------------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                ##----------------------------------------------------
                


                accu = []
                try:
                    if any(np.isnan(ctest)) == False and any(np.isnan(YTEST.to_numpy())) == False:
                        accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                        accu2 = svreg.score(XTEST.to_numpy(), YTEST.to_numpy())
                        accu = [np.absolute(accu1), np.absolute(accu2)]
                    elif any(np.isnan(ctest)) == True and any(np.isnan(YTEST.to_numpy())) == False:
                        accu1 = "NA"
                        accu2 = svreg.score(XTEST.to_numpy(), YTEST.to_numpy())
                        accu = [accu1, np.absolute(accu2)]
                    elif any(np.isnan(ctest)) == False and any(np.isnan(YTEST.to_numpy())) == True:
                        accu1 = "NA"
                        accu2 = "NA"
                        accu = [accu1, accu2]
                    elif any(np.isnan(ctest)) == True and any(np.isnan(YTEST.to_numpy())) == True:
                        accu = ["NA"]
                except:
                    pass
                    
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                #ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)
           

                slicerflag = True
                pointflag = True
               


                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)


                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                             rgbalist_ytest=rgbalist_ytest, pointer=[], kernel=ker, kernellist=kernellist,
                                                                                             totalplots=totalplots, plotnames=plotnames, gamma=gam, gammalist=gammalist, accuracy=accu[1],
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])


        if request.form.get("support_vector_classifier") == "Select":

                mod = "Support Vector Classifier"

                ker = request.form['kernel']
                gam = request.form['gamma']
                
                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])
            
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                #print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                #print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

           
            
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                slicerflag = True
                pointflag = True
               


                svclass = SVC(kernel=ker, gamma=gam)

                #logregTEST = SVC(kernel=ker, gamma=gam)
                svclass.fit(X.to_numpy(), Y.to_numpy())
                ctest = svclass.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)



                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)


                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                             rgbalist_ytest=rgbalist_ytest, pointer=[], kernel=ker, kernellist=kernellist,
                                                                                             totalplots=totalplots, plotnames=plotnames, gamma=gam, gammalist=gammalist,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             classlist=[], accuracy=accu)
        
        if request.form.get("mlp_classifier") == "Select":

                mod = "Neural Network (MLP Classifier)"

                activation = request.form['activation']
                solver = request.form['solver']
                learningrate = request.form['learningrate']
                learningrateinit = float(request.form['learningrateinit'])
                epochs = int(request.form['epochs'])
                hiddenlayers = int(request.form['hiddenlayers'])

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

                #processdata = request.form['dataprocessname']
                #model_inv = load(path_rm+processdata)
        
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                #print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                #print("x_and_y_list: ", x_and_y_list)

                # get *inputx data
                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass


                # get *labely data
                clean_tnamey = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(clean_tnamey)
                df_y = pd.read_sql_query(querytable, con=engine)

       
     
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)

                tnameYLABEL = x_and_y_list[2]
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                mlpc = MLPClassifier(random_state=6, max_iter=epochs, hidden_layer_sizes=hiddenlayers,
                                                   activation=activation, solver=solver, learning_rate=learningrate, learning_rate_init=learningrateinit) 

                mlpc.fit(X.to_numpy(), Y.to_numpy())
                ctest = mlpc.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)


             
                slicerflag = True
                pointflag = True
               
                

                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)

                #print("predtest: ", predtest)

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=[],
                                                                                             xyblockcurve2=[], pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                             rgbalist_ytest=rgbalist_ytest, pointer=[], accuracy=accu,
                                                                                             totalplots=totalplots, plotnames=plotnames, xyblockcurve3=blk2, xyblockcurve4=blk,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             activation=activation, solver=solver, learningrate=learningrate, epochs=epochs,
                                                                                             hiddenlayers=hiddenlayers, learningrateinit=learningrateinit, activationlist=activationlist,
                                                                                             solverlist=solverlist, learningratelist=learningratelist)






        if request.form.get("mlp_regressor") == "Select":

                mod = "Neural Network (MLP Regressor)"

                activation = request.form['activation']
                solver = request.form['solver']
                learningrate = request.form['learningrate']
                learningrateinit = float(request.form['learningrateinit'])
                epochs = int(request.form['epochs'])
                hiddenlayers = int(request.form['hiddenlayers'])

                #processdata = request.form['dataprocessname']
                #model_inv=load(path_rm+processdata)
                #model_inv = pickle.load(open(path_rm+processdata, 'rb'))
                #print("model_inv: ", model_inv)
            
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                #print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                #print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

       
            
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)


                
           
                mlpr = MLPRegressor(random_state=6, max_iter=epochs, hidden_layer_sizes=hiddenlayers,
                                                       activation=activation, solver=solver, learning_rate=learningrate, learning_rate_init=learningrateinit)
                mlpr.fit(X.to_numpy(), Y.to_numpy())
                ctest = mlpr.predict(XTEST.to_numpy())
                
                ctest_recovered = ctest
                    
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = mlpr.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)

                slicerflag = True
                pointflag = True
               


                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)


                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                             rgbalist_ytest=rgbalist_ytest, pointer=[], accuracy=accu[1],
                                                                                             totalplots=totalplots, plotnames=plotnames,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             activation=activation, solver=solver, learningrate=learningrate, epochs=epochs,
                                                                                             hiddenlayers=hiddenlayers, learningrateinit=learningrateinit, activationlist=activationlist,
                                                                                             solverlist=solverlist, learningratelist=learningratelist)

        
        if request.form.get("random_forest_regressor") == "Select":

                mod = "Random Forest Regressor"

                n_est = request.form['n_estimator']
                m_depth = request.form['max_depth']

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])
               
            
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                #print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                #print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass


            
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)


                inputStr = ""
                if m_depth == "None":
                    m_depth = None
                else:
                    m_depth = int(m_depth)
                
                rfr = RandomForestRegressor(max_depth=m_depth, random_state=6, n_estimators=int(n_est))

                rfr.fit(ORIX.to_numpy(), ORIY.to_numpy())
                    
                ctest = rfr.predict(XTEST.to_numpy())

                ##----------------------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                ##----------------------------------------------------

           

                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = rfr.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ##ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)

                slicerflag = True
                pointflag = True
               


                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)


                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                             rgbalist_ytest=rgbalist_ytest, pointer=[],
                                                                                             totalplots=totalplots, plotnames=plotnames, accuracy=accu[1],
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             max_depth = m_depth, n_estimator = n_est)
                

        
        if request.form.get("random_forest_classifier") == "Select":

                mod = "Random Forest Classifier"

                n_est = request.form['n_estimator']
                m_depth = request.form['max_depth']
                

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])
            
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                #print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                #print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

             
            
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)


                
                
                inputStr = ""
                if m_depth == "None":
                    m_depth = None
                else:
                    m_depth = int(m_depth)
                
             

                slicerflag = True
                pointflag = True
               

                rfc = RandomForestClassifier(max_depth=m_depth, random_state=4, n_estimators=int(n_est))

                #logregTEST = SVC(kernel=ker, gamma=gam)
                rfc.fit(X.to_numpy(), Y.to_numpy())
                ctest = rfc.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)

                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['forestgreen', 'carbon red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                            'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)


                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, colorlist_ytest=colorlist_ytest,
                                                                                             rgbalist_ytest=rgbalist_ytest, pointer=[],
                                                                                             totalplots=totalplots, plotnames=plotnames, accuracy=accu,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             max_depth = m_depth, n_estimator = n_est)

        if request.form.get("confirm_xrfrr") == "Find the Optimum Y Label/Target":

                mod = "Random Forest Regressor"

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

               

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                n_est = request.form['n_estimator']
                m_depth = request.form['max_depth']
              

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass


                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------

                inputStr = ""
                if m_depth == "None":
                    m_depth = None
                else:
                    m_depth = int(m_depth)

                rfr = RandomForestRegressor(max_depth=m_depth, random_state=6, n_estimators=int(n_est))
                rfr.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = rfr.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                #---------------------------------------
                ##----------------------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                
                jsonpair = ctest_recovered[smallest_val_position]
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                # elif list(transf_bin['transform_y'])[0] == "NT":
                #     ctest_recovered = ctest
                #     jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                #     point1 = [{'x': float(valuex), 'y': jsonpair}]
                #     pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------


                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = rfr.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ##ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)
                
      

                slicerflag = True
                pointflag = True
              
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)
                                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp,
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames, accuracy=accu[1],
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             pointList=pointlist, max_depth = m_depth, n_estimator = n_est)


        
        if request.form.get("confirm_xrfcr") == "Find the Optimum Y Label/Target":

                mod = "Random Forest Classifier"

              

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                n_est = request.form['n_estimator']
                m_depth = request.form['max_depth']

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

           

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

            
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

              
                
                inputStr = ""
                if m_depth == "None":
                    m_depth = None
                else:
                    m_depth = int(m_depth)
                
                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------    
                
                rfclass = RandomForestClassifier(max_depth=m_depth, random_state=6, n_estimators=int(n_est))
                rfclass.fit(X.to_numpy(), Y.to_numpy())
                ctest = rfclass.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                #---------------------------------------

                point1 = []
                pp = []
                pointlist = []
                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                        jsonpair = "\'" + str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]" + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]"]
                    except:
                        ctest_recovered = ctest
                        jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                    jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                    point1 = [{'x': float(valuex), 'y': jsonpair}]
                    pointlist = [float(valuex), ctest_recovered[smallest_val_position]]

                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #------------------------------------------------
                
             

                slicerflag = True
                pointflag = True


                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)

              
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)
                                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=[], pointerstr=pp,
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames, accuracy=accu,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             pointList=pointlist, max_depth = m_depth, n_estimator = n_est, classlist=[])

        if request.form.get("confirm_xsvrr") == "Find the Optimum Y Label/Target":

                mod = "Support Vector Regressor"

             

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                ker = request.form['kernel']
                gam = request.form['gamma']

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])
              

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

                
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                #print("SVR tnameYLABEL: ", tnameYLABEL)
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------

                svreg = SVR(kernel=ker, gamma=gam)

                #logregTEST = SVC(kernel=ker, gamma=gam)
                svreg.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = svreg.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                ##----------------------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                ##----------------------------------------------------

              
                jsonpair = ctest_recovered[smallest_val_position]
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------



                accu = []
                try:
                    if any(np.isnan(ctest)) == False and any(np.isnan(YTEST.to_numpy())) == False:
                        accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                        accu2 = svreg.score(XTEST.to_numpy(), YTEST.to_numpy())
                        accu = [np.absolute(accu1), np.absolute(accu2)]
                    elif any(np.isnan(ctest)) == True and any(np.isnan(YTEST.to_numpy())) == False:
                        accu1 = "NA"
                        accu2 = svreg.score(XTEST.to_numpy(), YTEST.to_numpy())
                        accu = [accu1, np.absolute(accu2)]
                    elif any(np.isnan(ctest)) == False and any(np.isnan(YTEST.to_numpy())) == True:
                        accu1 = "NA"
                        accu2 = "NA"
                        accu = [accu1, accu2]
                    elif any(np.isnan(ctest)) == True and any(np.isnan(YTEST.to_numpy())) == True:
                        accu = ["NA"]
                except:
                    pass
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                #ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)


                slicerflag = True
                pointflag = True

                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)
                                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp, pointerstr=[],
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames, accuracy=accu[1],
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             pointList=pointlist, kernel = ker, gamma = gam, gammalist = gammalist, kernellist = kernellist)
            
        if request.form.get("confirm_mlpregressor") == "Find the Optimum Y Label/Target":

                mod = "Neural Network (MLP Regressor)"

                activation = request.form['activation']
                solver = request.form['solver']
                learningrate = request.form['learningrate']
                learningrateinit = float(request.form['learningrateinit'])
                epochs = int(request.form['epochs'])
                hiddenlayers = int(request.form['hiddenlayers'])


                valuex = request.form['findxlr']
                choicex = request.form['xitem']

               

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

         
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                

                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------

                mlpr = MLPRegressor(random_state=6, max_iter=epochs, hidden_layer_sizes=hiddenlayers,
                                                       activation=activation, solver=solver, learning_rate=learningrate, learning_rate_init=learningrateinit)
                    
                mlpr.fit(X.to_numpy(), Y.to_numpy())
                ctest = mlpr.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                #---------------------------------------

               
                ctest_recovered = ctest
                #jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                jsonpair = ctest_recovered[smallest_val_position]
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------


                #ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]


                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = mlpr.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)
               

               

                slicerflag = True
                pointflag = True
               
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)
                                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp, accuracy=accu[1],
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             pointList=pointlist, activation=activation, solver=solver, learningrate=learningrate, epochs=epochs,
                                                                                             hiddenlayers=hiddenlayers, learningrateinit=learningrateinit, activationlist=activationlist,
                                                                                             solverlist=solverlist, learningratelist=learningratelist)
        

        if request.form.get("confirm_mlpclassifier") == "Find the Optimum Y Label/Target":

                mod = "Neural Network (MLP Classifier)"

                activation = request.form['activation']
                solver = request.form['solver']
                learningrate = request.form['learningrate']
                learningrateinit = float(request.form['learningrateinit'])
                epochs = int(request.form['epochs'])
                hiddenlayers = int(request.form['hiddenlayers'])

              

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

               

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                
                slicerflag = True
                pointflag = True


                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------


                mlpc = MLPClassifier(random_state=12, max_iter=epochs, hidden_layer_sizes=hiddenlayers,
                                     activation=activation, solver=solver, learning_rate=learningrate, 
                                     learning_rate_init=learningrateinit)

                mlpc.fit(X.to_numpy(), Y.to_numpy())
                ctest = mlpc.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                #---------------------------------------

                point1 = []
                pp = []
                pointlist = []
                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                        jsonpair = "\'" + str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]" + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]"]
                    except:
                        ctest_recovered = ctest
                        jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                    jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                    point1 = [{'x': float(valuex), 'y': jsonpair}]
                    pointlist = [float(valuex), ctest_recovered[smallest_val_position]]

                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------


          
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)

            
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)
                                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=[],
                                                                                             xyblockcurve2=[], xyblockcurve3=blk2, pointer=[],
                                                                                             xyblockcurve4=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointerstr=pp,
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames, accuracy=accu, 
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             pointList=pointlist, activation=activation, solver=solver, learningrate=learningrate, epochs=epochs,
                                                                                             hiddenlayers=hiddenlayers, learningrateinit=learningrateinit, activationlist=activationlist,
                                                                                             solverlist=solverlist, learningratelist=learningratelist, classlist=[])




            
        
        
        if request.form.get("confirm_xsvcr") == "Find the Optimum Y Label/Target":

                mod = "Support Vector Classifier"

                

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                ker = request.form['kernel']
                gam = request.form['gamma']

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

           

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

               
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                
                

                slicerflag = True
                pointflag = True


                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------


                svclass = SVC(kernel=ker, gamma=gam)

                #logregTEST = SVC(kernel=ker, gamma=gam)
                svclass.fit(X.to_numpy(), Y.to_numpy())
                ctest = svclass.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)


                #---------------------------------------

                point1 = []
                pp = []
                pointlist = []
                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                        jsonpair = "\'" + str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]" + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]"]
                    except:
                        ctest_recovered = ctest
                        jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                    jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                    point1 = [{'x': float(valuex), 'y': jsonpair}]
                    pointlist = [float(valuex), ctest_recovered[smallest_val_position]]

                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------


                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = np.absolute(metrics.accuracy_score(YTEST.to_numpy(), ctest.reshape(-1, 1)))
                print("Accuracy:", metrics.accuracy_score(YTEST.to_numpy(), ctest))
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
              
                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)

              
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)
                                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex, accuracy=accu,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=[], pointerstr=pp,
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames, pointList=pointlist,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             kernel = ker, gamma = gam, gammalist = gammalist, kernellist = kernellist, classlist=[])
            
        
        if request.form.get("confirm_xlogr") == "Find the Optimum Y Label/Target":

                mod = "Logistic Regression"

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

              

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

            

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)#.astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                logreg = LogisticRegression(solver="newton-cg", max_iter=500000, random_state=6)
                    #logregTEST = ClassifierChain(base_lr, order='random', random_state=6)
                #logregTEST = logreg
                #logregTEST = LogisticRegression(random_state=6)
                logreg.fit(X.to_numpy(), Y.to_numpy())
                ctest = logreg.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                point1 = []
                pp = []
                pointlist = []
                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                        jsonpair = "\'" + str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]" + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]"]
                    except:
                        ctest_recovered = ctest
                        jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                    jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                    point1 = [{'x': float(valuex), 'y': jsonpair}]
                    pointlist = [float(valuex), ctest_recovered[smallest_val_position]]

                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("---> xte to list: ", XTESTList)
                print("--->pred to list: ", ctestList)
                print("--->yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = metrics.accuracy_score(YTEST.to_numpy().reshape(-1, 1), ctest)
                print("Accuracy:", accuracy_score(YTEST.to_numpy().reshape(-1, 1), ctest))
                print("ctest shape: ", ctest.shape)
                
                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)



                slicerflag = True
                pointflag = True
         
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']

               
                classlist = ['Not applicable']
                try:
                    print("model_inv: ", model_inv, ", Classes: ", model_inv.classes_)
                    classlist = model_inv.classes_
                except:
                    classlist = ['Not applicable']
                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=[],
                                                                                             xyblockcurve2=[], pointflag=pointflag, xitem=choicex, findxlr=valuex, pointList=pointlist,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointerstr=pp,
                                                                                             output=0, classlist = classlist, xyblockcurve3=blk2,
                                                                                             xyblockcurve4=blk, Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')], accuracy=accu,
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])

        if request.form.get("confirm_xridr") == "Find the Optimum Y Label/Target":

                mod = "Ridge Regression"

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

         
                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

             

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass


                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)

    
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)


                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------

                cvridge = Ridge(random_state=6)
                cvridge.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = cvridge.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                #---------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("ctest recovered list: ", ctest_recovered_list)
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()

                print("ytest recovered list: ", ytest_recovered_list)
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
             

                jsonpair = ctest_recovered[smallest_val_position]
                print("jsonpair: ", jsonpair)
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------

                


                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = cvridge.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ##ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)



                slicerflag = True
                pointflag = True

             

                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns), accuracy=accu[1],
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp,
                                                                                             output=0, pointList=pointlist, Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])

            
        if request.form.get("confirm_yridr") == "Find the Optimum X Inputs":

                mod = "Ridge Regression"

              
                
                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                valuey = request.form['findylr']
                choicey = request.form['yitem']

                x_and_y_list = ast.literal_eval(tablename)
                ###print("x_and_y_list: ", x_and_y_list)

                if valuex and choicex:

                    clean_tbname = x_and_y_list[0].replace('xtr', '')
                    querytable = """SELECT "{}" FROM {}""".format(choicex, clean_tbname)
                    dfcol = pd.read_sql_query(querytable, con=engine)
                    print("ylr dfcol: ", dfcol.values.flatten().tolist())


                
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                ###print("list_xy: ", list_xy)
                
                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0] #.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)
                
                tnameYLABEL = x_and_y_list[2] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                ###print("shapeOfX: ", shapeOfX)
                ###print("shapeOfY: ", shapeOfY)

                regr = Ridge(alpha=0.1)

                slicerflag = True
                pointflag = True
                
                packlistX = []
                chartid = []
                newlistall = []
                newpredlistall = []
                newslicerlistall = []
                linearreq = []
                xvalist = []
                newpointlist = []
                xi = 0
                yi = 0
                labelxy = []
                for i in X.columns:
                    newlist = []
                    newpredlist = []
                    newpoint = []
                    newcutlist = []
                    slicer_pt = ""
                    for ii in Y.columns:
                        labelxy.append([i, ii])
                        for a, b in zip(X[i].tolist(), Y[ii].tolist()):
                            regr.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
                            predicted = regr.predict(X[i].to_numpy().reshape(-1, 1))
                            predlist = [item for sublist in predicted for item in sublist]
                          
                            colpack = [X[i].tolist(), Y[ii].tolist()]
                            preddat = [X[i].tolist(), predlist]
                          
                            packlistX.append(colpack)
                            newlist.append({'x': a, 'y': b})

                            A = np.vstack([X[i].tolist(), np.ones(len(X[i].tolist()))]).T
                            m1, b1 = np.linalg.lstsq(A, Y[ii].to_numpy(), rcond=None)[0]
                            y02_y = m1*X[i] + b1
                            y02_x = X[i].tolist()
                            y02_y = y02_y.tolist()

                            for h, w in zip(y02_x, y02_y):
                                newpredlist.append({'x': h, 'y': w})
                            linearreq = str(newpredlist).replace('\'', '')

                            numrows = len(X)

                            horizontal_line = np.full((numrows, 1), float(valuey)).ravel()
                            
                            ###print("horizontal_line: ", horizontal_line, type(horizontal_line), horizontal_line.shape)
                            #print("np.array(X[i].tolist()): ", np.array(X[i].tolist()), type(np.array(X[i].tolist())), np.array(X[i].tolist()).shape)
                            w2 = np.stack((np.array(X[i].tolist()), horizontal_line), axis=-1)
                            ###print("w2: ", w2, type(w2))

                            A = np.vstack([X[i].tolist(), np.ones(len(X[i].tolist()))]).T
                            m2, b2 = np.linalg.lstsq(A, horizontal_line, rcond=None)[0]
                            y02cut_y = m2*X[i] + b2
                            y02cut_x = X[i].tolist()
                            y02cut_y = y02cut_y.tolist()

                            for h, w in zip(y02cut_x, y02cut_y):
                                newcutlist.append({'x': h, 'y': w})
                            slicer_eq = str(newcutlist).replace('\'', '')

                            xi = (b1-b2) / (m2-m1)
                            yi = m1 * xi + b1

                            ###print("xi: ", xi, "  ", "yi: ", yi)

                        newpoint.append({'x': xi, 'y': yi})
                        slicer_pt = str(newpoint).replace('\'', '')

                    xvalist.append(xi)
                        
                    newpointlist.append(slicer_pt)
                            
                    newpredlistall.append(linearreq)

                    newslicerlistall.append(slicer_eq)
        
                    newlistall.append(newlist)
                 

                dff = pd.DataFrame(xvalist, index=list(X.columns), columns=['Computed Value'])
                    
                datset = []
                for num in range(len(X.columns)):
                    linearreq_eq = str(newlistall[num]).replace('\'', '')
                    datset.append(linearreq_eq)

                newstr = []
                for xe in list(X.columns):
                    s = xe.translate({ord(c): None for c in string.whitespace})
                    newstr.append(s)

                listall = [list(X.columns), list(Y.columns)]
                
                flatlist = [element for sub_list in listall for element in sub_list]  

                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']
        
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1,
                                                                                             tablenamex=x_and_y_list, block=datset, lister=list(X.columns), ycolumn=list(Y.columns), xitem=choicex, findxlr=valuex,
                                                                                             yitem=choicey, findylr=valuey, labelxy = labelxy, y_label=choicey,
                                                                                             colorlist=colorlist, rgbalist=rgbalist, blockline=newpredlistall, flatlist=flatlist, datframe=[dff.to_html(index=True)], newpointlist=newpointlist,
                                                                                             slicerflag=slicerflag, pointflag=pointflag, newslicerlistall=newslicerlistall)




        
        if request.form.get("confirm_xridge") == "Find the Optimum Y Label/Target":

                mod = "Ridge Regression"

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

            

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

      

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

         
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)


                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------

                #ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
                ridge = Ridge(random_state=6)
                ridge.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = ridge.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                #---------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("ctest recovered list: ", ctest_recovered_list)
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()

                print("ytest recovered list: ", ytest_recovered_list)
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
   

                jsonpair = ctest_recovered[smallest_val_position][0]
                print("jsonpair: ", jsonpair)
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position][0]]
                
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------

            


                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = ridge.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ##ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)



                slicerflag = True
                pointflag = True
                
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns), accuracy=accu[1],
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp,
                                                                                             output=0, pointList=pointlist, Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])

                
        if request.form.get("confirm_xlassor") == "Find the Optimum Y Label/Target":

                mod = "Lasso Regression"

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

              

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

            

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass


                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)

            
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)


                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------

                lasso = Lasso(random_state=6)
                lasso.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = lasso.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                #---------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("ctest recovered list: ", ctest_recovered_list)
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()

                print("ytest recovered list: ", ytest_recovered_list)
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
              

                jsonpair = ctest_recovered[smallest_val_position]
                print("jsonpair: ", jsonpair)
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------

                


                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = lasso.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ##ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)



                slicerflag = True
                pointflag = True
                
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns), accuracy=accu[1],
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp,
                                                                                             output=0, pointList=pointlist, Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])

        if request.form.get("confirm_xelasticr") == "Find the Optimum Y Label/Target":

                mod = "ElasticNet Regression"

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

            

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])
                

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass


                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)


                #---------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------

                elastic = ElasticNet(random_state=6, max_iter=50000)
                elastic.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = elastic.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                ##----------------------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("ctest recovered list: ", ctest_recovered_list)
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()

                print("ytest recovered list: ", ytest_recovered_list)
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
               

                jsonpair = ctest_recovered[smallest_val_position]
                print("jsonpair: ", jsonpair)
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------


                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = elastic.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                
                ##ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list30 = []
                    xy_list31 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list30.append({'x': c2, 'y': d2})
                    
                    x_y30 = str(xy_list30).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list31.append({'x': c2, 'y': d2})
                            
                    x_y31 = str(xy_list31).replace('\'', '')

                    blk.append(x_y30)
                    blk2.append(x_y31)
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)



                slicerflag = True
                pointflag = True
          
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']

                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp,
                                                                                             output=0, pointList=pointlist, accuracy=accu[1], Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')])


##            
        if request.form.get("confirm_xchainedr") == "Find the Optimum Y Label/Target":

                mod = "Chained Regression"

               

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

       

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])

               

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

     
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                tnameXINPUT = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                ORIX = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                ORIY = pd.read_sql_query(querytable, con=engine).astype(float)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)



                slicerflag = True
                pointflag = True
               
                #---------------------------------------

                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())

                #---------------------------------------    

                orderlist = range(len(list(Y.columns)))
                
                lreg = LinearRegression()
          

                chained = RegressorChain(base_estimator=lreg, order=orderlist)
                chained.fit(ORIX.to_numpy(), ORIY.to_numpy())
                ctest = chained.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)

                ##----------------------------------------------------

                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                    except:
                        ctest_recovered = ctest
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                jsonpair = ctest_recovered[smallest_val_position][0]
                print("jsonpair: ", jsonpair)
                point1 = [{'x': float(valuex), 'y': jsonpair}]
                pointlist = [float(valuex), ctest_recovered[smallest_val_position][0]]
                
                pp = []
                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)

                #---------------------------------------

            

                #----------------------------------------------------------------------
                
                #ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                accu1 = metrics.r2_score(YTEST.to_numpy(), ctest.reshape(-1, 1))
                accu2 = chained.score(XTEST.to_numpy(), YTEST.to_numpy())
                accu = [np.absolute(accu1), np.absolute(accu2)]
                #print("Accuracy:", metrics.r2_score(YTEST.to_numpy(), ctest))
                print("Accuracy:", accu)
                print("ctest shape: ", ctest.shape)
                print("ytest shape: ", YTEST.to_numpy().shape)
                
                
                ##ytest_recovered_list = YTESTList
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)
                blk = []
                blk2 = []
                for i in list(XTEST.columns):
                    xy_list35 = []
                    xy_list36 = []
                    xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                    for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list35.append({'x': c2, 'y': d2})
                    
                    x_y35 = str(xy_list35).replace('\'', '')

                    for c2, d2 in zip(xtest_column, ytest_recovered_list):
                        #print("X: ", X[i].tolist())
                        #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                        xy_list36.append({'x': c2, 'y': d2})
                            
                    x_y36 = str(xy_list36).replace('\'', '')

                    blk.append(x_y35)
                    blk2.append(x_y36)

                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)


                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)
                                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=pp,
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames, accuracy=accu[1],
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             pointList=pointlist)
        
        if request.form.get("confirm_xnbr") == "Find the Optimum Y Label/Target":

                mod = "Naive Bayes"

             

                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                pathx = app.config['MODEL_FOLDER']
                listOfFiles = getListOfFiles(pathx)
                sorted_files = sorted(listOfFiles, key=os.path.getmtime, reverse=True)
                model_inv=load(sorted_files[0])
                

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                querytable2 = "SELECT transform_y FROM xymarker WHERE mark_x='{}'".format(clean_tname)
                transf_bin = pd.read_sql_query(querytable2, con=engine)
                print("transf_bin list: ", list(transf_bin['transform_y']), " | value: ", list(transf_bin['transform_y'])[0])
                for filesingle in sorted_files:
                    if list(transf_bin['transform_y'])[0] in filesingle:
                        kf = os.path.join(app.config['MODEL_FOLDER'], list(transf_bin['transform_y'])[0])
                        if list(transf_bin['transform_y'])[0] != "NT":
                            try:
                                model_inv = load(kf)
                                print("Scaler/Encoder model loaded: ", model_inv)
                            except:
                                pass

             
                
                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0]#.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYLABEL = x_and_y_list[2].replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Ywhole = pd.read_sql_query(querytable, con=engine)

                tnameYLABEL = x_and_y_list[2]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine)

                tnameXTEST = x_and_y_list[1]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3]#.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                print("shapeOfXnp: ", X.to_numpy().shape)
                print("shapeOfYnp: ", Y.to_numpy().shape)

              
                #-------------------------------------------------------------------
                np_choicex = np.absolute(XTEST[choicex].to_numpy() - float(valuex))
                #dff_np_choicex = np.diff(np_choicex)
                smallest_val_position = np_choicex.argmin()
                print("Smallest val position: ", smallest_val_position, " of the array: ", np_choicex, np_choicex.shape)
                #minv = np.amin(dff_np_choicex)
                #print("min val & index: ", minv)
                print("Before: ", XTEST[choicex].to_numpy())
                XTEST.at[smallest_val_position, choicex] = float(valuex)
                #print("Mod XTEST: ", XTEST.at[0, choicex])
                print("After: ", XTEST[choicex].to_numpy())
                #-------------------------------------------------------------------

                mnb = MultinomialNB(fit_prior=True)
                mnb.fit(X.to_numpy(), Y.to_numpy())
                ctest = mnb.predict(XTEST.to_numpy())

                print("Show ctest: ", ctest, ctest.shape)
                #-------------------------------------------------------------------

                point1 = []
                pp = []
                pointlist = []
                ctest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ctest_recovered = model_inv.inverse_transform(ctest)
                        jsonpair = "\'" + str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]" + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), str(int(ctest[smallest_val_position])) + "[" + str(ctest_recovered[smallest_val_position]) + "]"]
                    except:
                        ctest_recovered = ctest
                        jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                        point1 = [{'x': float(valuex), 'y': jsonpair}]
                        pointlist = [float(valuex), ctest_recovered[smallest_val_position]]
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ctest_recovered = ctest
                    jsonpair = "\'" + str(ctest_recovered[smallest_val_position]) + "\'"
                    point1 = [{'x': float(valuex), 'y': jsonpair}]
                    pointlist = [float(valuex), ctest_recovered[smallest_val_position]]

                pp1 = str(point1).replace('\'', '')
                pp.append(pp1)
                print("Point: ", pp)
                print("Point Values: ", pointlist)
                #-------------------------------------------------------------------

                
                
                ctest_recovered_list = [item for sublist in ctest_recovered.reshape(-1, 1).tolist() for item in sublist] #ctest_recovered.tolist()
                
                print("xtr: ", X.to_numpy(), " shape: ", X.to_numpy().shape)
                print("ytr: ", Y.to_numpy(), " shape: ", Y.to_numpy().shape)
                print("xte: ", XTEST.to_numpy(), " shape: ", XTEST.to_numpy().shape)
                print("prediction: ", ctest, " shape: ", ctest.shape)
                XList = X.to_numpy().tolist()
                Y_List = Y.to_numpy().tolist()
                YList = [item for sublist in Y_List for item in sublist]
                XTESTList = XTEST.to_numpy().tolist()
                ctestList = ctest.tolist()
                YTESTList = [item for sublist in YTEST.to_numpy().reshape(-1, 1).tolist() for item in sublist]

                ytest_recovered = np.array([2,2])
                if list(transf_bin['transform_y'])[0] != "NT":
                    try:
                        ytest_recovered = model_inv.inverse_transform(np.array(YTESTList))
                    except:
                        ytest_recovered = np.array(YTESTList)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    ytest_recovered = np.array(YTESTList)
                    
                ytest_recovered_list = ytest_recovered.tolist()
                print("np.array(YTESTList): ", np.array(YTESTList))
                print("np.array(YTESTList).tolist(): ", np.array(YTESTList).tolist())
                print("xtr to list: ", XList)
                print("ytr to list: ", YList)
                print("xte to list: ", XTESTList)
                print("pred to list: ", ctestList)
                print("yte to list: ", YTESTList)
                print("prediction recovered label: ", ctest_recovered_list)
                print("yte recovered label: ", ytest_recovered_list)

                accu = metrics.accuracy_score(YTEST.to_numpy().reshape(-1, 1), ctest)
                print("Accuracy:", accuracy_score(YTEST.to_numpy().reshape(-1, 1), ctest))
                print("ctest shape: ", ctest.shape)

                blk = []
                blk2 = []
                if list(transf_bin['transform_y'])[0] != "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2, e2 in zip(xtest_column, ctestList, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': dd2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2, e2 in zip(xtest_column, YTESTList, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': dd2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                elif list(transf_bin['transform_y'])[0] == "NT":
                    for i in list(XTEST.columns):
                        xy_list30 = []
                        xy_list31 = []
                        xtest_column = [item for sublist in XTEST[i].to_numpy().reshape(-1, 1).tolist() for item in sublist]
                        for c2, d2 in zip(xtest_column, ctest_recovered_list):
                                    #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list30.append({'x': c2, 'y': d2})
                        
                        x_y30 = str(xy_list30).replace('\'', '')

                        for c2, d2 in zip(xtest_column, ytest_recovered_list):
                            #print("X: ", X[i].tolist())
                            #dd2 = "\'" + str(int(d2)) + "[" + str(e2) + "]" + "\'"
                            xy_list31.append({'x': c2, 'y': d2})
                                
                        x_y31 = str(xy_list31).replace('\'', '')

                        blk.append(x_y30)
                        blk2.append(x_y31)
                    
                #print("blk: ", blk)
                #print("blk2: ", blk2)

                
                pred01 = pd.DataFrame(ctest_recovered_list, columns=["Predicted"])
                pred02 = pd.DataFrame(ytest_recovered_list, columns=["YTEST"])
                pred_df = pd.concat([pred01, pred02], axis=1)
                print("pred01: ", pred01)
                print("pred02: ", pred02)
                print("pred_df:L ", pred_df)                
           

        

                slicerflag = True
                pointflag = True
                
                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']

                colorlist_ytest = ['#07A40E', '#D21D64', '#182ABF', '#DCA01E', '#07A40E', '#182ABF', '#D21D64', '#DCA01E', 'green', 'red'] 
                rgbalist_ytest = ['rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(33, 165, 70, 0.7)', 'rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 255, 0.7)', 'rgba(255, 165, 0, 0.7)',
                                'rgba(55, 224, 75, 0.7)', 'rgba(224, 23, 167, 0.7)']
                
                totalplots = len(X.columns) * len(Y.columns)
                #print("Total number of plots: ", totalplots)

                plotnames = []
                for nn in X.columns:
                    for mm in Y.columns:
                        stringName = [nn, mm]
                        plotnames.append(stringName)

                #print("chart1_list: ", chart1_list)
                #print("chart2_list: ", chart2_list)
                try:
                    classlist = model_inv.classes_
                except:
                    classlist = ['Not Applicable']
                
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy,
                                                                                             column_names=columnames1, tablenamex=x_and_y_list, lister=list(X.columns),
                                                                                             block=[], colorlist=colorlist, rgbalist=rgbalist, ycolumn=list(Y.columns),
                                                                                             blockline=[], xyblockcurve=[], slicerflag=slicerflag, xyblockcurve1=blk2,
                                                                                             xyblockcurve2=blk, pointflag=pointflag, xitem=choicex, findxlr=valuex,
                                                                                             colorlist_ytest=colorlist_ytest, rgbalist_ytest=rgbalist_ytest, pointer=[], pointerstr=pp,
                                                                                             output=0, totalplots=totalplots, plotnames=plotnames, accuracy=accu,
                                                                                             Xdf = [X.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             Ydf = [Y.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             XTESTdf = [XTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             YTESTdf = [YTEST.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             PREDICTEDdf = [pred_df.to_html(classes='table table-striped text-center', justify='center')],
                                                                                             pointList=pointlist, classlist=classlist)
                 
        
        if request.form.get("confirm_xlr") == "Find the Optimum Y Label/Target":

                mod = "Linear Regression"

                yflag = False

           

                valuex = request.form['findxlr']
                choicex = request.form['xitem']
                
             

                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                #-------------------------------------------

                clean_tbname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tbname)
                dfcol = pd.read_sql_query(querytable, con=engine)
                print("dfcol: ", dfcol)

                #-------------------------------------------
                
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0] #.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)

                choicecol = X[choicex].tolist()
                print("choicecol: ", choicecol)
                
                
                #choicecoly = X[choicey].tolist()
                #print("choicecol: ", choicecoly)

                dfremainder = X.drop([choicex], axis=1)
                
                tnameYLABEL = x_and_y_list[2] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)
                
                X = pd.concat([Y, dfremainder], axis=1)
                Xdisplay = pd.concat([Y, dfremainder], axis=1)
                print("dfremainder: ", dfremainder)
                print("Data X: ", X)

                tnameXTEST = x_and_y_list[1] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)
                ORITEST = XTEST

                dfremainder_xtest = XTEST.drop([choicex], axis=1)

                tnameYTEST = x_and_y_list[3] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                XTEST = pd.concat([YTEST, dfremainder_xtest], axis=1)
                Xtestdisplay = pd.concat([YTEST, dfremainder_xtest], axis=1)
                print("dfremainder xtest: ", dfremainder_xtest)
                print("Data XTEST: ", XTEST)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                print("shapeOfX: ", shapeOfX)
                print("shapeOfY: ", shapeOfY)

                regr = LinearRegression()

                slicerflag = True
                pointflag = True
                slicelist = []
                packlistX = []
                chartid = []
                newlistall = []
                newpredlistall = []
                newslicerlistall = []
                linearreq = ""
                slicer_eq = ""
                xvalist = []
                newpointlist = []
                xi = 0
                yi = 0
                labelxy = []
                linregline = []
                accy = []
                y02_x = []
                y02_y = []
                y02cut_x = []
                y02cut_y = []
                for i in X.columns:
                    newlist = []
                    newpredlist = []
                    newpoint = []
                    newcutlist = []
                    slicer_pt = ""
                    slice1 = []
                    line0 = []
                    for ii in Y.columns:
                        labelxy.append([i, ii])
                        for a, b in zip(X[i].tolist(), choicecol):
                            regr.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
                            predicted = regr.predict(XTEST[i].to_numpy().reshape(-1, 1))
                            predlist = [item for sublist in predicted for item in sublist]

                            #print("XTEST[i].to_numpy().reshape(-1, 1): ", XTEST[i].to_numpy().reshape(-1, 1))
                            
                            accuracy = regr.score(XTEST[i].to_numpy().reshape(-1, 1), ORITEST[choicex].to_numpy().reshape(-1, 1))
                            print("Accuracy: ", accuracy)
                            accy.append(accuracy)
                          
                            colpack = [X[i].tolist(), choicecol]
                            preddat = [X[i].tolist(), predlist]
                          
                            packlistX.append(colpack)
                            newlist.append({'x': a, 'y': b})

                            A = np.vstack([X[i].tolist(), np.ones(len(X[i].tolist()))]).T
                            m1, b1 = np.linalg.lstsq(A, np.array(choicecol), rcond=None)[0]
                            y02_y = m1*X[i] + b1
                            y02_x = X[i].tolist()
                            y02_y = y02_y.tolist()

##                            for h, w in zip(y02_x, y02_y):
##                                newpredlist.append({'x': h, 'y': w})
##                                line0.append([h, w])
##                            linearreq = str(newpredlist).replace('\'', '')

                            numrows = len(X)

                            horizontal_line = np.full((numrows, 1), float(valuex)).ravel()
                            #print("horizontal_line: ", horizontal_line, type(horizontal_line), horizontal_line.shape)
                            #print("np.array(X[i].tolist()): ", np.array(X[i].tolist()), type(np.array(X[i].tolist())), np.array(X[i].tolist()).shape)
                            w2 = np.stack((np.array(X[i].tolist()), horizontal_line), axis=-1)
                            #print("w2: ", w2, type(w2))

                            A = np.vstack([X[i].tolist(), np.ones(len(X[i].tolist()))]).T
                            m2, b2 = np.linalg.lstsq(A, horizontal_line, rcond=None)[0]
                            y02cut_y = m2*X[i] + b2
                            y02cut_x = X[i].tolist()
                            y02cut_y = y02cut_y.tolist()

##                            for h, w in zip(y02cut_x, y02cut_y):
##                                newcutlist.append({'x': h, 'y': w})
##                                slice1.append([h, w])
##                            slicer_eq = str(newcutlist).replace('\'', '')
                            
                            xi = (b1-b2) / (m2-m1)
                            yi = m1 * xi + b1

                            #print("xi: ", xi, "  ", "yi: ", yi)
                        for h, w in zip(y02_x, y02_y):
                            newpredlist.append({'x': h, 'y': w})
                            line0.append([h, w])
                        linearreq = str(newpredlist).replace('\'', '')
                            
                        for h, w in zip(y02cut_x, y02cut_y):
                            newcutlist.append({'x': h, 'y': w})
                            slice1.append([h, w])
                        slicer_eq = str(newcutlist).replace('\'', '')
                            
                        newpoint.append({'x': xi, 'y': yi})
                        slicer_pt = str(newpoint).replace('\'', '')

                    xvalist.append(xi)

                    slicelist.append(slice1)
                    linregline.append(line0)
                        
                    newpointlist.append(slicer_pt)
                            
                    newpredlistall.append(linearreq)

                    newslicerlistall.append(slicer_eq)
        
                    newlistall.append(newlist)

                tot = abs(sum(accy)/len(accy))
                print("TotAVG: ", tot)
                #print("newpointlist: ", newpointlist)
                #print("newslicerlistall: ", newslicerlistall)
                #print("xvalist: ", xvalist)
                print("slicelist: ", np.array(slicelist))
                print("linregline: ", np.array(linregline))
                print("labelxy: ", labelxy, len(labelxy))
                #print("lister: ", lister, len(lister))

                solutionpoint = intersect(np.array(linregline[1]), np.array(slicelist[1]))
                print(solutionpoint)

                dff = pd.DataFrame(xvalist, index=list(Xdisplay.columns), columns=['Computed Value'])
                    
                datset = []
                for num in range(len(X.columns)):
                    linearreq_eq = str(newlistall[num]).replace('\'', '')
                    datset.append(linearreq_eq)

                #newstr = []
                #for xe in list(X.columns):
                    #s = xe.translate({ord(c): None for c in string.whitespace})
                    #newstr.append(s)

                listall = [list(X.columns), list(Y.columns)]
                
                flatlist = [element for sub_list in listall for element in sub_list]  

                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']
        
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1, labelxy = labelxy,
                                                                                             tablenamex=x_and_y_list, block=datset, lister=list(dfcol.columns), ycolumn=list(Y.columns), xitem=choicex, findxlr=valuex,
                                                                                             colorlist=colorlist, rgbalist=rgbalist, blockline=newpredlistall, flatlist=flatlist, datframe=[dff.to_html(index=True)], newpointlist=newpointlist,
                                                                                             slicerflag=slicerflag, pointflag=pointflag, newslicerlistall=newslicerlistall, accuracy=tot, yflag=yflag)

            
        if request.form.get("confirm_ylr") == "Find the Optimum X Inputs":

                mod = "Linear Regression"

                yflag = True

       
                
                valuex = request.form['findxlr']
                choicex = request.form['xitem']

                valuey = request.form['findylr']
                choicey = request.form['yitem']

                x_and_y_list = ast.literal_eval(tablename)
                ###print("x_and_y_list: ", x_and_y_list)

                if valuex and choicex:

                    clean_tbname = x_and_y_list[0].replace('xtr', '')
                    querytable = """SELECT "{}" FROM {}""".format(choicex, clean_tbname)
                    dfcol = pd.read_sql_query(querytable, con=engine)
                    print("ylr dfcol: ", dfcol.values.flatten().tolist())


                
                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                ###print("list_xy: ", list_xy)
                
                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)
                        
                tnameXINPUT = x_and_y_list[0] #.replace('xtr', '')
                querytable = "SELECT * FROM {}".format(tnameXINPUT)
                X = pd.read_sql_query(querytable, con=engine).astype(float)
                
                tnameYLABEL = x_and_y_list[2] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYLABEL)
                Y = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameXTEST = x_and_y_list[1] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameXTEST)
                XTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                tnameYTEST = x_and_y_list[3] #.replace('ytr', '')
                querytable = "SELECT * FROM {}".format(tnameYTEST)
                YTEST = pd.read_sql_query(querytable, con=engine).astype(float)

                shapeOfX = X.shape
                shapeOfY = Y.shape

                ###print("shapeOfX: ", shapeOfX)
                ###print("shapeOfY: ", shapeOfY)

                regr = LinearRegression()

                slicerflag = True
                pointflag = True
                accy = []
                packlistX = []
                chartid = []
                newlistall = []
                newpredlistall = []
                newslicerlistall = []
                linearreq = ""
                xvalist = []
                newpointlist = []
                xi = 0
                yi = 0
                labelxy = []
                y02_x = []
                y02_y = []
                y02cut_x = []
                y02cut_y = []
                slicer_eq = ""
                for i in X.columns:
                    newlist = []
                    newpredlist = []
                    newpoint = []
                    newcutlist = []
                    slicer_pt = ""
                    for ii in Y.columns:
                        labelxy.append([i, ii])
                        for a, b in zip(X[i].tolist(), Y[ii].tolist()):
                            regr.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
                            predicted = regr.predict(X[i].to_numpy().reshape(-1, 1))
                            predlist = [item for sublist in predicted for item in sublist]

                            accuracy = regr.score(XTEST[i].to_numpy().reshape(-1, 1), YTEST[ii].to_numpy().reshape(-1, 1))
                            accy.append(accuracy)
                          
                            colpack = [X[i].tolist(), Y[ii].tolist()]
                            preddat = [X[i].tolist(), predlist]
                          
                            packlistX.append(colpack)
                            newlist.append({'x': a, 'y': b})

                            A = np.vstack([X[i].tolist(), np.ones(len(X[i].tolist()))]).T
                            m1, b1 = np.linalg.lstsq(A, Y[ii].to_numpy(), rcond=None)[0]
                            y02_y = m1*X[i] + b1
                            y02_x = X[i].tolist()
                            y02_y = y02_y.tolist()

##                            for h, w in zip(y02_x, y02_y):
##                                newpredlist.append({'x': h, 'y': w})
##                            linearreq = str(newpredlist).replace('\'', '')

                            numrows = len(X)

                            horizontal_line = np.full((numrows, 1), float(valuey)).ravel()
                            
                            ###print("horizontal_line: ", horizontal_line, type(horizontal_line), horizontal_line.shape)
                            #print("np.array(X[i].tolist()): ", np.array(X[i].tolist()), type(np.array(X[i].tolist())), np.array(X[i].tolist()).shape)
                            w2 = np.stack((np.array(X[i].tolist()), horizontal_line), axis=-1)
                            ###print("w2: ", w2, type(w2))

                            A = np.vstack([X[i].tolist(), np.ones(len(X[i].tolist()))]).T
                            m2, b2 = np.linalg.lstsq(A, horizontal_line, rcond=None)[0]
                            y02cut_y = m2*X[i] + b2
                            y02cut_x = X[i].tolist()
                            y02cut_y = y02cut_y.tolist()

##                            for h, w in zip(y02cut_x, y02cut_y):
##                                newcutlist.append({'x': h, 'y': w})
##                            slicer_eq = str(newcutlist).replace('\'', '')

                            xi = (b1-b2) / (m2-m1)
                            yi = m1 * xi + b1

                            ###print("xi: ", xi, "  ", "yi: ", yi)

                        newpoint.append({'x': xi, 'y': yi})
                        slicer_pt = str(newpoint).replace('\'', '')

                        for h, w in zip(y02_x, y02_y):
                            newpredlist.append({'x': h, 'y': w})
                        linearreq = str(newpredlist).replace('\'', '')

                        for h, w in zip(y02cut_x, y02cut_y):
                            newcutlist.append({'x': h, 'y': w})
                        slicer_eq = str(newcutlist).replace('\'', '')

                    xvalist.append(xi)
                        
                    newpointlist.append(slicer_pt)
                            
                    newpredlistall.append(linearreq)

                    newslicerlistall.append(slicer_eq)
        
                    newlistall.append(newlist)
                    
                #print("newpointlist: ", newpointlist)
                #print("newslicerlistall: ", newslicerlistall)
                ###print("xvalist: ", xvalist)

                dff = pd.DataFrame(xvalist, index=list(X.columns), columns=['Computed Value'])
                    
                datset = []
                for num in range(len(X.columns)):
                    linearreq_eq = str(newlistall[num]).replace('\'', '')
                    datset.append(linearreq_eq)

                tot = abs(sum(accy)/len(accy))

                newstr = []
                for xe in list(X.columns):
                    s = xe.translate({ord(c): None for c in string.whitespace})
                    newstr.append(s)

                listall = [list(X.columns), list(Y.columns)]
                
                flatlist = [element for sub_list in listall for element in sub_list]  

                colorlist = ['green', 'red', 'blue', 'orange', 'green', 'red', 'blue', 'orange', 'forestgreen', 'carbon red']
                rgbalist = ['rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(33, 165, 70, 0.5)', 'rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)',
                                'rgba(55, 224, 75, 0.5)', 'rgba(224, 23, 167, 0.5)']
        
                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1,
                                                                                             tablenamex=x_and_y_list, block=datset, lister=list(X.columns), ycolumn=list(Y.columns), xitem=choicex, findxlr=valuex,
                                                                                             yitem=choicey, findylr=valuey, labelxy = labelxy, accuracy=tot, yflag=yflag,
                                                                                             colorlist=colorlist, rgbalist=rgbalist, blockline=newpredlistall, flatlist=flatlist, datframe=[dff.to_html(index=True)], newpointlist=newpointlist,
                                                                                             slicerflag=slicerflag, pointflag=pointflag, newslicerlistall=newslicerlistall)

                
        if request.form.get("confirmkmeans") == "Confirm":

                

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df_xtr = pd.read_sql_query(querytable, con=engine)

                

                columnames1 = []
                for xcol in df_xtr.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)



                mod = request.form['modelname']

                cat = request.form['coln1']
                xaxis = request.form['coln2']
                yaxis = request.form['coln3']
                zaxis = request.form['coln4']
                datecol = request.form['coln5']
                todate = request.form['todt']
                fromdate = request.form['fromdt']
                selectcat = request.form['selectcat']
                clustersize = request.form['clustersize']

                querytable = """SELECT "{}" FROM {}""".format(cat, clean_tname)
                dfcat = pd.read_sql_query(querytable, con=engine)
                listcat = dfcat[cat].tolist()

                querytable = """SELECT "{}" FROM {}""".format(xaxis, clean_tname)
                dfx = pd.read_sql_query(querytable, con=engine)
                listx = dfx[xaxis].tolist()

                print("listx: ", listx)

                querytable = """SELECT "{}" FROM {}""".format(yaxis, clean_tname)
                dfy = pd.read_sql_query(querytable, con=engine)
                listy = dfy[yaxis].tolist()

                querytable = """SELECT "{}" FROM {}""".format(zaxis, clean_tname)
                dfz = pd.read_sql_query(querytable, con=engine)
                listz = dfz[zaxis].tolist()
                
                querytable = """SELECT "{}" FROM {}""".format(datecol, clean_tname)
                dfdate = pd.read_sql_query(querytable, con=engine)
                listcustomdates = dfdate[datecol].tolist()

                betweenquery = """SELECT "{}" FROM {} WHERE "{}" >= '{}' AND "{}" <= '{}' ORDER BY "{}" ASC""".format(datecol, clean_tname, datecol, fromdate, datecol, todate, datecol)
                cropdatedf = pd.read_sql_query(betweenquery, con=engine)
                datelistcropped = cropdatedf.to_numpy().flatten()

                sqlquery = """SELECT "{}", "{}", "{}", "{}", "{}" FROM {} WHERE "{}" >= '{}' AND "{}" <= '{}' AND "{}" = '{}' ORDER BY "{}" ASC""".format(cat, xaxis, yaxis, zaxis, datecol, clean_tname, datecol, fromdate, datecol, todate, cat, selectcat, datecol)
                df_full = pd.read_sql_query(sqlquery, con=engine)
                category = df_full[cat].tolist()
                x_axis = df_full[xaxis].tolist()
                y_axis = df_full[yaxis].tolist()
                z_axis = df_full[zaxis].tolist()
                date_col = df_full[datecol].tolist()

                combined4xlabel = []
                for x, d in zip(x_axis, date_col):
                    combined = str(x) + ';' + str(d)
                    combined4xlabel.append(combined)

                print("combined4xlabel: ", combined4xlabel)

                combined4ylabel = []
                for x, d in zip(y_axis, date_col):
                    combined = str(x) + ';' + str(d)
                    combined4ylabel.append(combined)

                inx2D = df_full[[xaxis, yaxis]]
                inx3D = df_full[[xaxis, yaxis, zaxis]]
                
                kmeans2D = KMeans(n_clusters=int(clustersize), random_state=0).fit(inx2D)
                centroids2D = kmeans2D.cluster_centers_

                markcolor2D = kmeans2D.labels_.astype(float)     
                kmeansindexes2D = markcolor2D.tolist()
                print("kmeansindexes2D: ", kmeansindexes2D)

                labelsL = list(centroids2D[:, 0])
                valuesL = list(centroids2D[:, 1])

                newlist = []
                for h2, w2 in zip(labelsL, valuesL):
                    newlist.append({'x': h2, 'y': w2})
                block2D = str(newlist).replace('\'', '')
                print("block2D: ", block2D)
                
                xaxis2D = xaxis
                yaxis2D = yaxis
                title2D = xaxis + " vs " + yaxis

                rawlist1 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==0.:
                        rawlist1.append({'x': h2, 'y': w2})
                block2Draw1 = str(rawlist1).replace('\'', '')
                print("block2Draw1: ", block2Draw1)

                rawlist2 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==1.:
                      rawlist2.append({'x': h2, 'y': w2})
                block2Draw2 = str(rawlist2).replace('\'', '')
                print("block2Draw2: ", block2Draw2)

                rawlist3 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==2.:
                      rawlist3.append({'x': h2, 'y': w2})
                block2Draw3 = str(rawlist3).replace('\'', '')
                print("block2Draw3: ", block2Draw3)

                rawlist4 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==3.:
                      rawlist4.append({'x': h2, 'y': w2})
                block2Draw4 = str(rawlist4).replace('\'', '')
                print("block2Draw4: ", block2Draw4)

                rawlist5 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==4.:
                      rawlist5.append({'x': h2, 'y': w2})
                block2Draw5 = str(rawlist5).replace('\'', '')
                print("block2Draw5: ", block2Draw5)

                rawlist6 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==5.:
                      rawlist6.append({'x': h2, 'y': w2})
                block2Draw6 = str(rawlist6).replace('\'', '')
                print("block2Draw6: ", block2Draw6)

                rawlist7 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==6.:
                      rawlist7.append({'x': h2, 'y': w2})
                block2Draw7 = str(rawlist7).replace('\'', '')
                print("block2Draw7: ", block2Draw7)

                rawlist8 = []
                for h2, w2, indices in zip(x_axis, y_axis, kmeansindexes2D):
                    if indices==7.:
                      rawlist8.append({'x': h2, 'y': w2})
                block2Draw8 = str(rawlist8).replace('\'', '')
                print("block2Draw8: ", block2Draw8)

                # -------------------------------------------------------
                kmeans3D = KMeans(n_clusters=int(clustersize)).fit(inx3D)
                centroids3D = kmeans3D.cluster_centers_
                markcolor3D = kmeans3D.labels_.astype(float)
                kmeansindexes3D = markcolor3D.tolist()

                awlistx1 = []
                awlisty1 = []
                awlistz1 = []

                awlistx2 = []
                awlisty2 = []
                awlistz2 = []

                awlistx3 = []
                awlisty3 = []
                awlistz3 = []

                awlistx4 = []
                awlisty4 = []
                awlistz4 = []

                awlistx5 = []
                awlisty5 = []
                awlistz5 = []

                awlistx6 = []
                awlisty6 = []
                awlistz6 = []

                awlistx7 = []
                awlisty7 = []
                awlistz7 = []

                awlistx8 = []
                awlisty8 = []
                awlistz8 = []
                
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==0.:
                        awlistx1.append(float(h2))
                        awlisty1.append(float(w2))
                        awlistz1.append(float(p2))
                print("awlistx1: ", awlistx1)
                print("awlisty1: ", awlisty1)
                print("awlistz1: ", awlistz1)
                #block2Draw1 = str(rawlist1).replace('\'', '')

                #rawlist2 = []
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==1.:
                        awlistx2.append(float(h2))
                        awlisty2.append(float(w2))
                        awlistz2.append(float(p2))
                print("awlistx2: ", awlistx2)
                print("awlisty2: ", awlisty2)
                print("awlistz2: ", awlistz2)
                #block2Draw2 = str(rawlist2).replace('\'', '')

                #rawlist3 = []
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==2.:
                        awlistx3.append(float(h2))
                        awlisty3.append(float(w2))
                        awlistz3.append(float(p2))
                print("awlistx3: ", awlistx3)
                print("awlisty3: ", awlisty3)
                print("awlistz3: ", awlistz3)
                #block2Draw3 = str(rawlist3).replace('\'', '')

                #rawlist4 = []
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==3.:
                        awlistx4.append(float(h2))
                        awlisty4.append(float(w2))
                        awlistz4.append(float(p2))
                print("awlistx4: ", awlistx4)
                print("awlisty4: ", awlisty4)
                print("awlistz4: ", awlistz4)
                #block2Draw4 = str(rawlist4).replace('\'', '')

                #rawlist5 = []
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==4.:
                        awlistx5.append(float(h2))
                        awlisty5.append(float(w2))
                        awlistz5.append(float(p2))
                print("awlistx5: ", awlistx5)
                print("awlisty5: ", awlisty5)
                print("awlistz5: ", awlistz5)
                #block2Draw5 = str(rawlist5).replace('\'', '')

                #rawlist6 = []
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==5.:
                        awlistx6.append(float(h2))
                        awlisty6.append(float(w2))
                        awlistz6.append(float(p2))
                print("awlistx6: ", awlistx6)
                print("awlisty6: ", awlisty6)
                print("awlistz6: ", awlistz6)
                #block2Draw6 = str(rawlist6).replace('\'', '')

                #rawlist7 = []
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==6.:
                        awlistx7.append(float(h2))
                        awlisty7.append(float(w2))
                        awlistz7.append(float(p2))
                print("awlistx7: ", awlistx7)
                print("awlisty7: ", awlisty7)
                print("awlistz7: ", awlistz7)
                #block2Draw7 = str(rawlist7).replace('\'', '')

                #rawlist8 = []
                for h2, w2, p2, indices in zip(x_axis, y_axis, z_axis, kmeansindexes3D):
                    if indices==7.:
                        awlistx8.append(float(h2))
                        awlisty8.append(float(w2))
                        awlistz8.append(float(p2))
                print("awlistx8: ", awlistx8)
                print("awlisty8: ", awlisty8)
                print("awlistz8: ", awlistz8)

                title3D = xaxis + " vs " + yaxis + " vs " + zaxis

                dfo = pd.DataFrame(centroids3D, columns = [xaxis, yaxis, zaxis])

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1, tablenamex=x_and_y_list,
                                                                                        coln1=cat, coln2=xaxis, coln3=yaxis, coln4=zaxis, coln5=datecol, selectcat=selectcat, xaxislabel=xaxis, yaxislabel=yaxis, zaxislabel=zaxis,
                                                                                        todt=todate, fromdt=fromdate, listfromdates=listcustomdates, listtodates=listcustomdates, datelistcropped=datelistcropped,
                                                                                        listcat=listcat, final_cat=category, final_xaxis=x_axis, final_yaxis=y_axis, final_zaxis=z_axis, final_date_col=date_col,
                                                                                        combined4xlabel=combined4xlabel, combined4ylabel=combined4ylabel, block2D=block2D, clustersizelist=clustersizelist, clustersize=clustersize,
                                                                                        block2Draw1=block2Draw1, block2Draw2=block2Draw2, 
                                                                                        block2Draw3=block2Draw3, block2Draw4=block2Draw4, 
                                                                                        block2Draw5=block2Draw5, block2Draw6=block2Draw6, 
                                                                                        block2Draw7=block2Draw7, block2Draw8=block2Draw8,
                                                                                        awlistx1=awlistx1, awlisty1=awlisty1, awlistz1=awlistz1, 
                                                                                        awlistx2=awlistx2, awlisty2=awlisty2, awlistz2=awlistz2, 
                                                                                        awlistx3=awlistx3, awlisty3=awlisty3, awlistz3=awlistz3, 
                                                                                        awlistx4=awlistx4, awlisty4=awlisty4, awlistz4=awlistz4, 
                                                                                        awlistx5=awlistx5, awlisty5=awlisty5, awlistz5=awlistz5, 
                                                                                        awlistx6=awlistx6, awlisty6=awlisty6, awlistz6=awlistz6, 
                                                                                        awlistx7=awlistx7, awlisty7=awlisty7, awlistz7=awlistz7, 
                                                                                        awlistx8=awlistx8, awlisty8=awlisty8, awlistz8=awlistz8,
                                                                                        ctrdx=centroids3D[:,0].tolist(), ctrdy=centroids3D[:,1].tolist(), ctrdz=centroids3D[:,2].tolist(),
                                                                                        xaxis2D=xaxis2D, yaxis2D=yaxis2D, title2D=title2D, title3D=title3D, tables=[dfo.to_html(index=False)])
                     
        
        if request.form.get("confirm5cat") == "Confirm":

                mod = request.form['modelname']

            

                querytable = "SELECT mark_xy FROM xymarker"
                dfp = pd.read_sql_query(querytable, con=engine)
                list_xy = dfp['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                df = pd.DataFrame()
                clean_tname = ""

                #if mod == "Normal Plots":
                clean_tname = x_and_y_list[0].replace('inputxtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)



                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                cat = request.form['coln1']
                xaxis = request.form['coln2']
                yaxis = request.form['coln3']
                zaxis = request.form['coln4']
                datecol = request.form['coln5']
                todate = request.form['todt']
                fromdate = request.form['fromdt']
                selectcat = request.form['selectcat']

                querytable = """SELECT "{}" FROM {}""".format(cat, clean_tname)
                dfcat = pd.read_sql_query(querytable, con=engine)
                listcat = dfcat[cat].tolist()

                querytable = """SELECT "{}" FROM {}""".format(xaxis, clean_tname)
                dfx = pd.read_sql_query(querytable, con=engine)
                listx = dfx[xaxis].tolist()

                print("listx", listx)

                querytable = """SELECT "{}" FROM {}""".format(yaxis, clean_tname)
                dfy = pd.read_sql_query(querytable, con=engine)
                listy = dfy[yaxis].tolist()

                querytable = """SELECT "{}" FROM {}""".format(zaxis, clean_tname)
                dfz = pd.read_sql_query(querytable, con=engine)
                listz = dfz[zaxis].tolist()
                
                querytable = """SELECT "{}" FROM {}""".format(datecol, clean_tname)
                dfdate = pd.read_sql_query(querytable, con=engine)
                listcustomdates = dfdate[datecol].tolist()

                betweenquery = """SELECT "{}" FROM {} WHERE "{}" >= '{}' AND "{}" <= '{}' ORDER BY "{}" ASC""".format(datecol, clean_tname, datecol, fromdate, datecol, todate, datecol)
                cropdatedf = pd.read_sql_query(betweenquery, con=engine)
                datelistcropped = cropdatedf.to_numpy().flatten()

                sqlquery = """SELECT "{}", "{}", "{}", "{}", "{}" FROM {} WHERE "{}" >= '{}' AND "{}" <= '{}' AND "{}" = '{}' ORDER BY "{}" ASC""".format(cat, xaxis, yaxis, zaxis, datecol, clean_tname, datecol, fromdate, datecol, todate, cat, selectcat, datecol)
                df_full = pd.read_sql_query(sqlquery, con=engine)
                category = df_full[cat].tolist()
                x_axis = df_full[xaxis].tolist()
                y_axis = df_full[yaxis].tolist()
                z_axis = df_full[zaxis].tolist()
                date_col = df_full[datecol].tolist()

                combined4xlabel = []
                for x, d in zip(x_axis, date_col):
                    combined = str(x) + ";" +  str(d)
                    comb = str(combined).replace('\'', '')
                    combined4xlabel.append(comb)

                print("combined4xlabel: ", combined4xlabel)

                combined4ylabel = []
                for x, d in zip(y_axis, date_col):
                    combined = str(x) + ';' + str(d)
                    combined4ylabel.append(combined)

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1,
                                                                                        coln1=cat, coln2=xaxis, coln3=yaxis, coln4=zaxis, coln5=datecol, selectcat=selectcat, xaxislabel=xaxis, yaxislabel=yaxis, zaxislabel=zaxis,
                                                                                        todt=todate, fromdt=fromdate, listfromdates=listcustomdates, listtodates=listcustomdates, datelistcropped=datelistcropped, tablenamex=x_and_y_list,
                                                                                        listcat=listcat, final_cat=category, final_xaxis=x_axis, final_yaxis=y_axis, final_zaxis=z_axis, final_date_col=date_col, combined4xlabel=combined4xlabel, combined4ylabel=combined4ylabel)
        
        if request.form.get("pulldate") == "Pull":

           

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('inputxxtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                mod = request.form['modelname']

                cat = request.form['coln1']
                xaxis = request.form['coln2']
                yaxis = request.form['coln3']
                zaxis = request.form['coln4']
                datecol = request.form['coln5']
                todate = request.form['todt']
                fromdate = request.form['fromdt']
                selectcat = request.form['selectcat']
                querytable = """SELECT "{}" FROM {}""".format(datecol, clean_tname)
                dfdate = pd.read_sql_query(querytable, con=engine)
                listcustomdates = dfdate[datecol].tolist()
                querytable = """SELECT "{}" FROM {}""".format(cat, clean_tname)
                dfcat = pd.read_sql_query(querytable, con=engine)
                listcat = dfcat[cat].tolist()

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1, coln1=cat, coln2=xaxis, coln3=yaxis, coln4=zaxis, coln5=datecol,
                                                                                        listfromdates=listcustomdates, listtodates=listcustomdates, listcat=listcat, selectcat=selectcat,
                                                                                        todt=todate, fromdt=fromdate, tablenamex=x_and_y_list)

        if request.form.get("pulldatekmeans") == "Pull":

         

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                mod = request.form['modelname']
                clustersize = request.form['clustersize']
                cat = request.form['coln1']
                xaxis = request.form['coln2']
                yaxis = request.form['coln3']
                zaxis = request.form['coln4']
                datecol = request.form['coln5']
                todate = request.form['todt']
                fromdate = request.form['fromdt']
                selectcat = request.form['selectcat']
                querytable = """SELECT "{}" FROM {}""".format(datecol, clean_tname)
                dfdate = pd.read_sql_query(querytable, con=engine)
                listcustomdates = dfdate[datecol].tolist()
                querytable = """SELECT "{}" FROM {}""".format(cat, clean_tname)
                dfcat = pd.read_sql_query(querytable, con=engine)
                listcat = dfcat[cat].tolist()

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1, coln1=cat, coln2=xaxis, coln3=yaxis, coln4=zaxis, coln5=datecol,
                                                                                        listfromdates=listcustomdates, listtodates=listcustomdates, listcat=listcat, selectcat=selectcat, tablenamex=x_and_y_list,
                                                                                        todt=todate, fromdt=fromdate, clustersizelist=clustersizelist, clustersize=clustersize)

        if request.form.get("pullcat") == "Pull":

         

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('inputxxtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                mod = request.form['modelname']

                cat = request.form['coln1']
                xaxis = request.form['coln2']
                yaxis = request.form['coln3']
                zaxis = request.form['coln4']
                datecol = request.form['coln5']
                try: 
                    selectcat = request.form['selectcat']
                except:
                    selectcat = ""
                try:
                    todate = request.form['todt']
                except:
                    todate = ""
                try:
                    fromdate = request.form['fromdt']
                except:
                    fromdate = ""
                querytable = """SELECT "{}" FROM {}""".format(cat, clean_tname)
                dfcat = pd.read_sql_query(querytable, con=engine)
                listcat = dfcat[cat].tolist()
                querytable = """SELECT "{}" FROM {}""".format(datecol, clean_tname)
                dfdate = pd.read_sql_query(querytable, con=engine)
                listcustomdates = dfdate[datecol].tolist()

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1, coln1=cat, coln2=xaxis, coln3=yaxis, coln4=zaxis, coln5=datecol,
                                                                                        listfromdates=listcustomdates, listtodates=listcustomdates, listcat=listcat, selectcat=selectcat, tablenamex=x_and_y_list,
                                                                                        todt=todate, fromdt=fromdate)

        if request.form.get("pullcatkmeans") == "Pull":

             

                querytable = "SELECT mark_xy FROM xymarker"
                df = pd.read_sql_query(querytable, con=engine)
                list_xy = df['mark_xy'].tolist()
                print("list_xy: ", list_xy)
                
                x_and_y_list = ast.literal_eval(tablename)
                print("x_and_y_list: ", x_and_y_list)

                clean_tname = x_and_y_list[0].replace('xtr', '')
                querytable = "SELECT * FROM {}".format(clean_tname)
                df = pd.read_sql_query(querytable, con=engine)

                columnames1 = []
                for xcol in df.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames1.append(xcol)

                mod = request.form['modelname']
                clustersize = request.form['clustersize']
                cat = request.form['coln1']
                xaxis = request.form['coln2']
                yaxis = request.form['coln3']
                zaxis = request.form['coln4']
                datecol = request.form['coln5']
                try: 
                    selectcat = request.form['selectcat']
                except:
                    selectcat = ""
                try:
                    todate = request.form['todt']
                except:
                    todate = ""
                try:
                    fromdate = request.form['fromdt']
                except:
                    fromdate = ""
                querytable = """SELECT "{}" FROM {}""".format(cat, clean_tname)
                dfcat = pd.read_sql_query(querytable, con=engine)
                listcat = dfcat[cat].tolist()
                querytable = """SELECT "{}" FROM {}""".format(datecol, clean_tname)
                dfdate = pd.read_sql_query(querytable, con=engine)
                listcustomdates = dfdate[datecol].tolist()

                return render_template("algorithmsandmodeling.html", listoftables=listxy, modelnamelist=modelist, modelname=mod, listtables=list_xy, column_names=columnames1, coln1=cat, coln2=xaxis, coln3=yaxis, coln4=zaxis, coln5=datecol,
                                                                                        listfromdates=listcustomdates, listtodates=listcustomdates, listcat=listcat, selectcat=selectcat, tablenamex=x_and_y_list,
                                                                                        todt=todate, fromdt=fromdate, clustersizelist=clustersizelist, clustersize=clustersize)


                


@app.route("/preprocessing1", methods=['GET', 'POST'])
def preprocessing1():

    if request.method == 'GET':
    

        querytable = "SELECT mark_x, mark_y FROM xymarker"
        df = pd.read_sql_query(querytable, con=engine)
        df = df.dropna(how='all', axis=0)
        listxy = df.values.tolist()


        return render_template("preprocessing1.html", listoftables=listxy, modelnamelist=modelist)


    
    
    if request.method == 'POST':

        #tablename = request.form['tablename']
        
        if request.form.get("showtable1") == "Show Table":

                mod = request.form['transformmodel']

                testsize = request.form['testsize']

                markbinfile = request.form['binfilename']

          

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]

                
                #ay = request.form['tablenamey']

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                
                


                
                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)
              

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                #for c in selectedcolumnsX:
                #    print(c.index)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]



                return render_template("preprocessing1.html", modelnamelist=modelist, listoftables=myinput, df_tableX = [df_x.to_html(classes='data table-striped', escape=False)], df_tableY = [df_y.to_html(classes='data table-striped', escape=False)],
                                                                             tablenamex=x_and_y_list, tablenamey=ay, columnames_x=columnames_x, columnames_y=columnames_y, testsize=testsize, transformmodel=mod,  #tablenamex=ax
                                                                             xlabel="[Input X]", ylabel="[Target/Label Y]", labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test",
                                                                             markbinfile = markbinfile)
        
        elif request.form.get("processx1") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer()

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']

                markbinfile = request.form['binfilename']
                
                testsize = request.form['testsize']

              

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                


                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
              

                dfxxnew = df_x[columnames_x]
                dfxynew = df_y[columnames_y]

                md = np.empty([2, 2])

                for k in dfxxnew.columns:
                    if selectedcolumnx == k:
                        from sklearn import preprocessing
                        if mod == "LabelEncoder":
                            #global lefitx
                            
                            le = preprocessing.LabelEncoder()

                            lefitx = le.fit(dfxxnew[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefitx, kf, compress=True)

                            md = lefitx.transform(dfxxnew[k].to_numpy())

                            column_transform_pair_list = []
                            column_transform_pair = [k, "le"]
                            column_transform_pair_list.append(column_transform_pair)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxxnew[k] = singleColumn
                            print("dfxxnew: labelencoder snap ", dfxxnew)
                            dfxxnew.to_sql(ax, con=engine, if_exists='replace', index=False)
                        elif mod == "OneHotEncoder":
                            ohe = preprocessing.OneHotEncoder()
                            mdx = ohe.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = ohe.transform(dfxxnew[k].to_numpy().reshape(-1, 1))
                            column_transform_pair_list = []
                            column_transform_pair = [k, "ohe"]
                            column_transform_pair_list.append(column_transform_pair)
                            #singleColumn = pd.DataFrame(md)
                            #dfxxnew[k] = singleColumn
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxxnew = pd.concat([dfxxnew, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxxnew = dfxxnew.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxxnew: onehotencoder snap ", dfxxnew)
                            dfxxnew.to_sql(ax, con=engine, if_exists='replace', index=False)
                        elif mod == "LabelBinarizer":
                            lb = preprocessing.LabelBinarizer()
                            mdx = lb.fit(dfxxnew[k].to_numpy())
                            md = lb.transform(dfxxnew[k].to_numpy())
                            column_transform_pair_list = []
                            column_transform_pair = [k, "lb"]
                            column_transform_pair_list.append(column_transform_pair)
                            #singleColumn = pd.DataFrame(md)
                            #dfxxnew[k] = singleColumn
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            dfxxnew = pd.concat([dfxxnew, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxxnew = dfxxnew.drop(columns=k, axis=1)
                            print("dfxxnew: labelbinarizer snap ", dfxxnew)
                            dfxxnew.to_sql(ax, con=engine, if_exists='replace', index=False)
                        elif mod == "MinMaxScaler":
                            mm = preprocessing.MinMaxScaler()
                            mdx = mm.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = mm.transform(dfxxnew[k].to_numpy().reshape(-1, 1))
                            column_transform_pair_list = []
                            column_transform_pair = [k, "mm"]
                            column_transform_pair_list.append(column_transform_pair)

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxxnew[k] = singleColumn
                            print("dfxxnew: minmax snap ", dfxxnew)
                            dfxxnew.to_sql(ax, con=engine, if_exists='replace', index=False)
                        elif mod == "StandardScaler":
                            ss = preprocessing.StandardScaler()
                            mdx = ss.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = ss.transform(dfxxnew[k].to_numpy().reshape(-1, 1))
                            column_transform_pair_list = []
                            column_transform_pair = [k, "ss"]
                            column_transform_pair_list.append(column_transform_pair)

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'StandardScaler'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxxnew[k] = singleColumn
                            print("dfxxnew: standardscaler snap ", dfxxnew)
                            dfxxnew.to_sql(ax, con=engine, if_exists='replace', index=False)
                        elif mod == "KBinsDiscretizer":
                            kbd = preprocessing.KBinsDiscretizer()
                            mdx = kbd.fit(dfxxnew[k].to_numpy().reshape(-1, 1))
                            md = kbd.transform(dfxxnew[k].to_numpy().reshape(-1, 1))
                            column_transform_pair_list = []
                            column_transform_pair = [k, "kbd"]
                            column_transform_pair_list.append(column_transform_pair)
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            kf = os.path.join(app.config['MODEL_FOLDER'], 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin')
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxxnew = pd.concat([dfxxnew, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxxnew = dfxxnew.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxxnew: kbinsdiscretizer snap ", dfxxnew)
                            dfxxnew.to_sql(ax, con=engine, if_exists='replace', index=False)


                print("column_transform_pair_list: ", column_transform_pair_list)
                
##            return render_template("preprocessing1.html", listoftables=listoftables, tablename=tablename, table1 = [df.to_html(classes = 'data')])
                return render_template("preprocessing1.html", listoftables=myinput, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                 tablenamex=x_and_y_list, tablenamey=ay, columnames_x=columnames_x, columnames_y=columnames_y, df_tableX = [dfxxnew.to_html(classes='data table-striped', escape=False)], df_tableY = [dfxynew.to_html(classes='data table-striped', escape=False)],#xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                 labelstrXtrain="X", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny, labelstrYtrain="y",
                                                                                 column_transform_pair_list=column_transform_pair_list, markbinfile=markbinfile) # labelstrXtest="X Test", labelstrYtest="y Test")
        
        elif request.form.get("processy1") == "Fit Transform":

                
                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer()
         

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']
                
                testsize = request.form['testsize']
                
            

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                

                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])

                markbinfile = ""



                for k in dfxy.columns:
                    if selectedcolumny == k:
                        from sklearn import preprocessing
                        if mod == "LabelEncoder":

                            le = preprocessing.LabelEncoder()

                            lefity = le.fit(dfxy[k].to_numpy())

                         
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                      

                            markbinfile= 'LabelEncoder'+get_datetime()+alphanumeric+'.bin'
                           
                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(lefity, kf, compress=True)

                            md = lefity.transform(dfxy[k].to_numpy())

                            column_transform_pair_list = []
                            column_transform_pair = [k, "le"]
                            column_transform_pair_list.append(column_transform_pair)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxy[k] = singleColumn
                            print("dfxy: labelencoder snap ", dfxy)
                            dfxy.to_sql(ay, con=engine, if_exists='replace', index=False)

                       
                            
                        elif mod == "OneHotEncoder":
                            
                            ohe = preprocessing.OneHotEncoder()

                            mdx = ohe.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = ohe.transform(dfxy[k].to_numpy().reshape(-1, 1))

                            column_transform_pair_list = []
                            column_transform_pair = [k, "ohe"]
                            column_transform_pair_list.append(column_transform_pair)
                            
                          
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile = 'OneHotEncoder'+get_datetime()+alphanumeric+'.bin'
                            
                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []

                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxy = pd.concat([dfxy, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxy = dfxy.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxy: onehotencoder snap ", dfxy)
                            dfxy.to_sql(ay, con=engine, if_exists='replace', index=False)
                            
                        elif mod == "LabelBinarizer":

                            lb = preprocessing.LabelBinarizer()


                            mdx = lb.fit(dfxy[k].to_numpy())
                            md = lb.transform(dfxy[k].to_numpy())

                            column_transform_pair_list = []
                            column_transform_pair = [k, "lb"]
                            column_transform_pair_list.append(column_transform_pair)
    
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile = 'LabelBinarizer'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            dfxy = pd.concat([dfxy, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxy = dfxy.drop(columns=k, axis=1)
                            print("dfxy: labelbinarizer snap ", dfxy)
                            dfxy.to_sql(ay, con=engine, if_exists='replace', index=False)
                            
                        elif mod == "MinMaxScaler":

                            mm = preprocessing.MinMaxScaler()

                            mdx = mm.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = mm.transform(dfxy[k].to_numpy().reshape(-1, 1))

                            column_transform_pair_list = []
                            column_transform_pair = [k, "mm"]
                            column_transform_pair_list.append(column_transform_pair)

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile = 'MinMaxScaler'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxy[k] = singleColumn
                            print("dfxy: minmax snap ", dfxy)
                            dfxy.to_sql(ay, con=engine, if_exists='replace', index=False)
                            
                        elif mod == "StandardScaler":

                            ss = preprocessing.StandardScaler()

                            mdx = ss.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = ss.transform(dfxy[k].to_numpy().reshape(-1, 1))

                            column_transform_pair_list = []
                            column_transform_pair = [k, "ss"]
                            column_transform_pair_list.append(column_transform_pair)

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile = 'StandardScaler'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                            
                            singleColumn = pd.DataFrame(md)
                            dfxy[k] = singleColumn
                            print("dfxy: standardscaler snap ", dfxy)
                            dfxy.to_sql(ay, con=engine, if_exists='replace', index=False)
                            
                        elif mod == "KBinsDiscretizer":

                            kbd = preprocessing.KBinsDiscretizer()

                            mdx = kbd.fit(dfxy[k].to_numpy().reshape(-1, 1))
                            md = kbd.transform(dfxy[k].to_numpy().reshape(-1, 1))

                            column_transform_pair_list = []
                            column_transform_pair = [k, "kbd"]
                            column_transform_pair_list.append(column_transform_pair)

                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()

                            markbinfile = 'KBinsDiscretizer'+get_datetime()+alphanumeric+'.bin'

                            kf = os.path.join(app.config['MODEL_FOLDER'], markbinfile)
                            dump(mdx, kf, compress=True)
                                    
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            dfxy = pd.concat([dfxy, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            dfxy = dfxy.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("dfxy: onehotencoder snap ", dfxy)
                            dfxy.to_sql(ay, con=engine, if_exists='replace', index=False)

       
                print("column_transform_pair_list: ", column_transform_pair_list)

                return render_template("preprocessing1.html", listoftables=myinput, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, ylabel="[Target/Label Y]", xlabel="[INPUT X]",
                                                                                 tablenamex=x_and_y_list, tablenamey=ay, columnames_x=columnames_x, columnames_y=columnames_y, df_tableX = [dfxx.to_html(classes='data table-striped', escape=False)],
                                                                                 df_tableY = [dfxy.to_html(classes='data table-striped', escape=False)], column_transform_pair_list=column_transform_pair_list,#xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                 labelstrXtrain="X", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny, labelstrYtrain="y",
                                                                                 markbinfile=markbinfile)
           

        elif request.form.get("resetx1") == "RESET":

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer()

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']
                
                testsize = request.form['testsize']

                markbinfile = request.form['binfilename']

       

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                


                querytablex = "SELECT * FROM {}".format(ax+'2')
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay+'2')
                df_y = pd.read_sql_query(querytabley, con=engine)

               
                df_x.to_sql(ax, con=engine, if_exists='replace', index=False)
                df_y.to_sql(ay, con=engine, if_exists='replace', index=False)
            
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                return render_template("preprocessing1.html", listoftables=myinput, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[INPUT X]", ylabel="[Target/Label Y]",
                                                                                           tablenamex=x_and_y_list, tablenamey=ay, columnames_x=columnames_x, columnames_y=columnames_y, df_tableX = [dfxx.to_html(classes='data table-striped', escape=False)],
                                                                                           df_tableY = [dfxy.to_html(classes='data table-striped', escape=False)], markbinfile=markbinfile, #xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                           labelstrXtrain="X", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny) #, labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

                
        
        elif request.form.get("splitdata1") == "Split Data":

                testsize = request.form['testsize']
            
                mod = request.form['transformmodel']

                markbinfile = request.form['binfilename']


                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                
                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)
                

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                #for c in selectedcolumnsX:
                #    print(c.index)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                
               
                X = dfxx
                print("X shape: ", X, X.shape)

           
                y = dfxy
                print("y shape: ", y, y.shape)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=6)
                print("Split shapes: ", X_train.shape,X_test.shape,y_train.shape,y_test.shape)
                xtraindf = pd.DataFrame(X_train)
                ytraindf = pd.DataFrame(y_train)
                xtestdf = pd.DataFrame(X_test)
                ytestdf = pd.DataFrame(y_test)

                xtrainname = ax + "xtr"
                ytrainname = ay + "ytr"
                xtestname = ax + "xte"
                ytestname = ay + "yte"

                insertstatement = """ UPDATE xymarker SET dateandtime='%s',
                                                                              mark_x_train='%s',
                                                                              mark_x_test='%s',
                                                                              mark_y_train='%s',
                                                                              mark_y_test='%s',
                                                                              username='%s',
                                                                              transform_y='%s'
                                                                              WHERE mark_xy='%s' """ % \
                                                                               (insertdatetime(), xtrainname, xtestname, ytrainname, ytestname, "tommy", markbinfile, ax.replace('inputx', '')) #tabname rm substing
                cur.execute(insertstatement)
                db.commit()

                xtraindf.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                ytraindf.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                xtestdf.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                ytestdf.to_sql(ytestname, con=engine, if_exists='replace', index=False)

                xtrainname2 = ax + "xtr2"
                ytrainname2 = ay + "ytr2"
                xtestname2 = ax + "xte2"
                ytestname2 = ay + "yte2"

                xtraindf.to_sql(xtrainname2, con=engine, if_exists='replace', index=False)
                ytraindf.to_sql(ytrainname2, con=engine, if_exists='replace', index=False)
                xtestdf.to_sql(xtestname2, con=engine, if_exists='replace', index=False)
                ytestdf.to_sql(ytestname2, con=engine, if_exists='replace', index=False)

           
                
                return render_template("preprocessing1.html", listoftables=myinput, testsize=testsize, columnames_x=columnames_x, columnames_y=columnames_y, transformmodel=mod,
                                                                      xtrain = [xtraindf.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                      ytrain = [ytraindf.to_html(classes = 'data')], labelstrYtrain="y Train", markbinfile=markbinfile,
                                                                      xtest = [xtestdf.to_html(classes = 'data')], labelstrXtest="X Test", tablenamex=x_and_y_list, tablenamey=ay,
                                                                      ytest = [ytestdf.to_html(classes = 'data')], labelstrYtest="y Test", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny)


        elif request.form.get("resetsplit1") == "RESET":

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer()

                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']
                
                mod = request.form['transformmodel']

                markbinfile = request.form['binfilename']
                
                testsize = request.form['testsize']

          

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                


                querytablex = "SELECT * FROM {}".format(ax+'2')
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay+'2')
                df_y = pd.read_sql_query(querytabley, con=engine)
           
                xtrainname = ax + "xtr2"
                ytrainname = ay + "ytr2"
                xtestname = ax + "xte2"
                ytestname = ay + "yte2"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                xtrainname0 = ax + "xtr"
                ytrainname0 = ay + "ytr"
                xtestname0 = ax + "xte"
                ytestname0 = ay + "yte"

                df_xtr.to_sql(xtrainname0, con=engine, if_exists='replace', index=False)
                df_ytr.to_sql(ytrainname0, con=engine, if_exists='replace', index=False)
                df_xte.to_sql(xtestname0, con=engine, if_exists='replace', index=False)
                df_yte.to_sql(ytestname0, con=engine, if_exists='replace', index=False)
                
                columnames_x = []
                for xcol in df_x.columns:
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)

                combix = []
                combiy = []
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                return render_template("preprocessing1.html", listoftables=myinput, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[INPUT X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #df_tableX = [dfxx.to_html(classes='data table-striped', escape=False)], df_tableY = [dfxy.to_html(classes='data table-striped', escape=False)], #xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train", markbinfile=markbinfile,
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test", tablenamex=x_and_y_list, tablenamey=ay,
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test") #, labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

        
        elif request.form.get("processxtrain1") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

                
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumnx_xtrain = request.form['selectedcolumnx_xtrain']
                
            

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]
                
                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)

                xtrainname = ax + "xtr"
                ytrainname = ay + "ytr"
                xtestname = ax + "xte"
                ytestname = ay + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_xtr.columns:
                    if selectedcolumnx_xtrain == k:
                        if modsp == "LabelEncoder":
                            #global lefitx
                            
                            lefitx = le.fit(df_xtr[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefitx, kf, compress=True)
                            #pickle.dump(lefit, open(kf, 'wb'))
                            
                            md = lefitx.transform(df_xtr[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_xtr[k] = singleColumn
                            print("df_xtr: labelencoder snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))
                          

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xtr = pd.concat([df_xtr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xtr = df_xtr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xtr: onehotencoder snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.fit_transform(df_xtr[k].to_numpy())
                            

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_xtr = pd.concat([df_xtr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xtr = df_xtr.drop(columns=k, axis=1)
                            print("df_xtr: labelbinarizer snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_xtr[k] = singleColumn
                            print("df_xtr: minmax snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_xtr[k] = singleColumn
                            print("df_xtr: standardscaler snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.fit_transform(df_xtr[k].to_numpy().reshape(-1, 1))
                         
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xtr = pd.concat([df_xtr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xtr = df_xtr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xtr: kbinsdiscretizer snap ", df_xtr)
                            df_xtr.to_sql(xtrainname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing1.html", listoftables=myinput, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train", selectedcolumnx_xtrain=selectedcolumnx_xtrain,
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny,
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test", tablenamex=x_and_y_list, tablenamey=ay)
                                                                                 

        elif request.form.get("processxtest1") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

            
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumnx_xtest = request.form['selectedcolumnx_xtest']

           

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]
                
                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                
         

                xtrainname = ax + "xtr"
                ytrainname = ay + "ytr"
                xtestname = ax + "xte"
                ytestname = ay + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_xte.columns:
                    if selectedcolumnx_xtest == k:
                        if modsp == "LabelEncoder":
                            #lefit = le.fit(df_xte[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefitx, kf, compress=True)
                            #pickle.dump(lefit, open(kf, 'wb'))
                            
                            md = lefitx.transform(df_xte[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_xte[k] = singleColumn
                            print("df_xte: labelencoder snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xte = pd.concat([df_xte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xte = df_xte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xte: onehotencoder snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.transform(df_xte[k].to_numpy())
                            

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_xte = pd.concat([df_xte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xte = df_xte.drop(columns=k, axis=1)
                            print("df_xte: labelbinarizer snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_xte[k] = singleColumn
                            print("df_xte: minmax snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_xte[k] = singleColumn
                            print("df_xte: standardscaler snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.transform(df_xte[k].to_numpy().reshape(-1, 1))
                            

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_xte = pd.concat([df_xte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_xte = df_xte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_xte: kbinsdiscretizer snap ", df_xte)
                            df_xte.to_sql(xtestname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing1.html", listoftables=myinput, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny,
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test", selectedcolumnx_xtest=selectedcolumnx_xtest,
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test", tablenamex=x_and_y_list, tablenamey=ay)

        elif request.form.get("processytrain1") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumny_ytrain = request.form['selectedcolumny_ytrain']

                #ax = request.form['tablenamex']
                #ay = request.form['tablenamey']

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]
                
                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                
                

                xtrainname = ax + "xtr"
                ytrainname = ay + "ytr"
                xtestname = ax + "xte"
                ytestname = ay + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_ytr.columns:
                    if selectedcolumny_ytrain == k:
                        if modsp == "LabelEncoder":
                            #global lefity
                            
                            lefity = le.fit(df_ytr[k].to_numpy())

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefity, kf, compress=True)
                            #pickle.dump(lefit, open(kf, 'wb'))
                            
                            md = lefity.transform(df_ytr[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_ytr[k] = singleColumn
                            print("df_ytr: labelencoder snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))
                            

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_ytr = pd.concat([df_ytr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_ytr = df_ytr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_ytr: onehotencoder snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.fit_transform(df_ytr[k].to_numpy())
                           

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_ytr = pd.concat([df_ytr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_ytr = df_ytr.drop(columns=k, axis=1)
                            print("df_ytr: labelbinarizer snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_ytr[k] = singleColumn
                            print("df_ytr: minmax snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_ytr[k] = singleColumn
                            print("df_ytr: standardscaler snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.fit_transform(df_ytr[k].to_numpy().reshape(-1, 1))
                            

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_ytr = pd.concat([df_ytr, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_ytr = df_ytr.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_ytr: kbinsdiscretizer snap ", df_ytr)
                            df_ytr.to_sql(ytrainname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing1.html", listoftables=myinput, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train", selectedcolumny_ytrain=selectedcolumny_ytrain,
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny,
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test", tablenamex=x_and_y_list, tablenamey=ay)

        elif request.form.get("processytest1") == "Fit Transform":

                from sklearn.preprocessing import KBinsDiscretizer
                kbd = KBinsDiscretizer()

                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                modsp = request.form['transformmodelsp']

                selectedcolumny_ytest = request.form['selectedcolumny_ytest']

                #ax = request.form['tablenamex']
                #ay = request.form['tablenamey']

                tabx = request.form['tablenamex']
                x_and_y_list = ast.literal_eval(tabx)
                ax = x_and_y_list[0]
                ay = x_and_y_list[1]
                
                selectedcolumnx = request.form['selectedcolumnx']
                selectedcolumny = request.form['selectedcolumny']

                selecte = request.form.get('tnames')
                myinput = ast.literal_eval(selecte)
                
             

                xtrainname = ax + "xtr"
                ytrainname = ay + "ytr"
                xtestname = ax + "xte"
                ytestname = ay + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                querytablex = "SELECT * FROM {}".format(ax)
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ay)
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]

                md = np.empty([2, 2])



                for k in df_yte.columns:
                    if selectedcolumny_ytest == k:
                        if modsp == "LabelEncoder":
                            #lefity = le.fit(df_yte[k].to_numpy())
                            
                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                                    
                            kf = os.path.join(app.config['MODEL_FOLDER'], 'LabelEncoder'+get_datetime()+alphanumeric+'.bin')
                            dump(lefity, kf, compress=True)
                            #pickle.dump(lefit, open(kf, 'wb'))
                            
                            md = lefity.transform(df_yte[k].to_numpy())
                            
                            singleColumn = pd.DataFrame(md)
                            df_yte[k] = singleColumn
                            print("df_yte: labelencoder snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "OneHotEncoder":
                            md = ohe.transform(df_yte[k].to_numpy().reshape(-1, 1))
                           

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_yte = pd.concat([df_yte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_yte = df_yte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_yte: onehotencoder snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "LabelBinarizer":
                            md = lb.transform(df_yte[k].to_numpy())
                          

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md, columns=strcol)
                            df_yte = pd.concat([df_yte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_yte = df_yte.drop(columns=k, axis=1)
                            print("df_yte: labelbinarizer snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "MinMaxScaler":
                            md = mm.transform(df_yte[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_yte[k] = singleColumn
                            print("df_yte: minmax snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "StandardScaler":
                            md = ss.transform(df_yte[k].to_numpy().reshape(-1, 1))
                            singleColumn = pd.DataFrame(md)
                            df_yte[k] = singleColumn
                            print("df_yte: standardscaler snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)
                        elif modsp == "KBinsDiscretizer":
                            md = kbd.transform(df_yte[k].to_numpy().reshape(-1, 1))
                           

                            alphanumeric = ""
                            for character in k:
                                if character.isalnum():
                                    alphanumeric += character.lower()
                            n = md.shape[1]
                            strcol = []
                            for i in range(n):
                                strcol.append(alphanumeric + str(i))
                            singleColumn = pd.DataFrame(md.toarray(), columns=strcol)
                            df_yte = pd.concat([df_yte, singleColumn], axis=1)
                            #print("totdf: ", totdf)
                            df_yte = df_yte.drop(columns=k, axis=1)
                            #dfxxnew[k] = singleColumn
                            print("df_yte: kbinsdiscretizer snap ", df_yte)
                            df_yte.to_sql(ytestname, con=engine, if_exists='replace', index=False)

                return render_template("preprocessing1.html", listoftables=myinput, testsize=testsize, transformmodelsp=modsp, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, #xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                xtrain = [df_xtr.to_html(classes = 'data')], labelstrXtrain="X Train",
                                                                                ytrain = [df_ytr.to_html(classes = 'data')], labelstrYtrain="y Train", selectedcolumny_ytest=selectedcolumny_ytest,
                                                                                xtest = [df_xte.to_html(classes = 'data')], labelstrXtest="X Test", selectedcolumnx=selectedcolumnx, selectedcolumny=selectedcolumny,
                                                                                ytest = [df_yte.to_html(classes = 'data')], labelstrYtest="y Test", tablenamex=x_and_y_list, tablenamey=ay)

                
            
        elif request.form.get("fitdata") == "Fit Data":
                
                
                testsize = request.form['testsize']
                mod = request.form['transformmodel']
               

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                # ----------------------------------------------------------
        
                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]
             
                X = dfxx[0:int(len(dfxx)/2)]
                print(X, X.shape)
                y = dfxy[int(len(dfxy)/2)+1:len(dfxy)]
                print(y, y.shape)

                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=6)

                # ----------------------------------------------------------

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                xtraindf = pd.DataFrame(X_train)
                ytraindf = pd.DataFrame(y_train)
                xtestdf = pd.DataFrame(X_test)
                ytestdf = pd.DataFrame(y_test)

                xtraindf.columns = columnames_x
                ytraindf.columns = columnames_y
                xtestdf.columns = columnames_x
                ytestdf.columns = columnames_y
                
                if mod=="MinMaxScaler":
                    mm.fit(X_train)
                    
                    model_min = mm.data_min_
                    model_max = mm.data_max_
                    model_range = mm.data_range_
                    model_feat = mm.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_min, model_max, model_range, model_feat]
                    attrnames = ['Min', 'Max', 'Range', 'Features']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    mm.fit(X_train)
                    mm_xtrain = mm.transform(X_train)
                    print("mm_xtrain: ", mm_xtrain)
                    mm_xtest = mm.transform(X_test)
                    print("mm_xtest: ", mm_xtest)

                    #----------------------------

                    mm.fit(y_train)
                
                    model_min = mm.data_min_
                    model_max = mm.data_max_
                    model_range = mm.data_range_
                    model_feat = mm.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_min, model_max, model_range, model_feat]
                    attrnames = ['Min', 'Max', 'Range', 'Features']

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)

                    # ----------------------------
                    
                    xtraindf = pd.DataFrame(X_train)
                    ytraindf = pd.DataFrame(y_train)
                    xtestdf = pd.DataFrame(X_test)
                    ytestdf = pd.DataFrame(y_test)
                                    
                    mm.fit(y_train)
                    mm_ytrain = mm.transform(y_train)
                    print("mm_ytrain: ", mm_ytrain)
                    mm_ytest = mm.transform(y_test)
                    print("mm_ytest: ", mm_xtest)

                    return render_template("preprocessing1.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=mm_xtrain, xtest=mm_xtest, ytrain=mm_ytrain, ytest=mm_ytest,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")
##                                                                                xtrain = [xtraindf.to_html(classes = 'data')], labelstrXtrain="X Train",
##                                                                                ytrain = [ytraindf.to_html(classes = 'data')], labelstrYtrain="y Train",
##                                                                                xtest = [xtestdf.to_html(classes = 'data')], labelstrXtest="X Test",
##                                                                                ytest = [ytestdf.to_html(classes = 'data')], labelstrYtest="y Test")
                                                                   
                                                                  

                if mod=="StandardScaler":
                
                    ss.fit(X_train)
                    
                    model_mean = ss.mean_
                    model_var = ss.var_
                    model_feat = ss.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_mean, model_var, model_feat]
                    attrnames = ['Mean', 'Variance', 'Features']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    ss.fit(X_train)
                    ss_xtrain = ss.transform(X_train)
                    print("ss_xtrain: ", ss_xtrain)
                    ss_xtest = ss.transform(X_test)
                    print("ss_xtest: ", ss_xtest)

                    #----------------------------

                    ss.fit(y_train)
                
                    model_mean = ss.mean_
                    model_var = ss.var_
                    model_feat = ss.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_mean, model_var, model_feat]
                    attrnames = ['Mean', 'Variance', 'Features']

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)

                    
                    
                    ss.fit(y_train)
                    ss_ytrain = ss.transform(y_train)
                    print("ss_ytrain: ", ss_ytrain)
                    ss_ytest = ss.transform(y_test)
                    print("ss_ytest: ", ss_xtest)

                
                    return render_template("preprocessing1.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=ss_xtrain, xtest=ss_xtest, ytrain=ss_ytrain, ytest=ss_ytest,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

                if mod=="OneHotEncoder":

                    ohe.fit(X_train)
                    
                    model_cat = ohe.categories_
                    #model_featnamein = ohe.feature_names_in_
                    #model_feat = ohe.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_cat]
                    attrnames = ['Categories']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    #ohe.fit(X_train)
                    ohe_xtrain = ohe.transform(X_train)
                    print("ohe_xtrain: ", ohe_xtrain)

                    ohe.fit(X_test)
                    ohe_xtest = ohe.transform(X_test)
                    print("ohe_xtest: ", ohe_xtest)

                    #----------------------------

                    ohe.fit(y_train)
                
                    model_cat = ohe.categories_
                    #model_featnamein = ohe.feature_names_in_
                    #model_feat = ohe.n_features_in_
                    #model_seen= mm.n_samples_seen_
                    #model_names = mm.feature_names_in_
                    modelattr = [model_cat]
                    attrnames = ['Categories']

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)

                    #ohe.fit(y_train)
                    ohe_ytrain = ohe.transform(y_train)
                    print("ohe_ytrain: ", ohe_ytrain)

                    ohe.fit(y_test)
                    ohe_ytest = ohe.transform(y_test)
                    print("ohe_ytest: ", ohe_xtest)

                
                    return render_template("preprocessing1.html", listoftables=[x_and_y_list], tablename=x_and_y_list, modelattrx=combix, modelattry=combiy, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=ohe_xtrain, xtest=ohe_xtest, ytrain=ohe_ytrain, ytest=ohe_ytest,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test")

                if mod=="LabelEncoder":

                    testsize = request.form['testsize']
                    #tablename = request.form['tablename']

                    x_and_y = request.form['tablename']
                    print("PTNS: ", x_and_y, type(x_and_y))
                    #x_and_y_list = x_and_y.strip('][').split(', ')
                    x_and_y_list = ast.literal_eval(x_and_y)
                    print("to list: ", x_and_y_list[0], x_and_y_list[1])


                    
                    querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                    df_x = pd.read_sql_query(querytablex, con=engine)

                    querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                    df_y = pd.read_sql_query(querytabley, con=engine)
       

                    columnames_x = []
                    for xcol in df_x.columns:
                        #print(col)
                        if xcol != "Edit" and xcol != "Delete":
                            columnames_x.append(xcol)

                    columnames_y = []
                    for ycol in df_y.columns:
                        #print(col)
                        if ycol != "Edit" and ycol != "Delete":
                            columnames_y.append(ycol)

                    #for c in selectedcolumnsX:
                    #    print(c.index)
                    
                    dfxx = df_x[columnames_x]
                    dfxy = df_y[columnames_y]

                    
       
                    
                    X = dfxx[0:int(len(dfxx)/2)]
                    print(X, X.shape)
                    y = dfxy[int(len(dfxy)/2)+1:len(dfxy)]
                    print(y, y.shape)
                    while X.shape[0] != y.shape[0]:
                        s = int(len(dfxx)/2)
                        s = s - 1
                        X = dfxx[0:s]
                        
               

                    selectedcolumn = request.form['selectedcolumn']
                    scol = dfxx[selectedcolumn].to_numpy()

                    #dfxy

                    transdat = le.fit_transform(scol)
                    

                    return render_template("preprocessing1.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                columnames_x=columnames_x, columnames_y=columnames_y, xtrain=transdat,
                                                                                 labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test", selectedcolumn=selectedcolumn)

                

                 
                if mod=="KBinsDiscretizer":
                
                    from sklearn.preprocessing import KBinsDiscretizer

                    kbd = KBinsDiscretizer()

                    # ---------------------------------------------

                    kbd.fit(X_train)
                
                    model_nbins = kbd.n_bins_
                    model_nfeatures_in = kbd.n_features_in_
                    modelattr = [model_nbins, model_nfeatures_in]
                    attrnames = ['Bins per Feature', 'Features']

                    encoderandstrategyflag = True

                    encode_method = ['onehot', 'onehot-dense', 'ordinal']
                    strategy_used = ['quantile', 'uniform', 'kmeans']

                    combix = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combix.append(strr)

                    
                    kbd_xtrain = kbd.transform(X_train)
                    print("kbd_xtrain: ", kbd_xtrain)
                    kbd_xtest = kbd.transform(X_test)
                    print("kbd_xtest: ", kbd_xtest)

                    # ----------------------------------------------

                    kbd.fit(y_train)
                
                    model_nbins = kbd.n_bins_
                    model_nfeatures_in = kbd.n_features_in_
                    modelattr = [model_nbins, model_nfeatures_in]
                    attrnames = ['Bins per Feature', 'Features']

                    encoderandstrategyflag = True

                    combiy = []

                    for a, b in zip(modelattr, attrnames):
                            strr = b + ": " + str(a)
                            combiy.append(strr)
                    
                    
                    kbd_ytrain = kbd.transform(y_train)
                    print("kbd_ytrain: ", kbd_ytrain)
                    kbd_ytest = kbd.transform(y_test)
                    print("kbd_ytest: ", kbd_xtest)

                    return render_template("preprocessing1.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                                 columnames_x=columnames_x, columnames_y=columnames_y, modelattrx=combix, modelattry=combiy,
                                                                                 #labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test",
                    #return render_template("preprocessing1.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize,
                                                                                  xtrain = kbd_xtrain, labelstrXtrain="X Train",
                                                                                  ytrain = kbd_ytrain, labelstrYtrain="y Train",
                                                                                  xtest = kbd_xtest, labelstrXtest="X Test", encode_methods=encode_method, strategies_used=strategy_used,
                                                                                  ytest = kbd_ytest, labelstrYtest="y Test", encoderandstrategyflag=encoderandstrategyflag)

       
                    
        elif request.form.get("get_encodeandstrategy") == "Select Encoder and Strategy":
                
                #tablename = request.form['tablename']
                testsize = request.form['testsize']
                mod = request.form['transformmodel']
                #queryalltables = """SELECT * FROM {}""".format(tablename)
                #df = pd.read_sql_query(queryalltables, con=postgres_db, index_col=None)

                x_and_y = request.form['tablename']
                print("PTNS: ", x_and_y, type(x_and_y))
                #x_and_y_list = x_and_y.strip('][').split(',')
                x_and_y_list = ast.literal_eval(x_and_y)
                print("to list: ", x_and_y_list[0], type(x_and_y_list), type(x_and_y_list[0]))

                xtrainname = x_and_y_list[0] + "xtr"
                ytrainname = x_and_y_list[1] + "ytr"
                xtestname = x_and_y_list[0] + "xte"
                ytestname = x_and_y_list[1] + "yte"

                querytablex = "SELECT * FROM {}".format(xtrainname)
                df_xtr = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytrainname)
                df_ytr = pd.read_sql_query(querytabley, con=engine)

                querytablex = "SELECT * FROM {}".format(xtestname)
                df_xte = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(ytestname)
                df_yte = pd.read_sql_query(querytabley, con=engine)

                # ----------------------------------------------------------
        
                querytablex = "SELECT * FROM {}".format(x_and_y_list[0])
                df_x = pd.read_sql_query(querytablex, con=engine)

                querytabley = "SELECT * FROM {}".format(x_and_y_list[1])
                df_y = pd.read_sql_query(querytabley, con=engine)

                columnames_x = []
                for xcol in df_x.columns:
                    #print(col)
                    if xcol != "Edit" and xcol != "Delete":
                        columnames_x.append(xcol)

                columnames_y = []
                for ycol in df_y.columns:
                    #print(col)
                    if ycol != "Edit" and ycol != "Delete":
                        columnames_y.append(ycol)
                
                dfxx = df_x[columnames_x]
                dfxy = df_y[columnames_y]
             
                X = dfxx[0:int(len(dfxx)/2)]
                print(X, X.shape)
                y = dfxy[int(len(dfxy)/2)+1:len(dfxy)]
                print(y, y.shape)
##                while X.shape[0] != y.shape[0]:
##                    s = int(len(dfxx)/2)
##                    s = s - 1
##                    X = dfxx[0:s]
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=6)

                # ----------------------------------------------------------

                X_train = df_xtr.to_numpy()
                y_train = df_ytr.to_numpy()
                X_test = df_xte.to_numpy()
                y_test = df_yte.to_numpy()

                # ------------------------------------------------

                enc = request.form['encode_method']
                strat = request.form['strategy_used']

                print("anc & strat: ", enc, strat)

                from sklearn.preprocessing import KBinsDiscretizer

                kbd = KBinsDiscretizer(encode=enc, strategy=strat)

                # ---------------------------------------------------------

                encoderandstrategyflag = False

                kbd.fit(X_train)
                    
                model_nbins = kbd.n_bins_
                model_nfeatures_in = kbd.n_features_in_
                modelattr = [model_nbins, model_nfeatures_in]
                attrnames = ['Bins per Feature', 'Features']

                encode_method = ['onehot', 'onehot-dense', 'ordinal']
                strategy_used = ['quantile', 'uniform', 'kmeans']

                combix = []

                for a, b in zip(modelattr, attrnames):
                        strr = b + ": " + str(a)
                        combix.append(strr)

                kbd_xtrain = kbd.transform(X_train)
                print("kbd_xtrain: ", kbd_xtrain)
                kbd_xtest = kbd.transform(X_test)
                print("kbd_xtest: ", kbd_xtest)

                # ----------------------------------------------

                kbd.fit(y_train)
            
                model_nbins = kbd.n_bins_
                model_nfeatures_in = kbd.n_features_in_
                modelattr = [model_nbins, model_nfeatures_in]
                attrnames = ['Bins per Feature', 'Features']

                combiy = []

                for a, b in zip(modelattr, attrnames):
                        strr = b + ": " + str(a)
                        combiy.append(strr)

                kbd_ytrain = kbd.transform(y_train)
                print("kbd_ytrain: ", kbd_ytrain)
                kbd_ytest = kbd.transform(y_test)
                print("kbd_ytest: ", kbd_xtest)

                return render_template("preprocessing1.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize, transformmodel=mod, xlabel="[Input X]", ylabel="[Target/Label Y]",
                                                                             columnames_x=columnames_x, columnames_y=columnames_y, modelattrx=combix, modelattry=combiy,
                                                                             #labelstrXtrain="X Train", labelstrYtrain="y Train", labelstrXtest="X Test", labelstrYtest="y Test",
                #return render_template("preprocessing.html", listoftables=[x_and_y_list], tablename=x_and_y_list, testsize=testsize,
                                                                              xtrain = kbd_xtrain, labelstrXtrain="X Train",
                                                                              ytrain = kbd_ytrain, labelstrYtrain="y Train",
                                                                              xtest = kbd_xtest, labelstrXtest="X Test", encode_methods=encode_method, strategies_used=strategy_used,
                                                                              ytest = kbd_ytest, labelstrYtest="y Test", encoderandstrategyflag=encoderandstrategyflag)

            


    flash("Response error.")
    return render_template("preprocessing1.html") #, listoftables=listoftables)



    


def minmaxscale_fitOnly(numpydata):
    mm.fit(numpydata)
    return mm

def minmaxscale_fitTransform(numpydata):
    out = mm.fit_transform(numpydata)
    return out

# for X_test set only
def minmaxscale_transform(numpydata):
    out = mm.transform(numpydata)
    return out

# -----------------------------------------
def standardscale_fitOnly(numpydata):
    ss.fit(numpydata)
    return ss

def standardscale_fitTransform(numpydata):
    out = ss.fit_transform(numpydata)
    return out

# for X_test set only
def standardscale_transform(numpydata):
    out = ss.transform(numpydata)
    return out

# -----------------------------------------
def labelencode_fitOnly(numpydata):
    le.fit(numpydata)
    return le

def labelencode_fitTransform(numpydata):
    out = le.fit_transform(numpydata)
    return out

# for X_test set only
def labelencode_transform(numpydata):
    out = le.transform(numpydata)
    return out

# -----------------------------------------
def onehotencode_fitOnly(numpydata):
    ohe.fit(numpydata)
    return ohe

def onehotencode_fitTransform(numpydata):
    out = ohe.fit_transform(numpydata)
    return out

# for X_test set only
def onehotencode_transform(numpydata):
    out = ohe.transform(numpydata)
    return out

# -----------------------------------------
def kbinsdiscretizer_fitOnly(numpydata):
    kbd.fit(numpydata)
    return kbd

def kbinsdiscretizer_fitTransform(numpydata):
    out = kbd.fit_transform(numpydata)
    return out

# for X_test set only
def kbinsdiscretizer_transform(numpydata):
    out = kbd.transform(numpydata)
    return out

# -----------------------------------------
def labelbinarizer_fitOnly(numpydata):
    lb.fit(numpydata)
    return lb

def labelbinarizer_fitTransform(numpydata):
    out = lb.fit_transform(numpydata)
    return out

# for X_test set only
def labelbinarizer_transform(numpydata):
    out = lb.transform(numpydata)
    return out

# -----------------------------------------



if __name__ == "__main__":
    
    app.run(debug=True, use_reloader=False, host="0.0.0.0")
