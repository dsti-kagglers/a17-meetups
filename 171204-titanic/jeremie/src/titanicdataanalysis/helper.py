import os
from time import gmtime, strftime

def save_model(modelPath='model/', column_names='error',
               score=0, model='error no model', comments='', logLevel="DEBUG"):
    file_name = modelPath+"results.csv"

    if os.path.exists(file_name):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    result_file = open(file_name,append_write)
    if append_write == 'w':
        result_file.write("Time;Score;Columns;Model;Comments" + '\n')

    result= strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ';' +str(score)+';'+''.join(map(str, column_names))+';'+''.join(str(model).splitlines())+';'+comments
    result_file.write(result + '\n')
    result_file.close()