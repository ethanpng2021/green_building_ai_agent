# Flask App for Machine Learning
## Set Up: 
Change these lines to your PC/laptop/server environment:
```python
path_rm = "D:\\Dropbox\\work\\home\\ai\\static\\models\\"
UPLOAD_FOLDER = "D:\\Dropbox\\work\\home\\ai\\static\\uploads"
DOWNLOAD_FOLDER = "D:\\Dropbox\\work\\home\\ai\\static\\downloads"
MODEL_FOLDER = "D:\\Dropbox\\work\\home\\ai\\static\\models"

engine = create_engine('postgresql+psycopg2://postgres:xxx@localhost:5432/mydb')
db = psycopg2.connect(database="mydb", user='postgres', password='xxx', host='localhost', port= '5432')
cur = db.cursor()
```

### Create This Table
```console
CREATE TABLE xymarker (
    dateandtime character varying(20),
    mark_xy character varying(250),
    username character varying(250),
    mark_x character varying(250),
    mark_y character varying(250),
    mark_x_train character varying(250),
    mark_x_test character varying(250),
    mark_y_train character varying(250),
    mark_y_test character varying(250),
    mark_x2 character varying(250),
    mark_y2 character varying(250),
    transform_y character varying(250)
);
```

## User Guide
![alt text](https://github.com/ethanpng2021/flask_for_machinelearning/blob/main/machinelearningapp/static/sampleimg/c33.jpg)
![alt text](https://github.com/ethanpng2021/flask_for_machinelearning/blob/main/machinelearningapp/static/sampleimg/c22.jpg)
![alt text](https://github.com/ethanpng2021/flask_for_machinelearning/blob/main/machinelearningapp/static/sampleimg/c11.jpg)
