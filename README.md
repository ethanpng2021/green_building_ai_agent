# Flask App for Machine Learning
## Set Up: 
Change these lines to your PC/laptop/server environment:
```python
path_rm = "G:\\Dropbox\\work2\\home\\aigood\\static\\models\\"
UPLOAD_FOLDER = "G:\\Dropbox\\work2\\home\\aigood\\static\\uploads"
DOWNLOAD_FOLDER = "G:\\Dropbox\\work2\\home\\aigood\\static\\downloads"
MODEL_FOLDER = "G:\\Dropbox\\work2\\home\\aigood\\static\\models"

engine = create_engine('postgresql+psycopg2://postgres:xxx@localhost:5432/mydb')
db = psycopg2.connect(database="mydb", user='postgres', password='xxx', host='localhost', port= '5432')
cur = db.cursor()
```
