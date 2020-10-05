import warnings
warnings.filterwarnings("ignore")
import time
import NewsLDA
import uuid
from flask import Flask,abort,jsonify,request,redirect,render_template,url_for
from werkzeug.utils import secure_filename
import json
import pandas as pd
ALLOWED_EXTENSIONS = set(['txt'])


app = Flask(__name__)

dataset = pd.read_csv("all_labels_comparison_with_details.csv",sep='\t',encoding='utf-8')
cluster_lbl= dataset['Cluster']
@app.route('/')
def index():
	row = dataset.iloc[0]
	top10terms = row['Top10Terms'].split(' ')
	top5docs = row['Top5Docs'].split('$')
	return render_template('index.html',
						   title='Home',clusterid='0',
						   top10terms=top10terms,top5docs=top5docs,
						   text = '',cluster_lbl= cluster_lbl,time_exe = '')

                           
@app.route('/cluster/<clusterid>', methods=['GET'])
def cluster_details(clusterid):
    #do your code here
	row = dataset.iloc[int(clusterid)]
	top10terms = row['Top10Terms'].split(' ')
	top5docs = row['Top5Docs'].split('$')
	return render_template('index.html',
						   title='Home',clusterid=clusterid,
						   top10terms=top10terms,top5docs=top5docs,
						   text = '',cluster_lbl= cluster_lbl,time_exe = '')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'file' not in request.files:
	  ##print 'No file part'
	  return redirect('/')
	file = request.files['file']
	# if user does not select file, browser also
	# submit a empty part without filename
	if file.filename == '':
	  ##print 'No selected file'
	  return redirect('/')
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		unique_filename = str(uuid.uuid4())  
			
		f = request.files['file']
		
		text = f.read().decode("utf-8")
		print(text)
		
		start = time.time()
		dataset,cluster_lbl = NewsLDA.apply_kmeans(text)
		end = time.time()
		if len(dataset.index) == 0:
			res = "no"
		else:
			res = "yes"
			
		return render_template('index.html',title='Clustering',cluster_lbl = cluster_lbl,finalsf=dataset,res = res,time_exe = end-start)

            

if __name__ == '__main__':
    app.run(port = 9000,debug = True)