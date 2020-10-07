import warnings
warnings.filterwarnings("ignore")
import time
import uuid
from flask import Flask,abort,jsonify,request,redirect,render_template,url_for
from werkzeug.utils import secure_filename
import json
import pandas as pd
ALLOWED_EXTENSIONS = set(['txt'])


app = Flask(__name__)

dataset = pd.read_csv("all_labels_comparison_with_details.csv",sep='\t',encoding='utf-8')
cluster_lbl= dataset['Cluster']
column_names = ["Cluster", "Zero-Order", "M-Order","T-Order","ProposedMethod"]

rating = pd.DataFrame(columns = column_names)
@app.route('/')
def home():
	
	return render_template('home.html')

@app.route('/index')
def index():
	row = dataset.iloc[0]
	top10terms = row['Top10Terms'].split(' ')
	docids = range(0,5)
	top5docs = row['Top5Docs'].split('$')
	return render_template('index.html',
						   title='Home',clusterid='0',row=row,
						   top10terms=top10terms,top5docs=zip(docids,top5docs),
						   text = '',cluster_lbl= cluster_lbl,time_exe = '')

                           
@app.route('/cluster/<clusterid>', methods=['GET'])
def cluster_details(clusterid):
    #do your code here
	row = dataset.iloc[int(clusterid)]
	top10terms = row['Top10Terms'].split(' ')
	docids = range(0,5)
	top5docs = row['Top5Docs'].split('$')
	return render_template('index.html',
						   title='Home',clusterid=clusterid,row=row,
						   top10terms=top10terms,top5docs=zip(docids,top5docs),
						   text = '',cluster_lbl= cluster_lbl,time_exe = '')
@app.route('/submitdata', methods=['POST'])
def submitdata():
		current_id=int(request.form['clusterid_current'])
		zero_order = [request.form['completeness_method1'],request.form['relevance_method1'],request.form['correctness_method1']]
		m_order = [request.form['completeness_method2'],request.form['relevance_method2'],request.form['correctness_method2']]
		t_order = [request.form['completeness_method3'],request.form['relevance_method3'],request.form['correctness_method3']]
		proposed_method = [request.form['completeness_method4'],request.form['relevance_method4'],request.form['correctness_method4']]
		nextid = int(request.form['clusterid_current']) + 1
		rating.loc[current_id] = [current_id,";".join(zero_order),";".join(m_order),";".join(t_order),";".join(proposed_method)]
		print(rating)
		if nextid>=13:
			unique_filename = str(uuid.uuid4()) 
			rating.to_csv(unique_filename + '.csv', sep='\t',index=False,encoding='utf-8')
			return render_template('thankyou.html')
		else:
			row = dataset.iloc[int(nextid)]
			top10terms = row['Top10Terms'].split(' ')
			docids = range(0,5)
			top5docs = row['Top5Docs'].split('$')
			
			return render_template('index.html',
							   title='Home',clusterid=nextid,row=row,
							   top10terms=top10terms,top5docs=zip(docids,top5docs),
							   text = '',cluster_lbl= cluster_lbl,time_exe = '')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#@app.route('/upload', methods=['POST'])
#def upload_file():
	# check if the post request has the file part
	# if 'file' not in request.files:
	  #print 'No file part'
	  # return redirect('/')
	# file = request.files['file']
	#if user does not select file, browser also
	#submit a empty part without filename
	# if file.filename == '':
	  #print 'No selected file'
	  # return redirect('/')
	# if file and allowed_file(file.filename):
		# filename = secure_filename(file.filename)
		# unique_filename = str(uuid.uuid4())  
			
		# f = request.files['file']
		
		# text = f.read().decode("utf-8")
		# print(text)
		
		# start = time.time()
		# dataset,cluster_lbl = NewsLDA.apply_kmeans(text)
		# end = time.time()
		# if len(dataset.index) == 0:
			# res = "no"
		# else:
			# res = "yes"
			
		# return render_template('index.html',title='Clustering',cluster_lbl = cluster_lbl,finalsf=dataset,res = res,time_exe = end-start)

            

if __name__ == '__main__':
    app.run(port = 9000,debug = True)