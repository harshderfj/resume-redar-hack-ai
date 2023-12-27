import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory,jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import nltk
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import json
from uagents import Agent, Context

agent = Agent(name="upload_agent")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

nltk.download('punkt')
glob_sim = 0.0
def calculate_bleu_score(reference, candidate):
    reference = [nltk.word_tokenize(reference)]
    candidate = nltk.word_tokenize(candidate)
    return nltk.translate.bleu_score.sentence_bleu(reference, candidate)

missing_sk = {}
def bert_similarity(text1, text2):
    # Encode the texts
    encoded_input1 = tokenizer(text1, return_tensors='pt', truncation=True, padding=True, max_length=128)
    encoded_input2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Compute BERT embeddings
    with torch.no_grad():
        model_output1 = model(**encoded_input1)
        model_output2 = model(**encoded_input2)

    # Mean pooling - take attention mask into account for correct averaging
    mask1 = encoded_input1['attention_mask'].unsqueeze(-1).expand(model_output1.last_hidden_state.size()).float()
    mask2 = encoded_input2['attention_mask'].unsqueeze(-1).expand(model_output2.last_hidden_state.size()).float()
    sum_embeddings1 = torch.sum(model_output1.last_hidden_state * mask1, 1)
    sum_embeddings2 = torch.sum(model_output2.last_hidden_state * mask2, 1)
    sum_mask1 = torch.clamp(mask1.sum(1), min=1e-9)
    sum_mask2 = torch.clamp(mask2.sum(1), min=1e-9)
    mean_embeddings1 = sum_embeddings1 / sum_mask1
    mean_embeddings2 = sum_embeddings2 / sum_mask2

    # Compute cosine similarity
    cosine_similarity = F.cosine_similarity(mean_embeddings1, mean_embeddings2)
    k = round((cosine_similarity.item() + 1)*100 / 2,2)
    glob_sim = k
    return glob_sim

def calculate_similarity(request):
    return glob_sim

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['SECRET_KEY'] = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@agent.on_event("upload")
async def handle_upload(ctx: Context):
    similarity_score = calculate_similarity(request)
    ctx.logger.info(f"Similarity Score: {similarity_score}")
    return similarity_score

@agent.on_event('recommended_skills')
async def handle_recommended_skills(ctx: Context):
    ctx.logger.info(f"Recommended Skill:{missing_sk}")
    return missing_sk

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    missing_keywords = {}
    if request.method == 'POST':
        # Check for both resume and job description files
        resume = request.files.get('resume')
        job_description = request.files.get('job_description')

        if not resume or resume.filename == '':
            flash('No resume file selected')
            return redirect(request.url)
        if not job_description or job_description.filename == '':
            flash('No job description file selected')
            return redirect(request.url)

        if resume and allowed_file(resume.filename) and job_description and allowed_file(job_description.filename):
            resume_filename = secure_filename(resume.filename)
            job_desc_filename = secure_filename(job_description.filename)

            resume.save(os.path.join(app.config['UPLOAD_FOLDER'], 'resume_' + resume_filename))
            job_description.save(os.path.join(app.config['UPLOAD_FOLDER'], 'job_desc_' + job_desc_filename))

            return redirect(url_for('screening', resume_name='resume_' + resume_filename, job_desc_name='job_desc_' + job_desc_filename)+"#hasil")
        score = agent.emit('upload')

    return render_template("index.html")

@app.route('/screening/<resume_name>/<job_desc_name>', methods=['GET', 'POST'])
def screening(resume_name,job_desc_name):
    missing_keywords = {}
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            pdfFileObj = open('uploads/{}'.format(filename), 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            num_pages = len(pdfReader.pages)
            count = 0
            text = ""
            while count < num_pages:
                pageObj = pdfReader.pages[count]
                count += 1
                text += pageObj.extract_text()

            resume_file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_name)
            with open(resume_file_path, 'rb') as pdfFileObj:
                pdfReader = PyPDF2.PdfReader(pdfFileObj)
                resume_text = ""
                for page in range(len(pdfReader.pages)):
                    resume_text += pdfReader.pages[page].extract_text()

            # Read and extract text from job description
            job_desc_file_path = os.path.join(app.config['UPLOAD_FOLDER'], job_desc_name)
            with open(job_desc_file_path, 'rb') as pdfFileObj:
                pdfReader = PyPDF2.PdfReader(pdfFileObj)
                job_desc_text = ""
                for page in range(len(pdfReader.pages)):
                    job_desc_text += pdfReader.pages[page].extract_text()

            resume_text = cleanResume(resume_text)
            job_desc_text = cleanResume(job_desc_text)

            # Calculate BLEU and ROUGE scores
            bleu_score = bert_similarity(job_desc_text, resume_text)
            # rouge_score = calculate_rouge_score(job_desc_text, resume_text)
            # Dictionary to store missing keywords
            missing_keywords = {}


            def cleanResume(resumeText):
                resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
                resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
                resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
                resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
                resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                                    resumeText)  # remove punctuations
                resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
                resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
                return resumeText.lower()

            text = cleanResume(text)

            bidang = {
                'Project Management': ['administration', 'agile', 'feasibility analysis', 'finance', 'leader', 'leadership',
                                       'management', 'milestones', 'planning', 'project', 'risk management', 'schedule',
                                       'stakeholders', 'teamwork', 'communication', 'organization', 'research',
                                       'public speaking', 'problem solving', 'negotiation', 'team management',
                                       'time management', 'adaptability', 'policy knowledge', 'reporting', 'technical',
                                       'motivation'],

                'Backend': ['flask', 'laravel', 'django', 'ruby on rails', 'express.js', 'codeigniter', 'golang', 'mysql',
                            'postgres', 'mongodb', 'relational database', 'non relational database', 'nosql',
                            'application programming interface', 'object oriented programming'],

                'Frontend': ['react', 'angular', 'vue.js', 'svelte', 'jquery', 'backbone.js ', 'ember.js', 'semantic-ui',
                             'html', 'css', 'bootstrap', 'javascript', 'jquery', 'xml', 'dom manipulation', 'json'],

                'Data Science': ['math', 'statistic', 'probability', 'preprocessing', 'machine learning',
                                 'data visualization',
                                 'python', 'r programming', 'tableau', 'natural language processing', 'data modeling',
                                 'big data', 'deep learning', 'relational database management', 'clustering', 'data mining',
                                 'text mining', 'jupyter', 'neural networks', 'deep neural network', 'pandas', 'scipy',
                                 'matplotlib', 'numpy', 'tensorflow', 'scikit learn', 'data analysis', 'data privacy',
                                 'enterprise resource planning', 'oracle', 'sybase', 'decision making', 'microsoft excel',
                                 'data collection', 'data cleaning', 'pattern recognition', 'google analytics'],

                'Devops': ['networking', 'tcp' 'udp', 'microsoft azure', 'amazon web services', 'alibaba cloud',
                           'google cloud',
                           'docker', 'kubernetes', 'virtual machine', 'cloud computing', 'security', 'linux', 'ubuntu',
                           'debian', 'arch linux', 'kali linux', 'automation', 'containers', 'operations', 'security',
                           'testing', 'troubleshooting']
            }
            # Check fo  r missing keywords in each domain
            for domain, keywords in bidang.items():
                missing_keywords[domain] = [word for word in keywords if word not in resume_text]
            # missing_keywords_list = [[domain] + keywords for domain, keywords in missing_keywords.items()]

            project = 0
            backend = 0
            frontend = 0
            data = 0
            devops = 0

            project_list = []
            backend_list = []
            frontend_list = []
            data_list = []
            devops_list = []

            # Create an empty list where the scores will be stored
            scores = []

            # Obtain the scores for each area
            for area in bidang.keys():
                if area == 'Project Management':
                    for word_project in bidang['Project Management']:
                        if word_project in text:
                            project += 1
                            project_list.append(word_project)
                    scores.append(project)
                elif area == 'Backend':
                    for word_backend in bidang['Backend']:
                        if word_backend in text:
                            backend += 1
                            backend_list.append(word_backend)
                    scores.append(backend)

                elif area == 'Frontend':
                    for word_frontend in bidang['Frontend']:
                        if word_frontend in text:
                            frontend += 1
                            frontend_list.append(word_frontend)
                    scores.append(frontend)

                elif area == 'Data Science':
                    for word_data in bidang['Data Science']:
                        if word_data in text:
                            data += 1
                            data_list.append(word_data)
                    scores.append(data)

                elif area == 'Devops':
                    for word_devops in bidang['Devops']:
                        if word_devops in text:
                            devops += 1
                            devops_list.append(word_devops)
                    scores.append(devops)

            data_all_list = {'Project Management': project_list, 'Backend': backend_list, 'Frontend': frontend_list,
                             'Data Science': data_list, 'DevOps': devops_list}
            # data_all_list_df = pd.DataFrame.from_dict(data_all_list, orient='index', dtype=object).transpose()

            summary = \
            pd.DataFrame(scores, index=bidang.keys(), columns=['score']).sort_values(by='score', ascending=False).loc[
                lambda df: df['score'] > 0]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.pie(summary['score'], labels=summary.index, autopct='%1.1f%%', startangle=90, shadow=True)
            ax.set_aspect('equal')
            ax.set_title("Ability score in each field.")
            buf = BytesIO()
            ax.figure.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            missing_json = jsonify(missing_keywords)
            score = agent.emit('recommended_skills')
            return render_template('index.html', data=data, bleu_score=bleu_score, missing_skills = missing_keywords)

    else:
        pdfFileObj = open('uploads/{}'.format(resume_name), 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        num_pages = len(pdfReader.pages)
        count = 0
        text = ""
        while count < num_pages:
            pageObj = pdfReader.pages[count]
            count += 1
            text += pageObj.extract_text()

        def cleanResume(resumeText):
            resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
            resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
            resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
            resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
            resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                                resumeText)  # remove punctuations
            resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
            resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
            return resumeText.lower()

        text = cleanResume(text)

        resume_file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_name)
        with open(resume_file_path, 'rb') as pdfFileObj:
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            resume_text = ""
            for page in range(len(pdfReader.pages)):
                resume_text += pdfReader.pages[page].extract_text()
        # Read and extract text from job description
        job_desc_file_path = os.path.join(app.config['UPLOAD_FOLDER'], job_desc_name)
        with open(job_desc_file_path, 'rb') as pdfFileObj:
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            job_desc_text = ""
            for page in range(len(pdfReader.pages)):
                job_desc_text += pdfReader.pages[page].extract_text()
        resume_text = cleanResume(resume_text)
        job_desc_text = cleanResume(job_desc_text)
        # Calculate BLEU and ROUGE scores
        bleu_score = bert_similarity(job_desc_text, resume_text)
        rouge_score = 0
        # rouge_score = calculate_rouge_score(job_desc_text, resume_text)
        # Dictionary to store missing keywords
        missing_keywords = {}
        bidang = {
            'Project Management': ['administration', 'agile', 'feasibility analysis', 'finance', 'leader', 'leadership',
                                   'management', 'milestones', 'planning', 'project', 'risk management', 'schedule',
                                   'stakeholders', 'teamwork', 'communication', 'organization', 'research',
                                   'public speaking', 'problem solving', 'negotiation', 'team management',
                                   'time management', 'adaptability', 'policy knowledge', 'reporting', 'technical',
                                   'motivation'],

            'Backend': ['flask', 'laravel', 'django', 'ruby on rails', 'express.js', 'codeigniter', 'golang', 'mysql',
                        'postgres', 'mongodb', 'relational database', 'non relational database', 'nosql',
                        'application programming interface', 'object oriented programming'],

            'Frontend': ['react', 'angular', 'vue.js', 'svelte', 'jquery', 'backbone.js ', 'ember.js', 'semantic-ui',
                         'html', 'css', 'bootstrap', 'javascript', 'jquery', 'xml', 'dom manipulation', 'json'],

            'Data Science': ['math', 'statistic', 'probability', 'preprocessing', 'machine learning', 'data visualization',
                             'python', 'r programming', 'tableau', 'natural language processing', 'data modeling',
                             'big data', 'deep learning', 'relational database management', 'clustering', 'data mining',
                             'text mining', 'jupyter', 'neural networks', 'deep neural network', 'pandas', 'scipy',
                             'matplotlib', 'numpy', 'tensorflow', 'scikit learn', 'data analysis', 'data privacy',
                             'enterprise resource planning', 'oracle', 'sybase', 'decision making', 'microsoft excel',
                             'data collection', 'data cleaning', 'pattern recognition', 'google analytics'],

            'Devops': ['networking', 'tcp' 'udp', 'microsoft azure', 'amazon web services', 'alibaba cloud', 'google cloud',
                       'docker', 'kubernetes', 'virtual machine', 'cloud computing', 'security', 'linux', 'ubuntu',
                       'debian', 'arch linux', 'kali linux', 'automation', 'containers', 'operations', 'security',
                       'testing', 'troubleshooting']
        }
        # Check for missing keywords in each domain
        for domain, keywords in bidang.items():
            missing_keywords[domain] = [word for word in keywords if word not in resume_text]
        # missing_keywords_list = [[domain] + keywords for domain, keywords in missing_keywords.items()]

        project = 0
        backend = 0
        frontend = 0
        data = 0
        devops = 0

        project_list = []
        backend_list = []
        frontend_list = []
        data_list = []
        devops_list = []

        # Create an empty list where the scores will be stored
        scores = []

        # Obtain the scores for each area
        for area in bidang.keys():
            if area == 'Project Management':
                for word_project in bidang['Project Management']:
                    if word_project in text:
                        project += 1
                        project_list.append(word_project)
                scores.append(project)
            elif area == 'Backend':
                for word_backend in bidang['Backend']:
                    if word_backend in text:
                        backend += 1
                        backend_list.append(word_backend)
                scores.append(backend)

            elif area == 'Frontend':
                for word_frontend in bidang['Frontend']:
                    if word_frontend in text:
                        frontend += 1
                        frontend_list.append(word_frontend)
                scores.append(frontend)

            elif area == 'Data Science':
                for word_data in bidang['Data Science']:
                    if word_data in text:
                        data += 1
                        data_list.append(word_data)
                scores.append(data)

            elif area == 'Devops':
                for word_devops in bidang['Devops']:
                    if word_devops in text:
                        devops += 1
                        devops_list.append(word_devops)
                scores.append(devops)

        data_all_list = {'Project Management': project_list, 'Backend': backend_list, 'Frontend': frontend_list,
                         'Data Science': data_list, 'DevOps': devops_list}
        #data_all_list_df = pd.DataFrame.from_dict(data_all_list, orient='index', dtype=object).transpose()

        summary = pd.DataFrame(scores, index=bidang.keys(), columns=['score']).sort_values(by='score', ascending=False).loc[
            lambda df: df['score'] > 0]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pie(summary['score'], labels=summary.index, autopct='%1.1f%%', startangle=90, shadow=True)
        ax.set_aspect('equal')
        ax.set_title("Ability score in each field.")
        buf = BytesIO()
        ax.figure.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        missing_sk = missing_keywords
        return render_template('index.html', data=data, bleu_score=bleu_score,missing_skills = missing_keywords)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
