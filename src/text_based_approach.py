import pdfplumber
import re
import string
import PyPDF2
import pandas as pd
from prettytable import PrettyTable
import functionalities as fn
import os


def extraction_of_text(resume_path) :
  # Task 1 : Opening, reading PDF and getting number of pages in it
  resume_path = "/Users/sagar_19/Desktop/BRT_Approach2/resumes/SagarMali_Resume.pdf"
  file = open(resume_path, 'rb')
  read_pdf = PyPDF2.PdfFileReader(file, strict = "True")
  # strict = "True" will inform the user about fatal error occured while reading the pdf file
  total_pages = read_pdf.numPages
  # print(total_pages)

  # Task 2 : Extracting text from PDF
  with pdfplumber.open(resume_path) as pdf :
    count = 0
    text = ""
    while count < total_pages :
      page = pdf.pages[count]
      text = text + page.extract_text()
      count = count + 1
    return text

def clean_text(text) :  
  # Task 3 : text cleaning

  # convert text into lower case
  text = text.lower()
  # print(text)

  # remove punctuations by fastest method
  # refer - https://datagy.io/python-remove-punctuation-from-string/
  # comparison between three methods 
  #  ________________________________________________
  # |Method	                      |  Time Taken      |
  # |str.translate()	            |    2.35 seconds  |
  # |regular expressions	        |   88.8 seconds   |
  # |for loop with str.replace()	|    20.6 seconds  |
  # |_____________________________|__________________|

  text = text.translate(str.maketrans('', '', string.punctuation))
  # print(text)

  # Remove numbers
  text = re.sub(r'\d+','',text)
  # print(text)

  # remove new lines
  cleaned_text = re.sub('\n', '', text)
  #print(text)

  # remove bullet points - not working with regular expression code part

  return cleaned_text

# Task 4 : dictionary setup
# Create dictionary with industrial and system engineering key terms by area
terms = {'Quality/Six Sigma':['black belt','capability analysis','control charts','doe','dmaic','fishbone',
                              'gage r&r', 'green belt','ishikawa','iso','kaizen','kpi','lean','metrics',
                              'pdsa','performance improvement','process improvement','quality',
                              'quality circles','quality tools','root cause','six sigma',
                              'stability analysis','statistical analysis','tqm'],      
        'Operations management':['automation','bottleneck','constraints','cycle time','efficiency','fmea',
                                 'machinery','maintenance','manufacture','line balancing','oee','operations',
                                 'operations research','optimization','overall equipment effectiveness',
                                 'pfmea','process','process mapping','production','resources','safety',
                                 'stoppage','value stream mapping','utilization'],
        'Supply chain':['abc analysis','apics','customer','customs','delivery','distribution','eoq','epq',
                        'fleet','forecast','inventory','logistic','materials','outsourcing','procurement',
                        'reorder point','rout','safety stock','scheduling','shipping','stock','suppliers',
                        'third party logistics','transport','transportation','traffic','supply chain',
                        'vendor','warehouse','wip','work in progress'],
        'Project management':['administration','agile','budget','cost','direction','feasibility analysis',
                              'finance','kanban','leader','leadership','management','milestones','planning',
                              'pmi','pmp','problem','project','risk','schedule','scrum','stakeholders'],
        'Data analytics':['analytics','api','aws','big data','busines intelligence','clustering','code',
                          'coding','data','database','data mining','data science','deep learning','hadoop',
                          'hypothesis test','iot','internet','machine learning','modeling','nosql','nlp',
                          'predictive','programming','python','r','sql','tableau','text mining',
                          'visualuzation']}

def return_count_list(terms) :
  count_dic = {k:len(v) for k,v in terms.items()}
  count =list(count_dic.values())
  #print(count)
  return count

def resume_analysis(text, count) :
  # Task 5 : Analysis per area of job descriptiopn
  # Initialize counter for each area 
  quality = operations = supplychain = project = data = 0
  scores = []
  for area in terms.keys():
    if area == "Quality/Six Sigma" :
      for word in terms[area]:
        if word in text:
          quality = quality + 1
      quality = (quality / count[0]) * 100
      scores.append(quality)
  
    elif area == 'Operations management':
          for word in terms[area]:
              if word in text:
                  operations +=1
          operations = (operations / count[1]) * 100
          scores.append(operations)
        
    elif area == 'Supply chain':
          for word in terms[area]:
              if word in text:
                  supplychain +=1
          supplychain = (supplychain / count[2]) * 100
          scores.append(supplychain)
        
    elif area == 'Project management':
          for word in terms[area]:
              if word in text:
                  project +=1
          project = (project / count[3]) * 100
          scores.append(project)
        
    elif area == 'Data analytics':
          for word in terms[area]:
              if word in text:
                  data +=1
          data = (data / count[4]) * 100
          scores.append(data)
  return scores

def is_shortlist(scores, path, threshold_percentage) :

  # Task 6 : Representing data in tabular form
  # Using data frame for representation
  '''df = pd.DataFrame(scores, index = terms.keys(), columns = ['score']) 
  print(df)'''

  names = ['Quality/Six Sigma', 'Operations management', 'Supply chain', 'Project management','Data analytics']

  # print(names[0] + " " + str(scores[0]))

  #i = 0
  #for i,item in enumerate(names, start=i):
  #    print(names[i] + " " + str(scores[i]))
  
  t = PrettyTable(['Job Profile', 'Scores'])
  j = 0
  for j,item in enumerate(names, start=j):
      t.add_row([names[j], scores[j]])
  print(t)

  # Task 7 : Checking resume is suitable for job description or not
  if scores[0] >= threshold_percentage :
    print("%s is shortlisted for Quality/Six Sigma with %d percentage\n" %path %scores[0])
  elif scores[1] >= threshold_percentage :
    print("%s is shortlisted for Operations management with %d percentage\n" %path %scores[1])
  elif scores[2] >= threshold_percentage :
    print("%s is shortlisted for Supply chain with %d percentage\n" %path %scores[2])
  elif scores[3] >= threshold_percentage :
    print("%s is shortlisted for Project management with %d percentage\n" %path %scores[3])
  elif scores[4] >= threshold_percentage :
    print("%s is shortlisted for Data analytics with %d percentage\n" %path %scores[4])
  else :
    print("\nResume is not shortlisted for any area\n")

# Time complexiy : O(n^2)