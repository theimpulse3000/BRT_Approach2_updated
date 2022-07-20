'''pdfminer uses strategy of lazy parsing i.e. parse the information only when needed 
   refer - https://pdfminer-docs.readthedocs.io/programming.html for help'''

from datetime import datetime
from dateutil import relativedelta
import io
from io import StringIO
from pdfminer.converter import TextConverter
# interpreter to process the page content
from pdfminer.pdfinterp import PDFPageInterpreter
# used to store shared resources such as fonts and images
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
#from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import resolve1
from pdfminer.layout import LAParams
import os
#from pdfminer.pdfparser import PDFSyntaxError
import docx2txt
import subprocess
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download(['stopwords','wordnet'])
import entities_pattern as ep
#spacy
import spacy
nlp = spacy.load("nl_core_news_sm")
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

'''# extract text from pdf
def extract_text_from_pdf(pdf_path):
    try :
        fp = open(pdf_path, 'rb')
        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        document = PDFDocument(parser)
        # Check if the document allows text extraction. If not, abort.
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
        # Create a PDF resource manager object that stores shared resources.
        resource_manager = PDFResourceManager()
        #The StringIO module is an in-memory file-like object.This object can be 
        #used as input or output to the most function that would expect a standard file object. 
        #When the StringIO object is created it is initialized by passing a string to the constructor. 
        #If no string is passed the StringIO will start empty.
        fake_file_object = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_object)
        # Create a PDF interpreter object.
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        with open(pdf_path, 'rb') as fp1 :
            for page in PDFPage.get_pages(fp1, caching = True, check_extractable = True):
                page_interpreter.process_page(page)
            text = fake_file_object.getvalue()
        # close open handels
        converter.close()
        fake_file_object.close()
        if text :
            return text           
    except PDFSyntaxError:
        return'''


def extract_text_from_pdf(pdf_path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(pdf_path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text

# extract text from docx
def extract_text_from_docx(docx_path) :
    text = docx2txt.process(docx_path)
    if text :
        text = [line.replace('\t', ' ') for line in text.split('\n') if line]
        return ' '.join(text)
    return None

# extract text from doc
def extract_text_from_doc(doc_path) :
    try :
        # execute a child program in a new process
        process = subprocess.Popen(['catdoc', '-w', doc_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    except(FileNotFoundError, ValueError, subprocess.TimeoutExpired, subprocess.SubprocessError) as err:
        return(None, str(err))
    else :
        '''Interact with process: Send data to stdin. Read data from stdout and stderr, 
        until end-of-file is reached. Wait for process to terminate and set the returncode 
        attribute. The optional input argument should be data to be sent to the child process, 
        or None, if no data should be sent to the child. If streams were opened in text mode, 
        input must be a string. Otherwise, it must be bytes.'''
        stdout, stderr = process.communicate()
        # remove leading and trailing spaces
        return (stdout.strip(), stderr.strip())

# getting extension of resume
def get_resume_extension(file_path) :
    # find out substring after . character
    extension = file_path.partition(".")[2]
    return extension

# wrapper function for extractig the text
def extract_text(file_path) :
    text = ''
    extension = get_resume_extension(file_path)
    if extension == "pdf" :
        text = extract_text_from_pdf(file_path)
    elif extension == "docx" :
        text = extract_text_from_docx(file_path)
    elif extension == "doc" :
        text = extract_text_from_doc(file_path)
    return text

'''# get number of pages in pdf
def get_pdf_no_of_pages(file_path) :
    file_object = open(file_path, 'rb')
    parser = PDFParser(file_object)
    document = PDFDocument(parser)
    return resolve1(document.catlog['Pages'])['Count']'''

# cleaning of the text using nltk
def clean_my_resume_text(extracted_resume_text) :
    clean_text = []
    # regex to remove hyperlinks, special characters, or punctuations.
    resume_text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"', " ", extracted_resume_text)
    # Lowering text
    resume_text = resume_text.lower()
    # Splitting text into array based on space
    resume_text = resume_text.split()
    # Lemmatizing text to its base form for normalizations
    lm = WordNetLemmatizer()
    # removing English stopwords
    resume_text = [lm.lemmatize(word) for word in resume_text if not word in set(stopwords.words("english"))]
    resume_text = " ".join(resume_text)
    # Appending the results into an array.
    clean_text.append(resume_text)
    return clean_text

# Helper function to extract different entities with custom trained model using SpaCy's NER
def extract_entities_with_NER(nlp_resume_text) :
    entities = {}
    for ent in nlp_resume_text.ents :
        if ent.label_ not in entities.keys() :
            entities[ent.label_] = [ent.text]
        else :
            entities[ent.label_].append(ent.text)
    for key in entities.keys() :
        entities[key] = list(set(entities[key]))
    return entities

# Helper function to extract all the raw text from sections of resume specifically for graduates and undergraduates
def grad_entity_sections_extract(resume_text) :
    key = False
    split_text = [i.strip() for i in resume_text.split('\n')]
    entities = {}
    for word_phrase in split_text :
        if len(word_phrase) == 1 :
            key_item = word_phrase
        else :
            key_item = set(word_phrase.lower().split()) & set(ep.GRAD_RESUME_SECTIONS)
        try :
            key_item = list(key_item)[0]
        except IndexError :
            pass
        if key_item in ep.GRAD_RESUME_SECTIONS :
            entities[key_item] = []
            key = key_item
        elif key and word_phrase.strip():
            entities[key].append(word_phrase)
    return entities

# Helper function to extract all the raw text from sections of resume specifically for professionals
def prof_entity_sections_extract(resume_text) :
    key = False
    split_text = [i.strip() for i in resume_text.split('\n')]
    entities = {}
    for word_phrase in split_text :
        if len(word_phrase) == 1 :
            key_item = word_phrase
        else :
            key_item = set(word_phrase.lower().split()) & set(ep.PROF_RESUME_SECTIONS)
        try :
            key_item = list(key_item)[0]
        except IndexError :
            pass
        if key_item in ep.PROF_RESUME_SECTIONS :
            entities[key_item] = []
            key = key_item
        elif key and word_phrase.strip():
            entities[key].append(word_phrase)
    return entities
# date1 : start date
# date2 : end date
def get_no_of_months(date1, date2) :
    if date2.lower() == "present" :
        date2 = datetime.now().strftime('%b %Y')
    try :
        if len(date1.split()[0]) > 3:
            date1 = date1.split()
            date1 = date1[0][:3] + ' ' + date1[1]
        if len(date2.split()[0]) > 3 :
            date2 = date2.split()
            date2 = date2[0][:3]+ ' ' + date2[1]
    except IndexError :
        return 0
    try :
        date1 = datetime.strptime(str(date1), '%b %Y')
        date2 = datetime.strptime(str(date2), '%b %Y')
        months_of_experience = relativedelta.relativedelta(date2, date1)
        months_of_experience = (months_of_experience.years * 12 + months_of_experience.months)
    except ValueError :
        return 0
    return months_of_experience        

# extract experience from resume text
# returns : list of experiences
def extract_experience(resume_text) :
    wordLemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    # tokenisation
    tokens_word = nltk.word_tokenize(resume_text)
    # remove stop words and lemmatize
    req_sentences = [w for w in tokens_word
        if w not in stopWords
        and wordLemmatizer.lemmatiza(w)
        not in stopWords]
    # a process to mark up the words in text format for a particular part of a speech based on its definition and context.
    sent = nltk.pos_tag(req_sentences)
    # parse regex
    ent_pattern = nltk.RegexParser('P: {<NNP>+}')
    cs = ent_pattern.parse(sent)
    test = []
    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)
    # VP = one of phrae tags
    for vp in list(cs.subtrees(filter = lambda x : x.label() == 'P')) :
            test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))
    # search the wprd 'experience' in the text and then print the text after it
    req_text = [req_text[req_text.lower().index('experience') + 10 : ] for req_text in enumerate(test) if req_text and 'experience' in req_text.lower()]
    return req_text

# getting total months of experience
# list of experiences from above function
def total_experience(list_experience) :
    expr = []
    for line in list_experience :
        # re.I -> Perform case-insensitive matching; expressions like [A-Z] will also match lowercase letters
        exp = re.search(r'(?P<fmonth>\w+.\d+)\s*(\D|to)\s*(?P<smonth>\w+.\d+|present)', line, re.I)
        if exp :
            # groups() -> This method returns a tuple containing all the subgroups of the match, from 1 up to however many groups are in the pattern. 
            expr.append(exp.groups())
    total_exp = sum([get_no_of_months(i[0], i[2]) for i in expr])
    total_exp_months = total_exp
    return total_exp_months

# extract education
def extract_education(nlp_resume_text) :
    education = {}
    for index, text in enumerate(nlp_resume_text) :
        for text1 in text.split() :
            text1 = re.sub(r'[?|$|.|!|,]', r'', text1)
            if text1.upper() in ep.EDUCATION and text1 not in ep.STOP_WORDS :
                education[text1] = text + nlp_resume_text[index + 1]
    # now extract year
    edu_year = []
    for key in education.keys() :
        year = re.search(re.compile(ep.YEAR), education[key])
        if year :
            edu_year.append((key, ''.join(year.group(0))))
        else :
            edu_year.append(key)
    return edu_year

# extrating name from spacy nlp text 
# nlp_text: object of `spacy.tokens.doc.Doc
# matcher : object of spacy.matcher.Matcher
def extract_name(nlp_resume_text, matcher) :
    pattern = [ep.NAME_PATTERN]
    matcher.add("NAME", pattern)
    matches = matcher(nlp_resume_text)
    for _, start, end in matches :
        span = nlp_resume_text[start:end]
        return span.text

# extract mobile number
def extract_mobile_number(resume_text, custom_regex = None) :
    if not custom_regex :
        mobile_number_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                              [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone_number = re.findall(re.compile(mobile_number_regex), resume_text)
    else :
        phone_number = re.findall(re.compile(custom_regex), resume_text)
    if phone_number :
        mobile_number = ''.join(phone_number[0])
        return mobile_number

# extract email address
def extract_email_address(resume_text) :
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", resume_text)
    if email :
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None


'''if __name__ == '__main__' :
    file_path = "/Users/sagar_19/Desktop/BRT_Approach2/resumes/SagarMali_Resume.pdf"
    file_path1 = "/Users/sagar_19/Desktop/BRT_Approach2/resumes/SamruddhiPatil_Resume.docx"
    text = extract_text(file_path)
    #clean_text = clean_my_resume_text(text)
    #print(clean_text)
    text1 = extract_text(file_path1)
    clean_text = clean_my_resume_text(text1)
    print(clean_text)'''