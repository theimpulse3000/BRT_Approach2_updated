import dataset_BRT_approach as dsa
import functionalities as fn
import text_based_approach as tba
from text_based_approach import terms
import resume_parser as rp
import os
import shutil
import pprint
import multiprocessing as multp

if __name__ == '__main__' :
    input_skills = input("\nEnter required skills : ")
    threshold_percetage = int(input("\nEnter threshold for shortlisting : "))
    main_dir = "/Users/sagar_19/Desktop/BRT_Approach2/src/"
    src_dir = input("\nEnter input folder name : ")
    src_dir = main_dir + src_dir + "/"
    dest_dir = input("\nEnter output output folder name : ")
    dest_dir = main_dir + dest_dir + "/"
    print("\nFollowing are the ways we can use BRT : ")
    print("\n")
    print("1. Apply BRT on dataset you have")
    print("2. Apply BRT based on text approach on resumes ")
    print("3. Apply BRT based on NLP approach on resumes")
    print("\n")
    choice = input("Enter your choice : ")
    print("\n")
    if choice == '1' :
        print("Please wait till your dataset is being processed......")
        skill_patterns_path = "/Users/sagar_19/Desktop/BRT_Approach2/dataset/skills_pattern.jsonl"
        csv_path = "/Users/sagar_19/Desktop/BRT_Approach2/dataset/Resume.csv"
        print("\nReading the dataset and converting into dataframe ....")
        data, df = dsa.read_resume_dataset(csv_path)
        #print(data.head())
        new_ruler = dsa.pipeline_newruler_adder(skill_patterns_path)
        #print(new_ruler)
        #print(nlp.pipe_names)
        clean_text = dsa.clean_resume_text(data)
        #print(clean_text)
        print("\nModifying the dataset....")
        data = dsa.modify_resume_csv(data, clean_text)
        #print(data["clean_resume"].iloc[0])
        #print(data["skills"].iloc[0])
        #job_category = dsa.create_job_cat_var(data)
        #print(job_category)
        '''['TEACHER' 'BANKING' 'ARTS' 'FITNESS' 'HEALTHCARE' 'APPAREL' 'ENGINEERING'
        'DIGITAL-MEDIA' 'FINANCE' 'AGRICULTURE' 'SALES' 'CONSTRUCTION'
        'CONSULTANT' 'AVIATION' 'ACCOUNTANT' 'INFORMATION-TECHNOLOGY' 'DESIGNER'
        'CHEF' 'ADVOCATE' 'BUSINESS-DEVELOPMENT' 'PUBLIC-RELATIONS' 'HR'
        'AUTOMOBILE' 'ALL']'''
        #markup_resume_text = dsa.use_entity_recg(data)
        #print(markup_resume_text)
        print("\nHighlighting the entities in the resume....")
        markup_text = dsa.custom_NER(data, df, new_ruler)
        f2 = open("output.html", "w")
        for line in markup_text :
            f2.write(line)
        f2.close()
        print("\nYou can see highlighted entities of first resume of dataset - just open output.html in browser")
        #print(markup_text)
        #my_resume_text = functionalities.extract_text(resume_path)
        #print(type(my_resume_text))
        #print(my_resume_text)
        #my_resume_text = functionalities.clean_my_resume_text(my_resume_text)
        #my_resume_text = ' '.join(my_resume_text)
        #print(type(my_resume_text))
        #print(my_resume_text)
        #nlp_resume_text = nlp(my_resume_text)
        #displacy.render(nlp_resume_text, style = "ent")
        resume_text = data["Resume_str"].iloc[0]
        clean_resume_text = data["clean_resume"].iloc[0]
        job_cat = data["Category"].iloc[0]
        #print(job_cat)
        #most_used = dsa.most_used_word(data, clean_resume_text)
        most_used = dsa.most_used(resume_text) 
        print("\nMost used words in given resume are %s" %most_used)
        total_skills = dsa.skills_distribution(data, job_cat)
        print("\nTotal skillset present in the given resume %s" %total_skills)
        match = dsa.matching_score(input_skills, clean_resume_text)
        print(f"\nThe current Resume is {match}% matched to your requirements\n")
        if(match >= threshold_percetage) :
            print("First resume of dataset is shortlisted")
        else :
            print("First resume of dataset is not shortlisted")
        print("\nEncoding labels....")
        data = dsa.label_encoding(data)
        #print(data.Category.value_counts())
        print("\nApplying word vectorizer....")
        try :
            X_train, X_test, y_train, y_test = dsa.word_vectorizer(data)
            #print(X_train, X_test, y_train, y_test)
            print("\nDoing prediction by classifier....\n")
            training_accuracy, testing_accuracy, report_classifier = dsa.predict_classifier(X_train, X_test, y_train, y_test)
            print('\nAccuracy of KNeighbors Classifier on training set: {:.2f}'.format(training_accuracy))
            print('\nAccuracy of KNeighbors Classifier on test set: {:.2f}'.format(testing_accuracy))
            print("\n Classification report for classifier - KNeighborsClassifier :\n%s\n" % (report_classifier))
        except ValueError :
            print("\nIn given dataset, y class has only 1 member whichh is too few for prediction by classifier")
    elif choice == '2' :
        resumes = []
        data = []
        for dirpath, dirnames, filenames in os.walk(src_dir) :
            for resume_file in filenames :
                file = os.path.join(dirpath, resume_file)
                resumes.append(file)
        start = 1
        end = len(resumes)
        #print(resumes)
        for i in range(start, end) :
            #print("Hi")
            resume_path = resumes[i]
            resume_text = fn.extract_text(resume_path)
            cleaned_text = tba.clean_text(resume_text)
            count = tba.return_count_list(terms)
            scores = tba.resume_analysis(cleaned_text, count)
            path = os.path.basename(os.path.normpath(resumes[i]))
            tba.is_shortlist(scores, path, threshold_percetage)
    elif choice == "3" :
        skill_patterns_path = "/Users/sagar_19/Desktop/BRT_Approach2/dataset/skills_pattern.jsonl"
        new_ruler = dsa.pipeline_newruler_adder(skill_patterns_path)
        # About pool object
        '''Pool object which offers a convenient means of parallelizing the execution 
         of a function across multiple input values, distributing the input data across processes (data parallelism)'''
        # About cpu_count()
        '''One of the useful functions in multiprocessing is cpu_count() . This returns the number 
        of CPUs (computer cores) available on your computer to be used for a parallel program'''
        pool = multp.Pool(multp.cpu_count())
        resumes = []
        data = []
        for dirpath, dirnames, filenames in os.walk(src_dir) :
            for resume_file in filenames :
                file = os.path.join(dirpath, resume_file)
                resumes.append(file)
        print("\nDeleting the files if pre-exist in shortlisted folder.....")
        for file_name in os.listdir(dest_dir):
            #print("In delete function")
            # construct full file path
            file = dest_dir + file_name
            #print(file)
            if os.path.isfile(file):
                #print('Deleting file:', file)
                os.remove(file)
        start = 1
        end = len(resumes) + 1
        for resume_path in resumes[start : end] :
            resume_text = fn.extract_text(resume_path)
            markup_resume_text = dsa.use_entity_recg_for_resume(resume_text)
            #print(markup_resume_text)
            #print(resume_text)
            clean_text = fn.clean_my_resume_text(resume_text)            
            #print(type(clean_text))
            clean_text_str = ""
            for i in clean_text:
                clean_text_str += i
            #print(clean_text_str)
            match = dsa.matching_score(input_skills, clean_text_str)
            #print(match)
            if(match >= threshold_percetage) :
                src_path = resume_path
                shutil.copy2(src_path, dest_dir)
                #print('Copied')
        shortlisted_resumes = []
        print("\nMoving shortlisted resume into shortlisted folder...")
        for dirpath, dirnames, filenames in os.walk(dest_dir) :
            #print("In result list")
            for resume_file in filenames :
                file = os.path.join(dirpath, resume_file)
                shortlisted_resumes.append(file)
        #print(shortlisted_resumes)
        #start = 1
        #end = len(shortlisted_resumes)
        results = [pool.apply_async(rp.resume_result_wrapper, args = (i,)) for i in shortlisted_resumes]
        #results = results.append(results)
        #print(results)
        results = [p.get() for p in results]
        #print(results)
        print("\nThe details of shortlisted candidates are as follows : \n")
        pprint.pprint(results)
        print("\n\n")