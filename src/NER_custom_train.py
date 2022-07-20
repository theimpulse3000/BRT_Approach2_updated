from __future__ import unicode_literals
from __future__ import print_function
# import requirements
import random
import logging
from pathlib import Path
import re
import json
import spacy
from spacy.training import Example
nlp = spacy.load("nl_core_news_sm")
# print(nlp.pipe_names)
# getting pipeline component
ner = nlp.get_pipe("ner")
# reference - https://medium.com/@dataturks/automatic-summarization-of-resumes-with-ner-8b97a5f562b
# For more details, see the documentation:
# Training: https://spacy.io/usage/training
# NER: https://spacy.io/usage/linguistic-features#named-entities
# Removes leading and trailing white spaces from entity spans


def entity_span_trim(data: list) -> list:
    span_tokens_invalid = re.compile(r'\s')
    cleaned_data = []
    for text, annotations in data:
        enti = annotations['entities']
        valid_entities = []
        for start, end, label in enti:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and span_tokens_invalid.match(text[valid_start]):
                valid_start += 1
            while valid_end > 1 and span_tokens_invalid.match(text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data

def convert_dataturks_to_spacy(json_file_path):
    try :
        training_data = []
        lines=[]
        with open(json_file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]
                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))
            training_data.append((text, {"entities" : entities}))
        return training_data
    except Exception :
        logging.exception("Unable to process " + json_file_path)
        return None

TRAIN_DATA = entity_span_trim(convert_dataturks_to_spacy("/Users/sagar_19/Desktop/BRT_Approach2/src/traindata.json"))

random.seed(0)
#model = None
new_model_name = "training"
#output_dir = "/Users/sagar_19/Desktop/BRT_Approach2/src"
output_dir = "/Users/sagar_19/Desktop/BRT_Approach2/testing" 

'''if model is not None:
    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank("en")  # create blank Language class
    print("Created blank 'en' model")'''

if "ner" not in nlp.pipe_names:
    print("Creating new pipe")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
# otherwise, get it, so we can add labels to it
else:
    ner = nlp.get_pipe("ner")

# adding labels to ner
for _, annotations in TRAIN_DATA :
    for ent in annotations.get("entities") :
        #print(annotations.get("entities"))
        ner.add_label(ent[2])

move_names = list(ner.move_names)
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

# Training the model
# only train NER
# An Optimizer object, which updates the model’s weights. If not set, spaCy will create a new one and save it for further use.
with nlp.disable_pipes(*other_pipes) :
    # Training for 100 iterations
    # use mini batch
    optimizer = nlp.begin_training()
    for iteration in range(100) :
        print("Starting iteration " + str(iteration))
        # shuffle before every iteration
        random.shuffle(TRAIN_DATA)
        losses = {}
        try :
            for text, annotations in TRAIN_DATA :
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # drop -> Dropout rate. Makes it harder for the model to just memorize the data.
                nlp.update([example], drop=0.2, sgd = optimizer, losses = losses)
                #print("Losses", losses)
        except ValueError :
            continue

test_text = "studied in College of Engineering Pune"
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)

new_model_name = "training"
# save model to output directory
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta["name"] = new_model_name  # rename model
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    # Check the classes have loaded back consistently
    assert nlp2.get_pipe("ner").move_names == move_names
    doc2 = nlp2(test_text)
    for ent in doc2.ents:
        print(ent.label_, ent.text)