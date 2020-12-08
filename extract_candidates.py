import json
from stanfordcorenlp import StanfordCoreNLP
import os
import re
# import cv2

host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port, timeout=90000)
props = {
    'annotators': 'tokenize,ner',
    'pipelineLanguage': 'en',
    'outputFormat': 'json'
}

invoice_date_candidates = []
invoice_no_candidates = []
total_amount_candidates = []

# filename = '457667-obama-wo-9-18-invoice-13496234090062-_-pdfpage_1.jpg'
# filename = '461396-priorities-usa-action-6278322-100912page_7.jpg'
filename = '461396-priorities-usa-action-6278322-100912page_15.jpg'

json_file = os.path.splitext(filename)[0] + '.json'

output_path = 'dataset/tesseract_results/' + json_file
image_path = 'dataset/praneet/' + filename

with open(output_path, 'rb') as f:
    data = json.load(f)

# all_words = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
all_words = []

for idx, word in enumerate(data['text']):
    if word.strip() != "":
        all_words.append({
            'text': data['text'][idx],
            'left': data['left'][idx],
            'top': data['top'][idx],
            'width': data['width'][idx],
            'height': data['height'][idx]})

text = ' '.join([word['text'].strip() for word in all_words])


def get_invoice_nums(all_words):
    inv_nums = []
    invoice_no_re = r'(?=^[\d-]{5,}$)(?=[0-9]+)'
    for word in all_words:
        # boolean result for match success "VIGU"
        # result = bool(pattern, word['text'])
        result = re.findall(invoice_no_re,word['text'])
        if result:
            inv_nums.append({
                'text': word['text'],
                'x1': word['left'],
                'y1': word['top'],
                'x2': word['left'] + word['width'],
                'y2': word['top'] + word['height']
            })

    return inv_nums


def get_dates(all_text):
    dates, all_dates = [], []
    indices = []
    index = -1
    annotations = json.loads(nlp.annotate(all_text))

    for x in annotations['sentences']:
        for y in x['entitymentions']:
            token_length = len(y['text'].split(' '))
            if y['ner'] == 'DATE':
                dates.append(y['text'])
                index = len(all_text[:y['characterOffsetEnd']].split(' '))
                if token_length < 2:
                    indices.append([index - 1])
                else:
                    indices.append(list(range(index - token_length, index)))

            index += token_length

    for date_indices in indices:
        date = ''
        left, top, right, bottom = [], [], [], []
        for i in date_indices:
            date += ' ' + all_words[i]['text']
            left.append(all_words[i]['left'])
            top.append(all_words[i]['top'])
            right.append(all_words[i]['left'] + all_words[i]['width'])
            bottom.append(all_words[i]['top'] + all_words[i]['height'])
        all_dates.append({
            'text': date.strip(),
            'x1': min(left),
            'y1': min(top),
            'x2': max(right),
            'y2': max(bottom)
        })

    return all_dates


def get_amounts(all_words):
    amounts = []

    for word in all_words:
        formatted_word = word['text'].replace(',', '')
        annotations = json.loads(nlp.annotate(formatted_word))
        for x in annotations['sentences']:
            for y in x['entitymentions']:
                if y['ner'] == 'MONEY':
                    amounts.append({
                        'text': word['text'],
                        'x1': word['left'],
                        'y1': word['top'],
                        'x2': word['left'] + word['width'],
                        'y2': word['top'] + word['height']
                    })
                elif y['ner'] == 'NUMBER':
                    try:
                        _ = float(y['text'])
                        if '.' in word['text']:
                            amounts.append({
                                'text': word['text'],
                                'x1': word['left'],
                                'y1': word['top'],
                                'x2': word['left'] + word['width'],
                                'y2': word['top'] + word['height']
                            })
                    except:
                        continue

    return amounts


invoice_date_candidates = get_dates(text)
total_amount_candidates = get_amounts(all_words)
invoice_no_candidates = get_invoice_nums(all_words)


# # Drawing candidates on image
# img = cv2.imread(image_path)
#
# for i in total_amount_candidates:
#     cv2.rectangle(img, (i['x1'], i['y1']), (i['x2'], i['y2']), (0, 0, 0), 5)
#
# cv2.imwrite('test.jpg', img)
