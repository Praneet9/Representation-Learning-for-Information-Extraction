import json
import re


def get_invoice_nums(all_words):
    inv_nums = []
    invoice_no_re = r'^[0-9a-zA-Z-:]+$'
    for word in all_words:
        if not re.search('\d', word['text']):
            continue
        if len(word['text']) < 3:
            continue
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


def get_dates(all_text,all_words):
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
    amount_re = r"\$?([0-9]*,)*[0-9]{3,}(\.[0-9]+)?"
    for word in all_words:
        if not re.search(amount_re, word['text']):
            continue
        try:
            formatted_word = re.sub(r'[$,]','', word['text'])
            float(formatted_word)
        
            amounts.append({
                'text': word['text'],
                'x1': word['left'],
                'y1': word['top'],
                'x2': word['left'] + word['width'],
                'y2': word['top'] + word['height']
            })

        except ValueError:
            continue

    return amounts

def get_candidates(data):
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
        try:
            invoice_date_candidates = get_dates(text,all_words)
        except Exception as e:
            invoice_date_candidates = []
        try:
            total_amount_candidates = get_amounts(all_words)
        except Exception as e:
            total_amount_candidates = []
        try:
            invoice_no_candidates = get_invoice_nums(all_words)
        except Exception as e:
            invoice_no_candidates = []
        candidate_data = {
            'invoice_no': invoice_no_candidates,
            'invoice_date': invoice_date_candidates,
            'total': total_amount_candidates
        }
        return candidate_data