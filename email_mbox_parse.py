'''
Parse email mailbox .mbox


# For Dev:
/usr/local/lib/python3.7/pdb.py email_mbox_parse.py
'''
import csv
import datetime
import email
import json
import mailbox
import pandas as pd
from pprint import pprint
import re


def get_message(message):
    ''' example message parsing code
    '''
    if not message.is_multipart():
        return message.get_payload()
    contents = ""
    for msg in message.get_payload():
        contents = contents + str(msg.get_payload()) + '\n'
    return contents


def parse_domain_from_email(raw):
    ''' Parse domain from string containing an email address.
    '''
    raw_str = str(raw)
    domain_str = re.findall(r'@[\w\.-]+\.\w+', raw_str)[0].replace('@', '')
    domain_list = domain_str.split('.')
    domain_out = '.'.join(domain_list[-2:])
    return domain_out
    

def getbody(message): #getting plain text 'email body'
    '''Decode methods:
    .decode('ascii') # simplest range(128)
    .decode('utf-8') # doesnt get all the windows, webpage chars
    .decode('cp1252')# seems to get all the windows, webpage chars

    '''
    body = None
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        try:
                            body = subpart.get_payload(decode=True).decode('utf-8')
                        except:
                            body = subpart.get_payload(decode=True).decode('cp1252')
            elif part.get_content_type() == 'text/plain':
                try:
                    body = part.get_payload(decode=True).decode('utf-8')
                except:
                    body = part.get_payload(decode=True).decode('cp1252')
    elif message.get_content_type() == 'text/plain':
        try:
            body = message.get_payload(decode=True).decode('utf-8')
        except:
            body = message.get_payload(decode=True).decode('cp1252')
    return body


def parse_mytext(body):
    '''Parse my text, seperate from the inlined replies and other
    conversations.

    '''
    re_date = re.compile('^On\ (Sun|Mon|Tue|Wed|Thu|Fri|Sat)\,\ (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\ .*')
    re_date_2 = re.compile('^On\ (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\ .*')
    re_forward = re.compile('.*-\ [fF]orwarded\ [mM]essage\ -.*')
    re_signature = re.compile('^Dan Starr.*')
    
    i_ends = []
    body_split = body.split('\n')
    for i, line in enumerate(body_split):
        if len(line) == 0:
            continue
        elif '-Dan' in line:
            i_ends.append(i)
        elif re_date.match(line):
            i_ends.append(i)
            break
        elif re_date_2.match(line):
            i_ends.append(i)
            break
        elif re_forward.match(line):
            i_ends.append(i)
            break
        elif re_signature.match(line):
            i_ends.append(i)
            break
        elif line[0] == '>':
            i_ends.append(i)
            break
    if len(i_ends) == 0:
        i_ends = [len(body_split) - 1]
    i_min = min(i_ends)

    #mytext = '\n'.join(body_split[:i_min])
    # return a list of lines, which combines better with other data:
    return body_split[:i_min]


def get_header_date(message):
    ''' Get send date from header in email message.
    Return a date tup of form: (2021, 6, 25, 13, 49, 17, 0, 1, -1)
    '''
    date_tup = None
    for (k,v) in message._headers:
        if k == 'Date':
            date_tup = email.utils.parsedate(v)
    return date_tup

    
def get_mytext_from_mbox(mbox_fpath, max_n_messages=1e12):
    ''' Given a .mbox file, get my text from emails
    '''
    meta_data = {}
    all_lines = []
    date_tup_list = []
    i_valid_messages = 0
    for i, message in enumerate(mailbox.mbox(mbox_fpath)):
        if i > max_n_messages:
            break
        #pprint(message.__dict__)
        date_tup = get_header_date(message)
        if date_tup is not None:
            date_tup_list.append(date_tup)
        if i % 10 == 0:
            print(i)
        #pprint(dict(message.items()))
        body = getbody(message)
        if isinstance(body, str):
            mytext = parse_mytext(body)
            all_lines.extend(mytext)
            i_valid_messages += 1

    #pprint(datetime.datetime(*date_tup_list[0][:7]).strftime('%Y-%m-%dT%H:%M:%S'))
    #pprint(datetime.datetime(*date_tup_list[0][:7]))
    #datetime.datetime(*min(date_tup_list)[:7])
    meta_data['email_dt_first'] = datetime.datetime(*min(date_tup_list)[:7]).strftime('%Y-%m-%dT%H:%M:%S')
    meta_data['email_dt_last'] = datetime.datetime(*max(date_tup_list)[:7]).strftime('%Y-%m-%dT%H:%M:%S')
    #meta_data['email_dt_first'] = min(date_tup_list)
    #meta_data['email_dt_last'] = max(date_tup_list)
    meta_data['email_n_msg'] = i_valid_messages
    meta_data['email_n_lines'] = len(all_lines)
    return {'all_lines':all_lines,
            'meta_data':meta_data}


    
