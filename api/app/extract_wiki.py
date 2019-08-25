import os
import re
import plac

from bs4 import BeautifulSoup
from sklearn.externals import joblib


@plac.annotations(
    raw_files_dir = ('directory containing raw xml dumps', 'positional', None, str),
    file_to_save = ('file name to save the extracted data into', 'positional', None, str),
    )


def main(raw_files_dir, file_to_save):

    files = os.listdir(raw_files_dir)

    extracts = []
    counter = 0

    for file in files:
        content = open(raw_files_dir + '/' + file, 'r').read()
        soup = BeautifulSoup(content, 'xml')

        exts = extract_sections(soup)

        extracts.append({'file': file,
                         'extracts': exts})

        counter += 1
        print('Extracted information from {}/{} articles'.format(counter, len(files)), end = '\r')


    joblib.dump(extracts, file_to_save)


def extract_sections(soup):
    '''
    Extracts headers and their corresponding sections enclosed in <p></p> tags

    Arguments:
    soup: a BeautifulSoup object corresponding to the xml file of the wiki page

    Returns:
    Json containing information regarding the sections in the wiki page
    JSON Structure:
    {'title': Title of wiki page,
     'introduction': Introduction of wiki page,
     'sections': [{'header': sub_header,
                   'content': information related to the header},
                   {'header': sub_header,
                   'content': information related to the header}]}
    '''
    
    title_tag = soup.find('title')
    all_contents = ' '.join([p.text for p in soup.find_all('p')])
    
    header_tags = soup.find_all('h2')
    
    headers = []
    paras = []
    
    for header in header_tags[::-1]:
        
        try:  
            ps = header.find_all_next('p')

            text = ' '.join([p.text for p in ps])

            headers.append(header.find('span').text)
            paras.append(text)
        except:
            pass
        
    assert len(headers) == len(paras), 'Length of headers does not match length of paras'
    
    headers, paras = process_tags(headers, paras)
    
    title = title_tag.text.split('-')[0].strip()
    try:
        intro = all_contents.replace(paras[-1], '').strip()
    except:
        intro = all_contents.strip()
    
    sections = organize_sections(headers, paras)
        
    return {'title':title, 'introduction': intro, 'sections': sections}



def process_tags(headers, paras):
    '''
    Processes the texts extracted from <p></p> tags to create mapping between headers and corresponding information
    '''
    new_paras = []
    
    for idx in range(len(headers)):
        if idx == 0:
            new_paras.append(paras[idx])
        else:
            para = paras[idx]
            para = para.replace(paras[idx-1], '').strip()
            new_paras.append(para)
    
    headers = headers[::-1]
    new_paras = new_paras[::-1]
    
    return headers, new_paras



def organize_sections(headers, paras):
    '''
    Organizes the headers and corresponding information in a json format
    '''
    sections = []
    
    for idx in range(len(headers)):
        sections.append({'header': headers[idx], 
                         'content': paras[idx]})
        
    return sections


if __name__ == '__main__':
    plac.call(main)