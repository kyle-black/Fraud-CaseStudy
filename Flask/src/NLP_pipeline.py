import pandas as pd
from bs4 import BeautifulSoup

def get_text_col(incoming):
    # Argument: incoming raw dataframe
    # Returns: augmented dataframe with 'all_text' column (contains docs)
  dat = incoming.copy()
  dat['description']=dat['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
  dat['org_desc']=dat['org_desc'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
#   dat['Fraud'] = dat['acct_type'].str.contains('fraud')
  text_cols = ['description','org_desc','email_domain','org_name','payee_name','name','venue_address','venue_name','venue_state']
  for i in text_cols:
    dat[i] = dat[i].apply(str)
  dat['all_text'] = dat[text_cols].apply(' '.join, axis=1)  
  
  return dat

if __name__ == 'main':
    print('NLP_pipeline called')