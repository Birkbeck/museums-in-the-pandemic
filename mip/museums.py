# -*- coding: utf-8 -*-

"""
functions to handle museum data
"""

import pandas as pd
import re 
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tldextract import extract
from utils import get_url_domain, get_url_domain_with_search

import logging
logger = logging.getLogger(__name__)

def load_museums_df_complete():
    """ Load and combine master list of museums for scraping and analysis """
    # TODO
    print("load_museums_df_complete")


def load_input_museums():
    """ Load MM museum data that includes ALL museums """
    fn = 'data/museums/museum_names_and_postcodes-2020-01-26.tsv'
    df = pd.read_csv(fn, sep='\t')
    df = exclude_closed(df)
    assert df["id"].is_unique
    if not df["Museum_Name"].is_unique:
        raise ValueError("Duplicate museum names exist.")
    print("loaded museums:",len(df), fn)
    return df


def get_museums_w_web_urls(data_folder=''):
    """ Get museums with website URLs """
    fn = data_folder+'data/museums/museum_websites_urls-samp.tsv'##DEBUG file should be -v3.tsv
    df = pd.read_csv(fn, sep='\t')
    print("museums urls:",fn)

    #df = df[df['url'].apply(is_valid_website)]
    #df = df.drop_duplicates(subset=['url'])
    #df['id_duplicated'] = df.duplicated(subset=['muse_id'])
    #assert df['url'].is_unique
    msg = "get_museums_w_web_urls Museums={} URLs={}".format(df.muse_id.nunique(), len(df))
    print(msg)
    logger.info(msg)
    return df


def combine_museums_w_web_urls():
    """
    Merge all museum website URL data (manual samples + predicted with random forests)
    @returns a data frame with urls
    """
    mdf = load_input_museums()

    fn = 'data/google_results/results_source_files/museum_urls_predicted.tsv'
    df = pd.read_csv(fn, sep="\t")
    df = df[df.predicted == 1]
    print(df.columns)
    subdf = df[['muse_id','musname','town','search_type','url']]
    print("loaded predicted museum URLs N =",len(subdf))

    # load manual sample 400
    sfn = 'data/samples/museums_manual_url_sample_400.tsv'
    mandf = pd.read_csv(sfn, sep="\t")
    mandf = mandf[['muse_id','musname','town','url','search_type','valid']]
    print(mandf.valid.describe())
    print(mandf.valid.value_counts())

    # load manual sample 60
    s60fn = 'data/samples/mip_data_sample_2020_01.tsv'
    man60df = pd.read_csv(s60fn, sep="\t")
    man60df = man60df[['mm_id','mm_name','website']]
    man60df.columns = ['muse_id','musname','url']
    man60df['valid']='T'
    man60df.loc[man60df['url'].isnull(),'valid'] = 'no_resource'
    man60df.loc[man60df['url'].isnull(),'url'] = 'no_resource'
    man60df['search_type']='website'
    man60df['town']=None

    # filter manual sample
    mandf = mandf[mandf.valid.isin(['T','no_resource'])]
    # replace url with "no resource" for museums without a website
    mandf.loc[mandf['valid']=='no_resource', 'url'] = 'no_resource'

    mandf = mandf[mandf.search_type == 'website']
    
    mandf = pd.concat([mandf, man60df])

    print("loaded manual museum URLs N =", len(mandf))

    # replace predicted with manual values - 400 sample
    preddf = subdf[~subdf.muse_id.isin(mandf.muse_id)]
    preddf['valid'] = 'pred'
    print('preddf',len(preddf))

    alldf = pd.concat([preddf,mandf])

    mdf2 = mdf.merge(alldf, left_on='id', right_on='muse_id', how='left')
    

    mdf2 = mdf2.sort_values("musname")
    
    # Museum_Name	id	location	year_closed	muse_id	musname	town	search_type	url	valid
    mdf2 = mdf2[['id','Museum_Name','location','url','valid']]
    mdf2.loc[mdf2['valid']=='T','valid'] = 'manual'
    mdf2.loc[mdf2['valid'].isnull(),'valid'] = 'no_pred'
    # rename cols
    mdf2.columns = ['muse_id','musname','town','url','url_source']
    mdf2 = mdf2.drop_duplicates(subset='muse_id')
    #mdf2['id_duplicated'] = mdf2.duplicated(subset=['muse_id'])
    assert mdf2['muse_id'].is_unique

    # save
    mdf2.to_csv('tmp/museum_websites_urls.tsv', sep='\t')
    mdf2.to_excel('tmp/museum_websites_urls.xlsx', index=False)
    return mdf2


def load_fuzzy_museums():
    """ Load MM museum data that includes ALL museums """
    df = pd.read_csv('data/google_results/results_source_files/google_extracted_results_reg.tsv.gz', sep='\t')
    df2=df[df['google_rank']<11]

    #df = exclude_closed(df)
    print("loaded urls:",len(df2))
    return df2


def load_input_museums_wattributes():
    """  """
    fn = 'data/museums/museums_wattributes-2020-02-23.tsv'
    df = pd.read_csv(fn, sep='\t')
    print(df.columns)
    # remove closed museums
    df = df[df.closing_date.str.lower() == 'still open']
    assert len(df) > 0 
    print("loaded museums w attributes (open):",len(df), fn)
    assert df["muse_id"].is_unique
    assert df["musname"].is_unique
    return df


def get_fb_tw_links():
    """ Extract FB and TW links from Google results """
    df = pd.read_csv('data/google_results/museum_searches_all-2021-02-16.tsv', sep='\t')
    df = df[df.search_type == 'website']
    print(df.columns)
    stats_df = pd.DataFrame()
    res_df = pd.DataFrame()
    for muse_id, muse_df in df.groupby('muse_id'):
        tw_df = muse_df[muse_df['domain'].str.contains('twitter')]
        fb_df = muse_df[muse_df['domain'].str.contains('facebook')]
        muse_name = muse_df['Museum_Name'].tolist()[0]
        assert muse_name
        tw_urls = tw_df['url'].tolist()
        fb_urls = fb_df['url'].tolist()
        row = pd.DataFrame({'muse_id':muse_id, 'links_n':len(muse_df), 'twitter_link_n':len(tw_df), 'facebook_link_n':len(fb_df)}, index=[muse_id])
        stats_df = stats_df.append(row)
        # save fb/tw links
        row = pd.DataFrame({'muse_id':muse_id, 'muse_name': muse_name, 'top_facebook': None, 'top_twitter': None}, index=[muse_id])
        if len(tw_urls)>0: row['top_twitter'] = tw_urls[0]
        if len(fb_urls)>0: row['top_facebook'] = fb_urls[0]
        res_df = res_df.append(row)
    print(res_df.describe())
    print(res_df.sum())
    print(res_df.isnull().sum(axis = 0))
    res_df.to_csv('tmp/google_tw_fb_links_df.tsv', index=False, sep='\t')


def load_extracted_museums(df):
    """ TODO: document """
    
  
    

    comparedf = pd.read_csv('data/websites_to_flag.tsv', sep='\t')
    
    urldict={}
    
    
    dfcheck=pd.DataFrame(columns=["google_rank","url","search", "muse_id", "location", "Museum_Name"])
    
    dfaccurate=pd.DataFrame(columns=["url","search", "muse_id", "location"])
    addedtocheck = False
    for item in df.iterrows():
        
        urlstring = item[1].url.split("/")[2]
        if item[1].google_rank ==1:
            addedtocheck = False
            if comparedf['website'].str.contains(urlstring).any():
                

                list1=[item[1].google_rank, item[1].url, item[1].search, item[1].muse_id, item[1].location, item[1].Museum_Name]
                dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id", "location", "Museum_Name"])
                dfcheck=dfcheck.append(dftoadd)
                addedtocheck=True
            else:
                list1=[item[1].google_rank,item[1].url, item[1].search, item[1].muse_id, item[1].location, item[1].Museum_Name]
                dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id", "location", "Museum_Name"])
                dfaccurate=dfaccurate.append(dftoadd)
                addedtocheck=False  
        else:
            if addedtocheck == True and (item[1].google_rank >1 and item[1].google_rank <11):
                list1=[item[1].google_rank, item[1].url, item[1].search, item[1].muse_id, item[1].location, item[1].Museum_Name]
                dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id", "location", "Museum_Name"])
                dfcheck=dfcheck.append(dftoadd)
                if not comparedf['website'].str.contains(urlstring).any():
                    list1=[item[1].google_rank,item[1].url, item[1].search, item[1].muse_id, item[1].location, item[1].Museum_Name]
                    dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id", "location", "Museum_Name"])
                    dfaccurate=dfaccurate.append(dftoadd)
            elif(item[1].google_rank >1 and item[1].google_rank <11):
                if not comparedf['website'].str.contains(urlstring).any():
                    list1=[item[1].google_rank,item[1].url, item[1].search, item[1].muse_id, item[1].location, item[1].Museum_Name]
                    dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id", "location", "Museum_Name"])
                    dfaccurate=dfaccurate.append(dftoadd)

    dfaccurate.to_csv('tmp/accurate_results.tsv', index=False, sep='\t')
    dfcheck.to_excel("tmp/tocheck_results_view.xlsx", index=False)
    return dfaccurate

def generate_samples():
        resultsdf=pd.read_csv("data/google_results/results_source_files/fuzzy_museum_scores_twitter.tsv", sep='\t')
        idealdf=load_museum_samples()
        search='twitter'
        score=0
        total=0
        res_df=pd.DataFrame()
        total=0
        for row in idealdf.iterrows():
            
            if row[1].search == search:
                total=total+1
                print(total)
                for item in resultsdf.iterrows():
                    rowfound=False
                    if item[1].id==row[1].muse_id:
                        rowfound=True
                        targetrow = pd.DataFrame({'muse_id':item[1].id, 'muse_name': item[1].Museum_Name, 'url': item[1].url, 'coorect_url':row[1].correct_url, 'google_rank': item[1].google_rank, 'score':item[1].score}, index=[item[1].id])
                        res_df = res_df.append(targetrow)
                    elif(rowfound==True):
                        break
        res_df.to_excel("tmp/result_sample.xlsx", index=False)
        return None

        
def exclude_closed(df):
    """ TODO: document """
    df=df[df.year_closed == '9999:9999']
    assert len(df) > 0
    return df

def compare_result_to_sample(search):
    """ Returns a percentage based on correctness of input file to the verified website sample. """
    resultsdf=pd.read_csv("data/google_results/results_source_files/fuzzy_museum_scores_twitter.tsv", sep='\t')
    idealdf=load_museum_samples()
    score=0
    total=0
    for row in idealdf.iterrows():
        if row[1].search == search:
            total=total+1
            print(total)
            itemfound=False
            for item in resultsdf.iterrows():
                if item[1].id==row[1].muse_id:
                    itemfound=True
                    correcturl=row[1].correct_url.lower()
                    itemurl=item[1].url.lower()
                    if search != 'web':
                        correcturl=correcturl.split('?')[0]
                        correcturl=correcturl.split('#')[0]
                        itemurl=itemurl.split('?')[0]
                        itemurl=itemurl.split('#')[0]
                    if not correcturl=='no_resource':
                        if get_url_domain_with_search(itemurl, search)==get_url_domain_with_search(correcturl, search):
                            print("yes "+get_url_domain_with_search(itemurl, search)+" "+get_url_domain_with_search(correcturl, search))
                            score=score+1
                        else:
                            print("no "+get_url_domain_with_search(itemurl, search)+" "+get_url_domain_with_search(correcturl, search))
                        print(item[1].url+" "+row[1].correct_url)
                    else:
                        print("no_resource")
                        total=total-1
                    break
            if itemfound==False:
                total=total-1
            itemfound=False
                
    percentage=score/(total/100)
    return percentage


def load_museum_samples():
    """ Load museum sample URLs for difficult museums """
    #fn = 'data/samples/manual_google_url_results_top10_2021-02-19.xlsx'
    fn2 = 'data/samples/sample_museum_search_with_loc.xlsx'
    fn3="data/samples/mip_data_sample_2020_01.tsv"
    #df = pd.read_excel(fn)
    df2=pd.read_excel(fn2)
    df3=pd.read_csv(fn3, sep='\t')
    musid=[]
    url=[]
    search=[]
    #for row in df.iterrows():
        #if not row[1].correct_url != row[1].correct_url:
            #musid.append(row[1].muse_id)
            #url.append(row[1].correct_url)
            #search.append("web")
    for row in df2.iterrows():
        
        musid.append(row[1].muse_id)
        url.append(row[1].correct_url)
        search.append(row[1].site)
    for row in df3.iterrows():
        if not row[1].website != row[1].website:
            musid.append(row[1].mm_id)
            url.append(row[1].website)
            search.append("web")
        if not row[1].facebook != row[1].facebook:
            musid.append(row[1].mm_id)
            url.append(row[1].facebook)
            search.append("facebook")
        if not row[1].twitter != row[1].twitter:
            musid.append(row[1].mm_id)
            url.append(row[1].twitter)
            search.append("twitter")
    sampledict={'muse_id': musid, 'correct_url':url, 'search_type':search}
    valid_websites_df=pd.DataFrame(sampledict)
    
    print("valid_websites_df", len(valid_websites_df))
    assert len(valid_websites_df) > 100
    return valid_websites_df

def load_manual_museum_urls():
    """ Load manually selected URLs for difficult museums """
    fn = 'data/samples/manual_google_url_results_top10_2021-02-19.xlsx'
    fn2 = 'data/samples/sample_museum_search_with_loc'
    df = pd.read_excel(fn)
    df.muse_id
    print(df.columns, len(df))
    print("museum IDs:",len(df.muse_id.value_counts()))
    manual_sites_df = df[df.correct_site.notnull()][['muse_id','correct_site']]
    manual_sites_df = manual_sites_df[manual_sites_df.correct_site.str.contains('http')]
    manual_sites_df.rename(columns={"correct_site":"url"})
    
    # get valid websites
    valid_websites_df = df[df.valid=='T'][['muse_id','url']]
    # concat results
    valid_websites_df = pd.concat([valid_websites_df, manual_sites_df], axis=0)
    print("valid_websites_df", len(valid_websites_df))
    assert len(valid_websites_df) > 100
    return valid_websites_df

def generate_derived_attributes_muse_df(df):
    print("generate_derived_attributes_muse_df")

    def get_region(x):
        x = x.replace("/England",'')
        r = x.split('/')
        reg = r[1]
        assert reg
        return reg

    def get_gov(x):
        s = x.split(":")
        assert s[0]
        return s[0]

    df['region'] = df['admin_area'].apply(get_region)
    df['gov'] = df['governance'].apply(get_gov)
    #print(df['gov'].value_counts())
    #print(df.sample(10))
    return df


def generate_stratified_museum_sample():
    """
    How to calculate SE/CI for this sample size: 
     http://sample-size.net/confidence-interval-proportion
    """
    print("generate_stratified_museum_sample")

    df1 = load_input_museums()
    print(df1.columns)
    df2 = load_input_museums_wattributes()
    print(df2.columns)
    print("difference between datasets:", set(df1.id).symmetric_difference(set(df2.muse_id)))
    # only select museums present in the initial dataset
    #df2 = df2[df2.muse_id.str.isin(df1.id.str)]

    # remove manual museums


    manual_museums_df = load_manual_museum_urls()
    print(manual_museums_df.columns)
    df = df2[~df2.muse_id.isin(manual_museums_df.muse_id)]
    df = generate_derived_attributes_muse_df(df)
    print("selected museums for sampling:", len(df))
    
    # generate sample
    fraction = .126
    sample_n = int(len(df) * fraction)
    print("sample_n", sample_n)
    cols = ["region","size","accreditation","gov"]
    sample_df = pd.DataFrame()
    for val, subdf in df.groupby(cols):
        sub_smpl_f = len(subdf) * fraction
        sub_smpl_n = int(round(sub_smpl_f,0))
        print(val, len(subdf), sub_smpl_f, sub_smpl_n)
        sample_df = sample_df.append(subdf.sample(sub_smpl_n, random_state=32))
    
    print("sample_df", len(sample_df))
    sample_df['valid'] = ''
    assert sample_df.muse_id.is_unique
    fout = 'tmp/museums_stratified_sample_{}.tsv'.format(len(sample_df))
    sample_df.to_csv(fout, sep="\t", index=False)
    print(fout)


def get_weighted_sum(musname, weighteddict):
    """ @returns the sum of the weights for museum name """
    weightsum=0
    for word in musname:
        if word !='':
        
            if word in weighteddict.keys():
                weightsum = weightsum+weighteddict[word]
            else:
                weightsum=weightsum+1
    return weightsum

def get_fuzzy_musname_score(musname, str_from_url, weighteddict, weightsum):
    """ @returns the normalised weighted score of the museum name """
    poolelementscore=0
    
    for word in musname:
        if word!='':
        
    
            score = fuzz.partial_ratio(word, str_from_url)
            if word in weighteddict.keys():
                score = score*(1/(weightsum/weighteddict[word]))
            else:
                score = score*(1/(weightsum/1))
            poolelementscore=poolelementscore+score
            
        
            
    return poolelementscore
def get_fuzzy_string_score(string1, string2):
    """ @returns a score based on an abreviation it generates from the museum name """
    
    
    if len(string1)<3 or len(string2)<3:
        return 0
    score = fuzz.partial_ratio(string1, string2)
    return score


def hasvisit(url, search):
    """ @returns True if the given url has visit in the domain """
    domain=get_url_domain_with_search(url, search)
    if 'visit' in domain.lower():
        return True
    else:
        return False

def hasmuseum(url, search):
    """ @returns True if the given url has museum in the domain """
    domain=get_url_domain_with_search(url, search)
    if 'museum' in domain.lower():
        return True
    else:
        return False

def haslocation(url, location):
    """ @returns True if the given url has location within it """
    
    if isinstance(location, float) or isinstance(location, int):
        location=""
    if location.lower() in url.lower():
        return True
    else:
        return False

def striphtml(string1):
    string1=string1.replace('www','')
    string1=string1.replace('https','')
    string1=string1.replace('http','')
    string1=string1.replace('.com','')
    string1=string1.replace('.org.uk','')
    string1=string1.replace('.org','')
    string1=string1.replace('.co.uk','')
    string1=string1.replace('.gov','')
    string1=re.sub(r'\W+', '', string1).lower()
    return string1

def get_exact_match(string1,string2):
    
    string1=striphtml(string1)
    string2=striphtml(string2)
    score=fuzz.ratio(string1,string2)
    return score
def get_musname_pool(mname, location):
    joiningwords=["or", "the", "a", "for", "th", ""]
    
    mnamepool={}
    namepool={}
    mname=mname.lower()
    if isinstance(location, float) or isinstance(location, int):
        location=""
    location=location.lower()
    mnamepool['musname']=mname.split(" ")
    mnamewithmus=mname+" museum"
    mnamepool['mnamewithmus']=mnamewithmus.split(" ")
    mnamewithloc=mname+" "+location
    mnamepool['musnamewithloc']= mnamewithloc.split(" ")
    mnamewlocandmus=mname+" "+location+" museum"
    mnamepool['musnamewlocandmus']=mnamewlocandmus.split(" ")
    for key, value in mnamepool.items():
        newvalue=''
        newvaluewand=''
        abbreviation=''
        abbreviationwand=''
        for word in value:
            if  word not in joiningwords:
                                  
            
                if word == 'and':
                    newvaluewand=newvaluewand+' &'
                    abbreviationwand=abbreviationwand+"&"
                elif re.search("^[0-9][0-9]*th", word) or re.search("^[0-9][0-9]*st", word):
                    a = re.sub('[^0-9]','', word)
                    abbreviationwand=abbreviationwand+a
                    abbreviation=abbreviation+a
                    newvalue=newvalue+" "+word
                    newvaluewand=newvaluewand+" "+word
                else:
                    abbreviationwand=abbreviationwand+word[0]
                    abbreviation=abbreviation+word[0]
                    newvalue=newvalue+" "+word
                    newvaluewand=newvaluewand+" "+word
        namepool[key]=newvalue.split(' ')
        newkey = key+'wand'
        namepool[newkey]=newvaluewand.split(' ')
        abbrevkey=key+'abbrev'
        namepool[abbrevkey]=abbreviation
        abbrevkeywand=key+'abbrevwand'
        namepool[abbrevkeywand]=abbreviationwand
    
    
    return namepool


def generate_weighted_fuzzy_scores(mname, str_from_url, weighteddict, location, domainonly, search):
    """ @returns maximum score for fuzzy string match with each word weighted based on number of occurances in all museum names """
    
    if domainonly==True:
        str_from_url=get_url_domain_with_search(str_from_url, search)
    str_from_url=str_from_url.lower()
    scores = []
    if isinstance(location, float) or isinstance(location, int):
        location=""
    
    for key in mname:
        if location != "":
            if 'loc'in key:
                if 'abbrev' in key:
                    score=get_fuzzy_string_score(mname[key],str_from_url)
                    scores.append(score)
                    #print(key)
                    #print(score)
                else:
                    weightsum=get_weighted_sum(mname[key], weighteddict)
                    score=get_fuzzy_musname_score(mname[key], str_from_url, weighteddict, weightsum)
                    scores.append(score)
                    #print(key)
                    #print(score)
        
                
        else:
            if 'loc' not in key:
                if 'abbrev' in key:
                    score=get_fuzzy_string_score(mname[key],str_from_url)
                    scores.append(score)
                    #print(key)
                    #print(score)
                else:
                    weightsum=get_weighted_sum(mname[key], weighteddict)
                    score=get_fuzzy_musname_score(mname[key], str_from_url, weighteddict, weightsum)
                    scores.append(score)
                    #print(key)
                    #print(score)
           
            
    
    

    
    max_score = max(scores)
    return max_score


def generate_string_pool_from_museum_name(mname):
    """ @returns variants of strings for fuzzy match on museum names """
    assert len(mname)>2
    pool = []
    joiningwords=["or", "the", "a", "for", "th"]
    pool.append(mname)
    pool.append(mname+" museum")
    
    mnamelist= mname.rstrip().split(" ")
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords and word != "and" :
            newphrase = newphrase+word+" "
    pool.append(newphrase.rstrip())
    pool.append(newphrase+"museum")
    pool.append(newphrase.replace(' ',''))
    pool.append(newphrase.replace(' ','')+"museum")
    
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords:
            if word == "and":
                newphrase = newphrase+"& "
            else:
                newphrase = newphrase+word+" "
    pool.append(newphrase.rstrip())
    pool.append(newphrase+"museum")
    pool.append(newphrase.replace(' ',''))
    pool.append(newphrase.replace(' ','')+"museum")
    
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords and word != "and" :
            newphrase = newphrase+word[0]
    pool.append(newphrase)
    pool.append(newphrase+" museum")
    pool.append(newphrase+"museum")
    
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords:
            if word == "and":
                newphrase = newphrase+"&"
            else:
                newphrase = newphrase+word[0]
    pool.append(newphrase)
    pool.append(newphrase+" museum")
    pool.append(newphrase+"museum")
    
    return pool


def fuzzy_string_match(a, b):
    """ @returns a similarity score based on the extent to which a is found in b"""
    assert len(a) > 0
    assert len(b) > 0
    
    ratio = fuzz.token_sort_ratio(a, b)
    return ratio
    # https://towardsdatascience.com/fuzzy-string-matching-in-python-68f240d910fe


    
def generate_combined_dataframe():
    print("generate_combined_dataframe")
    scrapetarget=[]
    searchtype=[]
    df_mus = pd.read_csv('data/google_results/results_source_files/google_extracted_results_reg.tsv.gz', sep='\t')
    for item in df_mus.iterrows():
        scrapetarget.append('web')
        searchtype.append('regular')
    df_mus['scrape_target']=scrapetarget
    df_mus['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]

    df_mus_exact=pd.read_csv('data/google_results/results_source_files/google_extracted_results_exact.tsv.gz', sep='\t')
    for item in df_mus_exact.iterrows():
        scrapetarget.append('web')
        searchtype.append('exact')
    df_mus_exact['scrape_target']=scrapetarget
    df_mus_exact['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_facebook=pd.read_csv('data/google_results/results_source_files/google_extracted_results_facebook.tsv.gz', sep='\t')
    for item in df_facebook.iterrows():
        scrapetarget.append('facebook')
        searchtype.append('location')
    df_facebook['scrape_target']=scrapetarget
    df_facebook['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_facebook_noloc=pd.read_csv('data/google_results/results_source_files/google_extracted_results_facebook_noloc.tsv.gz', sep='\t')
    for item in df_facebook_noloc.iterrows():
        scrapetarget.append('facebook')
        searchtype.append('regular')
    df_facebook_noloc['scrape_target']=scrapetarget
    df_facebook_noloc['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_twitter=pd.read_csv('data/google_results/results_source_files/google_extracted_results_twitter.tsv.gz', sep='\t')
    for item in df_twitter.iterrows():
        scrapetarget.append('twitter')
        searchtype.append('location')
    df_twitter['scrape_target']=scrapetarget
    df_twitter['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_twitter_noloc=pd.read_csv('data/google_results/results_source_files/google_extracted_results_twitter_noloc.tsv.gz', sep='\t')
    for item in df_twitter_noloc.iterrows():
        scrapetarget.append('twitter')
        searchtype.append('regular')
    df_twitter_noloc['scrape_target']=scrapetarget
    df_twitter_noloc['search_variety']=searchtype
    finaldf=df_mus.append(df_mus_exact)
    finaldf=finaldf.append(df_facebook)
    finaldf=finaldf.append(df_facebook_noloc)
    finaldf=finaldf.append(df_twitter)
    finaldf=finaldf.append(df_twitter_noloc)
    finaldf.to_csv('data/google_results/google_results_all_02_03_2021.tsv', index=False, sep='\t')


def has_correct_url(url, correct_url):
    if not isinstance(correct_url, float) or isinstance(correct_url, int):
        if url.lower()==correct_url.lower():
            return 1
        else: return 0
    else:
        return ''

    
def match_museum_name_with_string(mname, str_from_url):
    """@returns max similarity score between variants of mname and str_from_url)"""
    
    pool = generate_string_pool_from_museum_name(mname)
    scores = []
    print(mname)
    print(str_from_url)
    for name_variant in pool:
        
        score = fuzzy_string_match(name_variant, str_from_url)
        if score is not None:
            scores.append(score)
    max_score = max(scores)
    return max_score

def get_fuzzy_string_match_scores(musdf, search):
    scorerow=[]
    museweight = generate_weighted_museum_names()
    for row in musdf.iterrows():
        if search=='web':
            urlstring=row[1].url
        else:
            urlstring=row[1].url.split("/")[3].lower()
        if(urlstring=='events'):
            urlstring=row[1].url.split("/")[4].lower()
        
        musename = row[1].Museum_Name.lower()
        location=row[1].location
        if not isinstance(location, float) and not isinstance(location, int):
            location=location.lower()
        else:
            location=""
        if urlstring !='':
            scorerow.append(generate_weighted_fuzzy_scores(musename, urlstring, museweight, location))
        else:
            scorerow.append(0)
    musdf['score']=scorerow
    finaldf=pd.DataFrame()
    
    newdf=musdf.sort_values(['muse_id','score','google_rank'], ascending=[True,False,True])
    finaldf=pd.concat([finaldf, newdf])
    finaldf.to_csv('tmp/fuzzy_museum_scores.tsv', index=False, sep='\t')
    return None

def generate_weighted_museum_names():
    df1 = pd.read_csv('data/museums/museum_names_and_postcodes-2020-01-26.tsv', sep='\t')
    joiningwords=["or", "the", "a", "for", "th",""]
    weighteddict = {}
    for row in df1.iterrows():
        musname = row[1]["Museum_Name"].split(" ")
        for word in musname:
            nameword = word.lower()
            if nameword not in joiningwords:
                if nameword in weighteddict:
                    weighteddict[nameword] = weighteddict[nameword]+1
                else:
                    weighteddict[nameword]=1
    for item in weighteddict:
        weighteddict[item]=1/weighteddict[item]
    return weighteddict
            

def combinedatasets():
    df1 = pd.read_csv('tmp/all_museum_id.tsv', sep='\t')
    df2 = pd.read_csv('tmp/all_museum_data.csv', sep=',')
    df3=pd.merge(df1, df2, on='musname')
    df3.to_csv('tmp/museums_wattributes-2020-02-23.tsv', index=False, sep='\t')
    return None


def load_all_google_results():
    df = pd.read_csv('data/google_results/google_results_all_02_03_2021.tsv.gz', sep='\t')
    print("load_all_google_results", len(df))
    print(df.describe())
    print(df.columns)
    print(df.search_type.value_counts())
    print(df.search_variety.value_counts())
    print(df.year_closed.value_counts())
    print(df.scrape_target.value_counts())
    return df


def join_museum_info(df, muse_id_column):
    """ Add museum info based on muse_id_column in DF. 
    Useful to make DF more interpretable """
    assert len(df) > 0
    museums_df = load_input_museums()
    mdf = df.merge(museums_df, left_on=muse_id_column, right_on='id', how="left")
    assert len(mdf) == len(df)
    return mdf
