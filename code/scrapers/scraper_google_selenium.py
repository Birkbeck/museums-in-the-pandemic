# -*- coding: utf-8 -*-
# Andrea Ballatore
#
# Web scraper

import time
import uuid
import json
import os
import random
import calendar
import datetime
import requests
import sys
import urllib
import traceback
import subprocess
import pandas as pd
import numpy as np
from webbot import Browser


GROUP_TYPES = ['General group',"Jobs group",'Buy and sell group',
    'Gaming group','Social learning group','Parenting group','Custom group',
    'Teams & projects group','Work group']

GOOGLE_PAUSE_SECS = 3

VPN_SERVERS = ['uk-manchester','uk-london','uk-southampton','ireland','belgium',
                'isle-of-man','luxembourg','austria']

def gen_random_page_fn():
    import uuid
    fn = 'tmp/pages_dump/'+str(uuid.uuid4())+'.html'
    return(fn)


def read_file(fn):
    with open(fn, 'r') as content_file:
        content = content_file.read()
        return(content)


# extract numbers from text fields
def extract_int(s, no_string):
    import re
    if no_string and no_string in s.lower():
        i = 0
    else: i = int(re.findall(r'\d+', s.replace(',',''))[0])
    assert i >= 0
    return i


# extract numbers from text fields
def extract_hum_number(s):
    s = s.lower()
    import re
    numbs = re.findall(r"[-+]?\d*\.\d+|\d+",s)
    if len(numbs) == 0: 
        return None

    assert len(numbs)==1
    flnum = float(numbs[0])

    if 'k' in s:
        flnum = flnum * 1e3
    if 'm' in s:
        flnum = flnum * 1e6

    return(flnum)


def extract_fb_data_from_fb_page(html, fn):
    from bs4 import BeautifulSoup
    #print(fn)
    group_uid = fn.replace('.html','').replace('tmp/pages_dump_fb/','')
    #if group_uid != 'fbgr_001619' and group_uid != 'fbgr_000757':
    #    return None
    #print(group_uid)

    if "this content isn't available at the moment" in html.lower() or "something went wrong" in html.lower():
        print("Deleted/empty group",fn)
        df = pd.DataFrame({'group_uid':[group_uid], 'found':[False], 'html_file':[fn]}, index=[group_uid])
        return df
    
    soup = BeautifulSoup(html, 'html.parser')
    res = soup.select("span.d2edcug0.hpfvmrgz.qv66sw1b.c1et5uql.oi732d6d.ik7dh3pa.fgxwclzu.a8c37x1j.keod5gw0.nxhoafnm.aigsh9s9.d9wwppkn.fe6kdd0r.mau55g9w.c8b282yb.iv3no6db.jq4qci2q.a3bd9o3v.knj5qynh.oo9gr5id.hzawbc8m")
    if len(res) == 0: 
        raise Exception('data not found in page',fn)
        #print('>> data not found in page:',fn)  # DEBUG
        #return pd.DataFrame({'group_uid':[group_uid], 'found':[False], 'html_file':[fn]}, index=[group_uid])  # DEBUG
    
    i = -1
    for el in res:
        i += 1
        #print(i, el.text)
    del i

    assert len(res)==7 or len(res)==8 or len(res)==6 or len(res)==9, len(res)
    #print(len(res))
    # init variables
    place = None
    #tags = None
    gtype = None
    dailyposts_str = None
    
    # find easy fields
    for r in res:
        t = r.text.strip()
        if t in ["Public","Private"]:
            priv = t
        elif t in ["Visible"]:
            vis = t
        elif t in GROUP_TYPES:
            gtype = t
        elif 'post today' in t or 'posts today' in t:
            dailyposts_str = t
        elif 'total member' in t:
            members_str = t
    del r,t
    # find description
    desc = res[0].text.strip()
    if desc in ["Public","Private"]:
        desc = ''

    # find place
    place = res[3].text.strip()
    if place in GROUP_TYPES:
        place = None
    if not place:
        place = res[2].text.strip()
        if place == 'Visible':
            place = None

    assert vis == 'Visible',vis
    assert priv in ['Public','Private'],priv
    assert gtype in GROUP_TYPES,'group not found ' + str(gtype) +' '+ fn

    group_name = soup.select("span.d2edcug0.hpfvmrgz.qv66sw1b.c1et5uql.oi732d6d.ik7dh3pa.fgxwclzu.a8c37x1j.keod5gw0.nxhoafnm.aigsh9s9.ns63r2gh.fe6kdd0r.mau55g9w.c8b282yb.rwim8176.m6dqt4wy.h7mekvxk.hnhda86s.oo9gr5id.hzawbc8m")
    assert len(group_name)==1
    group_name = group_name[0].text.strip()

    # get secondary info
    sinfo = soup.select("span.d2edcug0.hpfvmrgz.qv66sw1b.c1et5uql.oi732d6d.ik7dh3pa.fgxwclzu.a8c37x1j.keod5gw0.nxhoafnm.aigsh9s9.d9wwppkn.fe6kdd0r.mau55g9w.c8b282yb.mdeji52x.e9vueds3.j5wam9gi.knj5qynh.m9osqain.hzawbc8m")
    i = -1
    for el in sinfo:
        i += 1
        #print(i, el.text)
    del i
    assert len(sinfo) == 5 or len(sinfo) == 6, len(sinfo) #  or len(sinfo) == 6
    if len(sinfo) == 5:
        creation_date_str = sinfo[2].text.strip()
        lastmonth_posts_str = sinfo[3].text.strip()
        week_members_str = sinfo[4].text.strip()
    if len(sinfo) == 6:
        creation_date_str = sinfo[3].text.strip()
        lastmonth_posts_str = sinfo[4].text.strip()
        week_members_str = sinfo[5].text.strip()

    members_n = extract_int(members_str, None)
    lastmonth_posts = extract_int(lastmonth_posts_str, 'no posts')
    week_members_new = extract_int(week_members_str, 'no new m')
    dailyposts = extract_int(dailyposts_str, 'no new post')

    # extract date
    import dateutil.parser
    creation_date_str2 = creation_date_str.replace('Group created on','')
    creation_date_str2 = ' '.join(creation_date_str2.replace('See more','').strip().split(' ')[0:3])
    creation_date = dateutil.parser.parse(creation_date_str2, fuzzy_with_tokens=True)[0]

    # get moderation rules
    mod = soup.select('div.rq0escxv.l9j0dhe7.du4w35lb.hybvsw6c.ue3kfks5.pw54ja7n.uo3d90p7.l82x9zwi.ni8dbmo4.stjgntxs.k4urcfbm.sbcfpzgs')
    mod_str = None
    for el in mod:
        if 'Group rules from the admins' in el.text.strip():
            mod_str = el.get_text(separator="\t").strip()
    
    # get members from another field (fix for FB bug that shows 1 member for a group)
    memb = soup.select('span.a8c37x1j.ni8dbmo4.stjgntxs.l9j0dhe7')
    assert len(memb)>0, 'members extra info not found ' + fn
    #print('\n\n')
    for el in memb:
        txt = el.text.strip()
        if 'Members' in txt:
            txt = txt.replace('Members','')
            members_n_2 = extract_hum_number(txt)
            if members_n_2 is not None and members_n_2 >= 0:
                # fix for Facebook bug on members
                members_n = max(members_n, members_n_2)
                del members_n_2
        del txt

    # build result
    infodf = pd.DataFrame({'group_uid':[group_uid],
                'group_name':[group_name],
                'description': [desc],
                'privacy':[priv],
                'found':[True],
                'html_file':[fn],
                'visibility':[vis],
                'creation_date_str':[creation_date_str],
                'creation_date':[creation_date],
                'creation_year':[creation_date.strftime('%Y')],
                'creation_yymm':[creation_date.strftime('%Y-%m')],
                'fb_place':[place],
                'group_type':[gtype],
                'members_str':[members_str],
                'members_n':[members_n],
                'week_members_str':[week_members_str],
                'week_members_new':[week_members_new],
                'lastmonth_posts_str':[lastmonth_posts_str],
                'lastmonth_posts':[lastmonth_posts],
                'dailyposts_str':[dailyposts_str],
                'dailyposts':[dailyposts],
                'group_rules_str':[mod_str],
                'group_name2':[group_name],
                'description2': [desc],
                'group_uid2':[group_uid]
            }, index=[group_uid])
    return(infodf)


def extract_fb_links_from_google_page(html, fn):
    from bs4 import BeautifulSoup    
    soup = BeautifulSoup(html, 'html.parser')

    place_code = fn.replace('.html','').replace('tmp/pages_dump/','')
    links = []
    for a_tag in soup.find_all("a"):
        href = a_tag.attrs.get("href")
        links.append(href)
    
    # keep only fb links
    links = [clean_group_url(l) for l in links if is_fb_link(l)]
    links = [l for l in links if len(l) > 10] # remove empty string
    links = pd.Series(links).drop_duplicates().tolist() # get unique

    if len(links) == 0: 
        return None

    # build results
    ids = [place_code+'_'+'{:03d}'.format(l+1) for l in range(len(links))]
    granks = range(1,len(links)+1)
    assert len(granks)==len(links) and len(granks)==len(links) and len(ids)==len(links)

    df = pd.DataFrame({'place_code': place_code, 'google_rank': granks,
        'html_file': fn,'url':links},index=ids)
    assert len(df.index)==len(links)
    return df


def write_file( content, fn ):
    file1 = open(fn,"w") #write mode 
    file1.write(content) 
    file1.close() 


def get_url( url, session ):
    p = session.get( url )
    print(p.content)


def random_sleep(min=0,max=2):
    n = random.randint(min*1000,max*1000)/1000 
    print("\trandom_sleep secs",n)
    time.sleep( n )


# l = d m y
def format_date_lexis(dd):
    s = str(dd.day)+'%2F'+str(dd.month)+'%2F'+str(dd.year)
    return(s)


def get_last_day_month(y,m):
    import calendar
    assert m in range(1,13,1)
    last_day = calendar.monthrange(y,m)[1]
    return(last_day)


def load_list_from_file(fn):
    cont = read_file(fn).strip()
    #print(cont)
    lcont = cont.split('\n')
    # remove commented lines
    import re
    out = []
    for l in lcont:
        m = re.match(r'^([^#]*)#(.*)$', l)
        if m:  # The line contains a hash / comment
            l = m.group(1)
        else: out.append(l)
    out = [x for x in out if x]
    return(out)


def get_timestamp():
    import datetime
    return(str(datetime.datetime.now()))

# =============== MAIN =============== #

# set VARIABLES

def tests():
    web = Browser()
    # END TESTS


def get_fb_page(web, url):
    print("get_fb_page", url)
    while True:
        try:
            random_sleep(0,1)
            web.go_to(url)
            random_sleep(0,1)
            tmp_html = web.get_page_source()
            # detect issues
            if 'you must log in to continue' in tmp_html.lower() or 'redirected you too many times' in tmp_html.lower():
                raise Exception('Facebook is blocking: '+url)
            if 'your request couldn''t be process' in tmp_html.lower():
                raise Exception('Facebook didn''t respond: '+url)

            n = tmp_html.lower().count('see more')
            if n > 2:
                web.click('See more', tag='div', multiple = True)
            random_sleep(0.2,1)
            html = web.get_page_source()
            return web,html
        except Exception as e:
            print(e)
            print('failed to download page, changing VPN')
            web.quit()
            vpn_random_region()
            random_sleep(1,1)
            web = Browser()
            #web = Web # init_google_browser()
            random_sleep(2,3)


def gen_google_url(query_str):
    n_results = 50
    query_enc = urllib.parse.quote_plus(query_str)
    url = 'https://www.google.co.uk/search?q='+query_enc+'&num='+str(n_results)+'&hl=en-GB'
    print(url)
    return url


def run_google_query(web, querytext):
    assert len(querytext)>3
    print("run_google_query", querytext)
    # get google url
    queryurl = gen_google_url(querytext)

    if True: # run webbot
        web.go_to(queryurl)
        #web.type(querytext, classname="form-input", number=1)
        #random_sleep(0,1)
        #web.click('Google Search', classname="form-input", number=2)
        #web.press(web.Key.TAB)
        random_sleep(1,3)
        web.press(web.Key.ENTER)
        random_sleep(GOOGLE_PAUSE_SECS,GOOGLE_PAUSE_SECS*1.5)
        html = web.get_page_source()
    else:
        # run VPN url
        random_sleep(0,2)
        html = get_url_vpn(queryurl)

    if 'unusual traffic from your computer network' in html.lower():
        raise Exception('Google is blocking. '+querytext)

    return html


def click_on_google_eula(web):
    print("click_on_google_eula")
    random_sleep(1,2)
    # clear user agreement
    web.press(web.Key.TAB)
    random_sleep(0,1)
    #web.press(web.Key.TAB)
    #random_sleep(1,2)
    web.press(web.Key.ENTER)
    random_sleep(0,1)


def get_url_vpn(url):
    pia_socks5 = 'socks5h://x1936726:9CammAz8bs@proxy-nl.privateinternetaccess.com:1080'
    proxies = {'http': pia_socks5,'https': pia_socks5}
    r = requests.get(url, proxies=proxies )
    if r.status_code == 429: 
        raise Exception("429 Too Many Requests")
    assert r.status_code == 200, 'failed to download '+str(r.status_code) + ' ' + url
    return r.text

def restart_browser(web):
    web.quit()

def init_google_browser():
    from webbot import Browser
    web = Browser()
    start_url = "https://www.google.co.uk/"
    random_sleep(0,1)
    web.go_to(start_url)
    click_on_google_eula(web)
    random_sleep(0,2)
    return web

def vpn_off():
    ret = run_os_command('piactl disconnect')
    print(ret)

def vpn_on():
    ret = run_os_command('piactl connect')
    print(ret)

def vpn_go_region(reg):
    assert reg in VPN_SERVERS
    ret = run_os_command('piactl set region '+reg)
    print(">> vpn_go_region:",reg)
    return ret

def run_os_command(cmd):
    ret = subprocess.check_output(cmd, shell=True)
    ret = ret.decode("utf-8").strip()
    return ret

def is_fb_link(url):
    if not url: return False
    if url == '': return False
    b = 'facebook.com/groups/' in url
    b = b and not ('webcache.googleusercontent.com' in url)
    b = b and not ('translate.google.' in url)
    return b

def vpn_random_region():
    vpn_go_region(random.choice(VPN_SERVERS))

def vpn_is_on():
    ret = run_os_command('piactl get connectionstate') == 'Connected'
    return ret

def scrape_google_london_place_groups(topicsdf):
    # NOTE: this function needs PIA VPN to work
    assert vpn_is_on(),'VPN must be on'

    # init browser and google settings
    web = init_google_browser()
    
    outdf = pd.DataFrame()
    # scan place names
    for index, row in topicsdf.iterrows():
        place_id = row['place_code'].strip()
        fn = 'tmp/pages_dump/'+place_id+'.html'
        if os.path.isfile(fn):
            print('file found, skip')
            continue
        
        print(index,row)
        if index % 1000 == 0:
            print("long pause idx=",index)
            random_sleep(60,600)
        
        query = row['place_name'].strip() + " site:en-gb.facebook.com/groups"
        
        # get data from google
        found = False
        while not found:
            try: 
                html = run_google_query(web, query)
                found = True
            except Exception as e:
                print(e)
                print('failed to download page, changing VPN')
                web.quit()
                vpn_random_region()
                random_sleep(1,1)
                web = init_google_browser()
                random_sleep(2,2)
        
        write_file(html, fn)
        outdf = pd.concat([outdf, pd.DataFrame({'google_query': [query],'file':fn},
                       index=[place_id])])
        print("\t", fn)
    dffn = 'tmp/scraping_google_out.csv'
    outdf.to_csv(dffn, index_label="place_code")
    print("scraping complete.",dffn)
    return


def extract_fbgroup_info(foldfn):
    print("extract_fbgroup_info", foldfn)
    import glob
    outdf = pd.DataFrame()
    i = 0
    
    for fn in sorted(glob.glob(foldfn+"/*.html")):
        i += 1
        if i % 1000 ==0: print('\t',i)
        html = read_file(fn)

        #if "did not match any documents" in html.lower():
            # empty results from Google
        #    print(">> No results in ",fn)
        #    continue
        #print(fn)
        page_df = extract_fb_data_from_fb_page(html, fn)
        assert page_df is not None, fn
        outdf = pd.concat([outdf, page_df])
        #if i > 50: break # DEBUG
    outfn = 'tmp/fb_groups_info_df'
    print(outfn)
    #outdf = outdf.sample(100) # DEBUG
    # save files
    outdf.to_csv(outfn+'.csv', index_label='row_id')
    #outdf.to_excel(outfn+'.xlsx', index_label='row_id')
    outdf.to_pickle(outfn+'.pik')


def extract_google_results(foldfn):
    print("extract_google_results", foldfn)
    import glob
    outdf = pd.DataFrame()
    i = -1
    for fn in sorted(glob.glob(foldfn+"/*.html")):
        i += 1
        #if i > 10: continue # DEBUG
        html = read_file(fn)
        if "did not match any documents" in html.lower():
            # empty results from Google
            print(">> No results in ",fn)
            continue
        # extract data from html
        page_df = extract_fb_links_from_google_page(html, fn)
        if page_df is not None:
            # build results
            outdf = pd.concat([outdf, page_df])
        else: 
            print('No FB links found in', fn)
    outfn = 'tmp/fb_groups_from_google.csv'
    print(outfn)
    outdf.to_csv(outfn,index_label='row_id')


def clean_group_url(url):
    # extract Facebook group url
    import re
    url = url.replace('/url?q=','')
    #print(url[0:3])
    if url[0:3]=='/se': return ''
    idx = [m.start() for m in re.finditer('/', url)]
    if len(idx) < 5: return ''
    end_str = idx[4]
    return url[0:end_str]


def analyse_facebook_groups_info(fn_pik):
    print("analyse_facebook_groups_info",fn_pik)
    df = pd.read_pickle(fn_pik)
    print(len(df))
    print(df.info())
    print(df.describe())
    

def scrape_facebook_groups_info(fn):
    print("scrape_facebook_groups_info")
    df = pd.read_csv(fn)
    print(len(df), df.columns)
    group_unique_urls = sorted(df["url"].unique())
    print(len(group_unique_urls))
    offset = 0
    if len(sys.argv) >= 2:
        offset = int(sys.argv[1])
        print('offset', offset)

    group_unique_urls = [x for x in group_unique_urls if len(x)>10] # remove empty string
    ids = range(1,len(group_unique_urls)+1)
    ids = ['fbgr_{:06d}'.format(i) for i in ids]    
    groups_df = pd.DataFrame({"url":group_unique_urls},index=ids)
    groups_df.to_csv('tmp/fb_groups_unique_ids_df.csv',index_label='row_id')
    assert len(groups_df.index)>0
    
    from webbot import Browser
    
    web = Browser()
    df = groups_df
    if offset > 0:
        df = groups_df.tail(offset) # this is used to split jobs
    
    for index, row in df.iterrows():
        print(index)
        fout = "tmp/pages_dump_fb/" + index + '.html'
        url = row['url']+'/about'
        if not 'facebook.com/groups/' in url:
            continue

        if os.path.isfile(fout):
            print('file found, skip')
            continue
        # get facebook page
        print(url)
        web,html = get_fb_page(web, url)
        write_file(html,fout)
        #print(fout)


def scrape_google_selenium(df):
    print("Scrape data with Selenium")

    web = init_google_browser()


    html = run_google_query(web, "Brading Roman Villa")
    random_sleep(2,5)
    write_file(html, 'test.html')
    return
    
    # load input     
    topics = pd.read_csv('data/input/london_placenames-v2.csv')
    print('>> Topics to scrape:',len(topics))
    

    if True: # step 1
        # scrape Google
        scrape_google_london_place_groups(topics)
    if False: # step 2
        # extract Facebook data from Google pages
        extract_google_results("tmp/pages_dump")
    if False: # step 3
        # scrape Facebook groups
        scrape_facebook_groups_info("tmp/fb_groups_from_google.csv")
    if False: # step 4
        # extract Facebook group info
        extract_fbgroup_info("tmp/pages_dump_fb")
    if False: # step 5
        # analyse Facebook groups
        analyse_facebook_groups_info("tmp/fb_groups_info_df.pik")
    
    # ==== the rest of the analysis is done in R ==== 
