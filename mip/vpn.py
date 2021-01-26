# -*- coding: utf-8 -*-

""" 
VPN functions 
https://www.privateinternetaccess.com/helpdesk/kb/articles/pia-desktop-command-line-interface

TODO: write instructions for VPN set up here
"""

import logging
logger = logging.getLogger(__name__)

vpn_servers = ['uk-manchester','uk-london','uk-southampton','ireland','belgium',
                'isle-of-man','luxembourg','austria']

vpn_os = 'mac' # 'win'

assert VPN_OS in ['mac','win']

# %% Mac functions
def vpn_off():
    if vpn_os == 'mac':
        ret = run_os_command('piactl disconnect')
        print(ret)

def vpn_on():
    if vpn_os == 'mac':
        ret = run_os_command('piactl connect')
        print(ret)

def vpn_go_region(reg):
    assert reg in VPN_SERVERS
    if vpn_os == 'mac':
        ret = run_os_command('piactl set region '+reg)
    print(">> vpn_go_region:",reg)
    return ret

def get_url_vpn(url):
    pia_socks5 = 'socks5h://XXXX@proxy-nl.privateinternetaccess.com:1080'
    proxies = {'http': pia_socks5,'https': pia_socks5}
    r = requests.get(url, proxies=proxies )
    if r.status_code == 429: 
        raise Exception("429 Too Many Requests")
    assert r.status_code == 200, 'failed to download '+str(r.status_code) + ' ' + url
    return r.text

def vpn_is_on():
    if vpn_os == 'mac':
        ret = run_os_command('piactl get connectionstate') == 'Connected'
        return ret

# %% Windows functions
def vpn_random_region():
    vpn_go_region(random.choice(vpn_servers))

def run_os_command(cmd):
    if vpn_os == 'mac':
        ret = subprocess.check_output(cmd, shell=True)
        ret = ret.decode("utf-8").strip()
        return ret


# TODO: add commands for PIA on Windows

